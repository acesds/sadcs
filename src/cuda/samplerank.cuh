/*
 * samplerank.cuh - Step 2 of docs/gpu-design.md: ranking the sample suffixes on the device.
 * The unresolved sample suffixes are kept as a list of positions.
 * Each suffix's group id is the number of sample suffixes known to be smaller than every member of its group
 * (the position of the group's head in the final order).
 * It's stored for every sample position in the rank array, which ends up holding the ranks.
 * A pass sorts the list by a subkey, then stably by group id, and splits each group where the subkey changes;
 * a new group's id is its old id plus its offset in the old group.
 * Groups of one are final and leave the list, so the list always holds whole groups.
 * Character passes use the next 7 characters (suffixKey(), as on the CPU) after the first pass.
 * The first pass, where everything is in one group and no group sort follows,
 * takes kWideChars at once with two sorts:
 * by characters 8 and on with the remaining length (capped at kWideChars) in 4 bits,
 * then by characters 0 to 7 (64 bits, zero past the end, no tag:
 * the padding can only tie a shorter suffix with a longer one, never order it after, and the tag tells them apart).
 * Every group then shares its first kWideChars + 21 characters.
 * Doubling rounds then use the group id at i + h for h = 32, 64, ...: i + h is again a sample position,
 * and h never exceeds the prefix length the groups already share, so each round extends it by h.
 * The ids read in a round are all from the round before (Larsson-Sadakane without the in-round refinement).
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cub/cub.cuh>
#include <cuda/functional>
#include "devicecommon.cuh"

namespace gpusa {
    constexpr int kKeyChars = 7;
    constexpr int kCharacterPasses = 4;  // the wide first pass and 3 of 7 characters
    constexpr int kWideHead = 8;         // the first pass: characters of the major key
    constexpr int kWideTail = 7;         // and of the minor key, followed by a 4-bit length tag
    constexpr int kWideChars = kWideHead + kWideTail;
    constexpr int kWideTailBits = 8 * kWideTail + 4;

    static_assert(kWideChars + (kCharacterPasses - 1) * kKeyChars >= kCoverPeriod,
        "doubling needs groups sharing v characters");
    static_assert(kWideChars < 16 && kWideTail < 8, "the length tag has 4 bits");

    // The next 7 characters at p, big-endian in the top 56 bits.
    // The remaining length (capped at 7) is stored in the low byte, as the CPU's suffixKey().
    // Returns 0 at or past the end.
    __device__ __forceinline__ uint64_t suffixKey(const uint64_t* __restrict__ words, uint32_t n, uint32_t p) {
        if (p >= n) {
            return 0;
        }

        const uint32_t remaining = n - p;
        const uint64_t characters = load8(words, p) & ~uint64_t(0xFF);

        if (remaining >= kKeyChars) {
            return characters | kKeyChars;
        }

        return (characters & (~uint64_t(0) << (64 - 8 * remaining))) | remaining;
    }

    // The first pass's keys for a sample position p < n: characters 0 to 7, zero past the end...
    __device__ __forceinline__ uint64_t wideHeadKey(const uint64_t* __restrict__ words, uint32_t n, uint32_t p) {
        const uint32_t remaining = n - p;
        const uint64_t characters = load8(words, p);
        return remaining >= kWideHead ? characters : characters & (~uint64_t(0) << (64 - 8 * remaining));
    }

    // ...and the next kWideTail characters, zero past the end, then min(n - p, kWideChars) in 4 bits.
    __device__ __forceinline__ uint64_t wideTailKey(const uint64_t* __restrict__ words, uint32_t n, uint32_t p) {
        const uint32_t remaining = n - p;
        uint64_t characters = 0;

        if (remaining > kWideHead) {
            characters = load8(words, p + kWideHead) >> (64 - 8 * kWideTail);

            if (remaining < kWideChars) {
                characters &= ~uint64_t(0) << (8 * (kWideChars - remaining));
            }
        }

        return (characters << 4) | min(remaining, static_cast<uint32_t>(kWideChars));
    }

    struct PassInput {
        const uint64_t* text;
        const uint32_t* ranks;
        uint32_t n;
        uint32_t offset;  // character depth, or the doubling distance h
        bool doubling;
    };

    // Character pass: suffixKey at p + depth.
    // Doubling: 1 + the group id at p + h, 0 past the end.
    __device__ __forceinline__ uint64_t subkey(const PassInput& in, uint32_t p) {
        if (in.doubling) {
            const uint32_t q = p + in.offset;
            return q < in.n ? uint64_t(in.ranks[compactIndex(q)]) + 1 : 0;
        }

        return suffixKey(in.text, in.n, p + in.offset);
    }

    __global__ void initSamplePositions(uint32_t* __restrict__ positions, uint32_t m) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < m) {
            positions[t] = (t / kCoverSize) * kCoverPeriod + cResidue[t % kCoverSize];
        }
    }

    // Key is uint64_t for the character passes, uint32_t for the doubling rounds (ranks + 1).
    template <class Key>
    __global__ void makeSubkeys(PassInput in, const uint32_t* __restrict__ positions, Key* __restrict__ keys, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < u) {
            keys[t] = static_cast<Key>(subkey(in, positions[t]));
        }
    }

    __global__ void makeWideKeys(PassInput in, bool tail, const uint32_t* __restrict__ positions, uint64_t* __restrict__ keys, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < u) {
            const uint32_t p = positions[t];

            keys[t] = tail ? wideTailKey(in.text, in.n, p) : wideHeadKey(in.text, in.n, p);
        }
    }

    __global__ void gatherGroups(const uint32_t* __restrict__ ranks, const uint32_t* __restrict__ positions, uint32_t* __restrict__ groups, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < u) {
            groups[t] = ranks[compactIndex(positions[t])];
        }
    }

    // t at the first element of each group, else 0: a max-scan gives each element its group's start.
    __global__ void markGroupStarts(const uint32_t* __restrict__ groups, uint32_t* __restrict__ out, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < u) {
            out[t] = t == 0 || groups[t] != groups[t - 1] ? t : 0;
        }
    }

    // At the first element of each new group its id, the old id plus the offset in the old group, else 0.
    // The ids grow with t, so a max-scan hands each element its new group's id.
    // (The first pass, where everything is in group 0, uses markWideGroups() instead.)
    // sortedKeys, if not null, holds the subkeys in list order (the grouped doubling rounds).
    // Otherwise each thread reads its own subkey and takes its predecessor's from the lane before;
    // only lane 0 reads a second one.
    // Every thread of a warp must reach the shuffles, so the range check comes after them.
    template <class Key>
    __global__ void markNewGroups(PassInput in, const Key* __restrict__ sortedKeys, const uint32_t* __restrict__ positions, const uint32_t* __restrict__ groups, const uint32_t* __restrict__ groupStarts, uint32_t* __restrict__ out, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
        const bool active = t < u;

        const uint32_t p = active ? positions[t] : 0;
        const uint64_t key = !active ? 0 : sortedKeys ? uint64_t(sortedKeys[t]) : subkey(in, p);
        uint64_t previous = __shfl_up_sync(0xFFFFFFFFu, key, 1);

        if (!active) {
            return;
        }

        const uint32_t group = groups[t];
        bool head = t == 0;

        if (!head) {
            if ((threadIdx.x & 31) == 0) {
                previous = sortedKeys ? uint64_t(sortedKeys[t - 1]) : subkey(in, positions[t - 1]);
            }

            head = groups[t - 1] != group || previous != key;
        }

        out[t] = head ? group + t - groupStarts[t] : 0;
    }

    // markNewGroups() for the first pass, whose sorted head keys are in list order.
    // The tail key, a random read of the text, is only needed where the head ties with a neighbour.
    __global__ void markWideGroups(PassInput in, const uint64_t* __restrict__ sortedKeys, const uint32_t* __restrict__ positions, uint32_t* __restrict__ out, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
        const bool active = t < u;
        const uint64_t key = active ? sortedKeys[t] : 0;
        const bool tiedBefore = active && t > 0 && sortedKeys[t - 1] == key;
        const bool tiedAfter = active && t + 1 < u && sortedKeys[t + 1] == key;
        const uint64_t key2 = tiedBefore || tiedAfter ? wideTailKey(in.text, in.n, positions[t]) : 0;
        uint64_t previous2 = __shfl_up_sync(0xFFFFFFFFu, key2, 1);

        if (!active) {
            return;
        }

        bool head = !tiedBefore;

        if (!head) {
            if ((threadIdx.x & 31) == 0) {
                previous2 = wideTailKey(in.text, in.n, positions[t - 1]);
            }

            head = previous2 != key2;
        }

        out[t] = head ? t : 0;
    }

    // Grouped doubling rounds: t at the first element of each group, else ~0.
    // The list's length is *count, at most bound.
    __global__ void markSegmentHeads(const uint32_t* __restrict__ groups, uint32_t* __restrict__ out, const int* __restrict__ count, uint32_t bound) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t u = static_cast<uint32_t>(*count);

        if (t < bound) {
            out[t] = t < u && (t == 0 || groups[t] != groups[t - 1]) ? t : ~0u;
        }
    }

    struct IsSegmentHead {
        __device__ bool operator()(uint32_t x) const {
            return x != ~0u;
        }
    };

    // offsets[count] = u, and the largest segment into *largest (zeroed before).
    __global__ void finishSegments(uint32_t* __restrict__ offsets, const int* __restrict__ count, const int* __restrict__ length, int* __restrict__ largest) {
        const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t segments = static_cast<uint32_t>(*count);
        const uint32_t u = static_cast<uint32_t>(*length);

        if (s < segments) {
            const uint32_t end = s + 1 < segments ? offsets[s + 1] : u;

            atomicMax(largest, static_cast<int>(end - offsets[s]));
        }

        if (s == 0) {
            offsets[segments] = u;
        }
    }

    // Stores the new group ids and flags the elements still tied with a neighbour.
    __global__ void storeGroups(uint32_t* __restrict__ ranks, const uint32_t* __restrict__ positions, const uint32_t* __restrict__ ids, uint8_t* __restrict__ unresolved, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < u) {
            const uint32_t id = ids[t];

            ranks[compactIndex(positions[t])] = id;
            unresolved[t] = (t > 0 && ids[t - 1] == id) || (t + 1 < u && ids[t + 1] == id);
        }
    }

    // Large passes: the ids with the unresolved flag in bit 31, in list order,
    // for the partition by position before they're stored.
    __global__ void flagIds(const uint32_t* __restrict__ ids, uint32_t* __restrict__ flagged, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < u) {
            const uint32_t id = ids[t];
            const bool tied = (t > 0 && ids[t - 1] == id) || (t + 1 < u && ids[t + 1] == id);

            flagged[t] = id | (tied ? 0x80000000u : 0);
        }
    }

    // storeGroups() for the partitioned list.
    __global__ void storeFlaggedGroups(uint32_t* __restrict__ ranks, const uint32_t* __restrict__ positions, const uint32_t* __restrict__ flagged, uint8_t* __restrict__ unresolved, uint32_t u) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < u) {
            const uint32_t v = flagged[t];

            ranks[compactIndex(positions[t])] = v & 0x7FFFFFFFu;
            unresolved[t] = static_cast<uint8_t>(v >> 31);
        }
    }

    // The working buffers of the ranking, for m sample suffixes:
    // the position list double-buffered (8m) and the 64-bit sort keys double-buffered,
    // which also hold the 32-bit group ids, scan results and flags; plus CUB's scratch.
    // A pass leaves the list in group order, and compacting it keeps that order.
    // Unless the partitioned store reordered it, a pass before a doubling round on at least
    // kGroupedMin elements also compacts the group ids and finds the groups' offsets, and the
    // round only has to sort each group by its subkey: one segmented sort instead of two radix
    // sorts and a gather of the group ids, which stay valid since a segment has one id.
    // Not if a group is over kMaxSegment, though: CUB sorts a large segment in one block.
    class SampleRanking {
     public:
        explicit SampleRanking(Construction& c) : m_construction(c) {}

        ~SampleRanking() {
            release();
        }

        // Device bytes beyond the text and the ranks.
        static cudaError_t workingBytes(uint32_t m, int64_t& bytes) {
            size_t temp = 0;

            GPUSA_TRY(tempBytes(m, temp));
            bytes = static_cast<int64_t>(m) * (2 * 4 + 2 * 8) + static_cast<int64_t>(temp) +
                    static_cast<int64_t>(3 * sizeof(int));

            return cudaSuccess;
        }

        // Ranks the sample into c.ranks.
        cudaError_t run(GpuStats& stats) {
            const uint32_t m = m_construction.m;

            GPUSA_TRY(tempBytes(m, m_tempBytes));
            GPUSA_TRY(m_construction.memory.allocate(m_positions[0], m));
            GPUSA_TRY(m_construction.memory.allocate(m_positions[1], m));
            GPUSA_TRY(m_construction.memory.allocate(m_keys[0], m));
            GPUSA_TRY(m_construction.memory.allocate(m_keys[1], m));
            GPUSA_TRY(m_construction.memory.allocate(m_temp, m_tempBytes));
            GPUSA_TRY(m_construction.memory.allocate(m_selectedCount, 3));
            m_groups = nullptr;

            int current = 0;
            uint32_t u = m;
            PassInput in{m_construction.text, m_construction.ranks, m_construction.n, 0, false};

            initSamplePositions<<<blocksFor(m), kThreads>>>(m_positions[0], m);

            for (int pass = 0; pass < kCharacterPasses && u > 0; pass++) {
                in.offset = pass == 0 ? 0 : kWideChars + (pass - 1) * kKeyChars;  // the first pass is wide
                GPUSA_TRY(runPass(in, pass == 0, pass + 1 == kCharacterPasses, current, u));
                stats.characterPasses++;
            }

            in.doubling = true;

            for (uint64_t h = kCoverPeriod; u > 0; h *= 2) {
                // Tied suffixes share more than h characters, so h < n while any are left.
                if (h >= m_construction.n) {
                    return cudaErrorUnknown;
                }

                in.offset = static_cast<uint32_t>(h);
                GPUSA_TRY(runPass(in, false, true, current, u));
                stats.doublingRounds++;
            }

            GPUSA_TRY(cudaGetLastError());
            release();

            return cudaSuccess;
        }

     private:
        Construction& m_construction;
        uint32_t* m_positions[2] = {};
        uint64_t* m_keys[2] = {};
        uint8_t* m_temp = nullptr;
        size_t m_tempBytes = 0;
        int* m_selectedCount = nullptr;  // [0] selected count, [1] largest segment, [2] segments
        uint32_t* m_groups = nullptr;    // the list's group ids, in one of the four key arrays, or null
        uint32_t* m_offsets = nullptr;   // the groups' offsets and u, in another one
        uint32_t m_segments = 0;

        static constexpr int kMaxSegment = 128;
        static constexpr uint32_t kGroupedMin = 1u << 20;

        void release() {
            m_construction.memory.release(m_positions[0]);
            m_construction.memory.release(m_positions[1]);
            m_construction.memory.release(m_keys[0]);
            m_construction.memory.release(m_keys[1]);
            m_construction.memory.release(m_temp);
            m_construction.memory.release(m_selectedCount);
        }

        // CUB's scratch for every call below on up to m items.
        static cudaError_t tempBytes(uint32_t m, size_t& bytes) {
            cub::DoubleBuffer<uint64_t> wideKeys(nullptr, nullptr);
            cub::DoubleBuffer<uint32_t> narrowKeys(nullptr, nullptr), values(nullptr, nullptr);
            size_t sizes[6] = {};

            GPUSA_TRY(cub::DeviceRadixSort::SortPairs(nullptr, sizes[0], wideKeys, values, m));
            GPUSA_TRY(cub::DeviceRadixSort::SortPairs(nullptr, sizes[1], narrowKeys, values, m));
            GPUSA_TRY(cub::DeviceScan::InclusiveScan(nullptr, sizes[2], static_cast<uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr), cuda::maximum<>{}, m));
            GPUSA_TRY(cub::DeviceSelect::Flagged(nullptr, sizes[3], static_cast<uint32_t*>(nullptr),
                static_cast<uint8_t*>(nullptr), static_cast<uint32_t*>(nullptr), static_cast<int*>(nullptr), m));
            GPUSA_TRY(cub::DeviceSelect::If(nullptr, sizes[4], static_cast<uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr), static_cast<int*>(nullptr), m, IsSegmentHead{}));
            GPUSA_TRY(cub::DeviceSegmentedSort::StableSortPairs(nullptr, sizes[5], static_cast<const uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr), static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
                static_cast<int64_t>(m), static_cast<int64_t>(m / 2 + 1), static_cast<const uint32_t*>(nullptr),
                static_cast<const uint32_t*>(nullptr)));
            bytes = *std::max_element(sizes, sizes + 6);

            return cudaSuccess;
        }

        // One pass over the u unresolved suffixes in m_positions[current]: updates the ranks,
        // and leaves the ones still unresolved in m_positions[current] (which may flip) and their number in u.
        // beforeDoubling: the next pass is a doubling round.
        cudaError_t runPass(const PassInput& in, bool first, bool beforeDoubling, int& current, uint32_t& u) {
            if (m_groups) {
                return groupedRound(in, current, u);
            }

            const uint32_t m = m_construction.m;
            const int groupBits = std::max(1, static_cast<int>(std::bit_width(m - 1)));
            const int subkeyBits = in.doubling ? static_cast<int>(std::bit_width(m)) : 64;
            size_t bytes = m_tempBytes;

            // The first pass is wide: an LSD sort by the subkey at 8, then by the one at 0.
            const bool wide = first;
            cub::DoubleBuffer<uint64_t> subkeys(m_keys[0], m_keys[1]);

            if (in.doubling) {
                // Ranks + 1 fit in 32 bits: a third less to move per radix pass.
                cub::DoubleBuffer<uint32_t> narrow(
                    reinterpret_cast<uint32_t*>(m_keys[0]), reinterpret_cast<uint32_t*>(m_keys[1]));
                cub::DoubleBuffer<uint32_t> bySubkey(m_positions[current], m_positions[current ^ 1]);

                makeSubkeys<<<blocksFor(u), kThreads>>>(in, m_positions[current], narrow.Current(), u);
                GPUSA_TRY(cub::DeviceRadixSort::SortPairs(m_temp, bytes, narrow, bySubkey, u, 0, subkeyBits));
                current = bySubkey.selector ? current ^ 1 : current;
            }

            for (int k = wide ? 1 : 0; k >= 0 && !in.doubling; k--) {
                if (wide) {
                    makeWideKeys<<<blocksFor(u), kThreads>>>(in, k == 1, m_positions[current], subkeys.Current(), u);
                } else {
                    makeSubkeys<<<blocksFor(u), kThreads>>>(in, m_positions[current], subkeys.Current(), u);
                }

                cub::DoubleBuffer<uint32_t> bySubkey(m_positions[current], m_positions[current ^ 1]);

                bytes = m_tempBytes;
                GPUSA_TRY(cub::DeviceRadixSort::SortPairs(
                    m_temp, bytes, subkeys, bySubkey, u, 0, wide && k == 1 ? kWideTailBits : subkeyBits));
                current = bySubkey.selector ? current ^ 1 : current;
            }

            // Apart from the first pass's, which are in list order, the subkeys are recomputed where needed,
            // so both key buffers are free now: four arrays of m uint32.
            uint32_t* const groupBuffers[2] = {
                reinterpret_cast<uint32_t*>(m_keys[0]), reinterpret_cast<uint32_t*>(m_keys[0]) + m};
            uint32_t* const scratch[2] = {
                reinterpret_cast<uint32_t*>(m_keys[1]), reinterpret_cast<uint32_t*>(m_keys[1]) + m};
            uint32_t* ids;
            uint8_t* unresolved = reinterpret_cast<uint8_t*>(scratch[0]);
            uint32_t* heads = scratch[0];
            uint32_t* spare = scratch[1];   // free once the ids are made
            uint32_t* groupsOut = nullptr;  // where the compacted group ids go, if the order is kept

            if (first) {
                // The sorted subkeys stay in Current(); the head marks and ids go to Alternate(),
                // and the unresolved flags over the subkeys once they're used.
                uint32_t* const alternate = reinterpret_cast<uint32_t*>(subkeys.Alternate());

                heads = alternate;
                spare = alternate;
                markWideGroups<<<blocksFor(u), kThreads>>>(in, subkeys.Current(), m_positions[current], heads, u);
                groupsOut = alternate;  // the head marks, free after the scan
                ids = alternate + m;
                unresolved = reinterpret_cast<uint8_t*>(subkeys.Current());
            } else {
                gatherGroups<<<blocksFor(u), kThreads>>>(in.ranks, m_positions[current], groupBuffers[0], u);
                cub::DoubleBuffer<uint32_t> groups(groupBuffers[0], groupBuffers[1]);
                cub::DoubleBuffer<uint32_t> byGroup(m_positions[current], m_positions[current ^ 1]);

                bytes = m_tempBytes;
                GPUSA_TRY(cub::DeviceRadixSort::SortPairs(m_temp, bytes, groups, byGroup, u, 0, groupBits));
                current = byGroup.selector ? current ^ 1 : current;

                markGroupStarts<<<blocksFor(u), kThreads>>>(groups.Current(), scratch[0], u);
                bytes = m_tempBytes;
                GPUSA_TRY(cub::DeviceScan::InclusiveScan(m_temp, bytes, scratch[0], scratch[1], cuda::maximum<>{}, u));
                markNewGroups<<<blocksFor(u), kThreads>>>(in, static_cast<const uint64_t*>(nullptr),
                    m_positions[current], groups.Current(), scratch[1], heads, u);
                ids = groups.Alternate();
                groupsOut = scratch[1];  // the group starts, free after markNewGroups
            }

            bytes = m_tempBytes;
            GPUSA_TRY(cub::DeviceScan::InclusiveScan(m_temp, bytes, heads, ids, cuda::maximum<>{}, u));

            return storeAndCompact(current, u, ids, unresolved, spare, beforeDoubling ? groupsOut : nullptr);
        }

        // A doubling round on a list in group order with its ids in m_groups and the offsets of its groups in m_offsets.
        cudaError_t groupedRound(const PassInput& in, int& current, uint32_t& u) {
            uint32_t* const arrays[4] = {
                reinterpret_cast<uint32_t*>(m_keys[0]), reinterpret_cast<uint32_t*>(m_keys[0]) + m_construction.m,
                reinterpret_cast<uint32_t*>(m_keys[1]), reinterpret_cast<uint32_t*>(m_keys[1]) + m_construction.m};
            uint32_t* const groups = m_groups;
            uint32_t* const y = m_offsets;
            uint32_t* free[2];

            for (int a = 0, f = 0; a < 4; a++) {
                if (arrays[a] != groups && arrays[a] != y) {
                    free[f++] = arrays[a];
                }
            }

            uint32_t* const x = free[0];
            uint32_t* const z = free[1];
            size_t bytes = m_tempBytes;

            // Subkeys into x, sorted within the segments into z, positions along.
            makeSubkeys<<<blocksFor(u), kThreads>>>(in, m_positions[current], x, u);
            GPUSA_TRY(cub::DeviceSegmentedSort::StableSortPairs(m_temp, bytes, x, z, m_positions[current],
                m_positions[current ^ 1], static_cast<int64_t>(u), static_cast<int64_t>(m_segments), y, y + 1));
            current ^= 1;

            // Group starts into y (offsets used), head marks into x, ids into y.
            markGroupStarts<<<blocksFor(u), kThreads>>>(groups, x, u);
            bytes = m_tempBytes;
            GPUSA_TRY(cub::DeviceScan::InclusiveScan(m_temp, bytes, x, y, cuda::maximum<>{}, u));
            markNewGroups<<<blocksFor(u), kThreads>>>(in, z, m_positions[current], groups, y, x, u);
            bytes = m_tempBytes;
            GPUSA_TRY(cub::DeviceScan::InclusiveScan(m_temp, bytes, x, y, cuda::maximum<>{}, u));

            // Unresolved flags into x (head marks used); the next ids over z (subkeys used),
            // or the partitioned store's spare.
            const bool partitioned = u >= kPartitionThreshold;

            return storeAndCompact(current, u, y, reinterpret_cast<uint8_t*>(partitioned ? z : x), x,
                partitioned ? nullptr : z);
        }

        // Stores the ids of the u elements of m_positions[current] and keeps the unresolved ones,
        // with their ids in groupsOut unless the store partitions the list (or groupsOut is null),
        // and then their groups' offsets over unresolved.
        cudaError_t storeAndCompact(int& current, uint32_t& u, uint32_t* ids, uint8_t* unresolved, uint32_t* spare, uint32_t* groupsOut) {
            size_t bytes = m_tempBytes;

            m_groups = nullptr;
            groupsOut = u >= kGroupedMin ? groupsOut : nullptr;

            if (u < kPartitionThreshold) {
                storeGroups<<<blocksFor(u), kThreads>>>(m_construction.ranks, m_positions[current], ids, unresolved, u);

                if (groupsOut) {
                    GPUSA_TRY(cub::DeviceSelect::Flagged(m_temp, bytes, ids, unresolved, groupsOut, m_selectedCount, u));
                    bytes = m_tempBytes;
                }
            } else {
                // The list's order doesn't matter to the next pass, which sorts it completely.
                const int high = static_cast<int>(std::bit_width(m_construction.n - 1));
                cub::DoubleBuffer<uint32_t> byPosition(m_positions[current], m_positions[current ^ 1]);
                cub::DoubleBuffer<uint32_t> flagged(spare, ids);

                groupsOut = nullptr;
                flagIds<<<blocksFor(u), kThreads>>>(ids, spare, u);
                bytes = m_tempBytes;
                GPUSA_TRY(cub::DeviceRadixSort::SortPairs(
                    m_temp, bytes, byPosition, flagged, u, std::max(0, high - kPartitionBits), high));
                current = byPosition.selector ? current ^ 1 : current;
                storeFlaggedGroups<<<blocksFor(u), kThreads>>>(
                    m_construction.ranks, m_positions[current], flagged.Current(), unresolved, u);
            }

            bytes = m_tempBytes;
            GPUSA_TRY(cub::DeviceSelect::Flagged(
                m_temp, bytes, m_positions[current], unresolved, m_positions[current ^ 1], m_selectedCount, u));
            current ^= 1;

            if (groupsOut) {
                // Offsets over unresolved (used), via head marks over ids (compacted).
                uint32_t* const offsets = reinterpret_cast<uint32_t*>(unresolved);

                markSegmentHeads<<<blocksFor(u), kThreads>>>(groupsOut, ids, m_selectedCount, u);
                GPUSA_TRY(cudaMemsetAsync(m_selectedCount + 1, 0, sizeof(int)));
                bytes = m_tempBytes;
                GPUSA_TRY(cub::DeviceSelect::If(m_temp, bytes, ids, offsets, m_selectedCount + 2, u, IsSegmentHead{}));
                finishSegments<<<blocksFor(u / 2 + 1), kThreads>>>(
                    offsets, m_selectedCount + 2, m_selectedCount, m_selectedCount + 1);
            }

            int counts[3] = {};

            GPUSA_TRY(cudaMemcpy(counts, m_selectedCount, (groupsOut ? 3 : 1) * sizeof(int), cudaMemcpyDeviceToHost));
            u = static_cast<uint32_t>(counts[0]);

            if (groupsOut && u > 0 && counts[1] <= kMaxSegment) {
                m_groups = groupsOut;
                m_offsets = reinterpret_cast<uint32_t*>(unresolved);
                m_segments = static_cast<uint32_t>(counts[2]);
            }

            return cudaSuccess;
        }
    };
}
