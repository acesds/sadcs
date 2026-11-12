/*
 * classsort.cuh - The sorted list of each class.

 * Class d contains the positions whose next sample position is d characters ahead.
 * For every position i in class d>=1, i+1 belongs to class d-1. Thus suffix i is ordered by (T[i],suffix i+1),
 * and class d-1 already provides the secondary order. Each class is therefore derived from the previous one:
 * Walk class d-1 in order, keep the positions whose predecessor belongs to class d, and stably radix-sort those
 * predecessors by their character T[i]. A single 8-bit pass is enough. The sample (class 0) is already ordered
 * by its ranks and only needs to be scattered.
 * The sort key for q packs T[q], T[q-1], T[q-2] and T[q-3] from the most significant byte down.
 * Only the top byte is sorted. The key for q-1 in the next class is this key shifted left by one byte.
 * Consequently, classes 2 to 4 and 6 obtain their characters from the previous class's sorted keys, read in order;
 * only classes 1 and 5 need random text reads.
 * Position n-1 has no successor in any list! Its suffix consists of a single character and is therefore a prefix
 * of every longer suffix beginning with T[n-1]. It must come first among those suffixes. To achieve this, walk
 * class d-1 with n prepended, as if the empty suffix n were a member of that class. The stable sort then keeps
 * n-1 ahead of the other suffixes with the same first character. All classes are scattered into a single array
 * of n positions, one class after another. This array is the input to the merge. The flags, sort keys and second
 * value buffer are stored in the merge output buffer, which is not needed until the merge begins.
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
#include "devicecommon.cuh"

namespace gpusa {
    // Where each class starts in the concatenated lists, and the total n.
    struct ClassLists {
        uint32_t start[kClasses + 1];
    };

    inline ClassLists classLists(const CoverTables& tables, int32_t n) {
        ClassLists lists{};

        for (int d = 0; d < kClasses; d++) {
            lists.start[d + 1] = lists.start[d] + static_cast<uint32_t>(tables.classSize(n, d));
        }

        return lists;
    }

    // The sample positions in rank order.
    __global__ void scatterSample(const uint32_t* __restrict__ ranks, uint32_t m, uint32_t* __restrict__ list) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < m) {
            list[ranks[t]] = (t / kCoverSize) * kCoverPeriod + cResidue[t % kCoverSize];
        }
    }

    // A large sample: the ranks and positions as pairs, to partition by rank before scattering.
    __global__ void samplePairs(const uint32_t* __restrict__ ranks, uint32_t m, uint32_t* __restrict__ keys, uint32_t* __restrict__ positions) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < m) {
            keys[t] = ranks[t];
            positions[t] = (t / kCoverSize) * kCoverPeriod + cResidue[t % kCoverSize];
        }
    }

    __global__ void scatterPairs(const uint32_t* __restrict__ keys, const uint32_t* __restrict__ positions, uint32_t m, uint32_t* __restrict__ list) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < m) {
            list[keys[t]] = positions[t];
        }
    }

    // Element k of class d-1 with n prepended.
    __device__ __forceinline__ uint32_t previousElement(const uint32_t* __restrict__ previous, uint32_t n, uint32_t k) {
        return k == 0 ? n : previous[k - 1];
    }

    __device__ __forceinline__ bool predecessorInClass(uint32_t p, int d) {
        return p > 0 && cClassOf[(p - 1) % kCoverPeriod] == d;
    }

    // T[q], T[q-1], T[q-2], T[q-3] from the top byte down, zero before the text.
    __device__ __forceinline__ uint32_t characterKey(const uint64_t* __restrict__ text, uint32_t q) {
        const uint8_t* const bytes = reinterpret_cast<const uint8_t*>(text);
        uint32_t key = 0;

#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            key = (key << 8) | (q >= i ? bytes[q - i] : 0);
        }

        return key;
    }

    // flags[k] for element k of class d - 1 with n prepended; scanned in place afterwards.
    __global__ void flagPredecessors(const uint32_t* __restrict__ previous, uint32_t count, uint32_t n, int d, uint32_t* __restrict__ flags) {
        const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

        if (k <= count) {
            flags[k] = predecessorInClass(previousElement(previous, n, k), d) ? 1 : 0;
        }
    }

    // The flagged predecessors in order, with their keys.
    // previousKeys (if not null) are the sorted keys of class d-1, whose lower three bytes are valid.
    // Otherwise, the key is read from the text.
    __global__ void gatherPredecessors(const uint64_t* __restrict__ text, const uint32_t* __restrict__ previous, const uint32_t* __restrict__ previousKeys, uint32_t count, uint32_t n, int d, const uint32_t* __restrict__ index, uint32_t* __restrict__ keys, uint32_t* __restrict__ positions) {
        const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

        if (k <= count) {
            const uint32_t p = previousElement(previous, n, k);

            if (predecessorInClass(p, d)) {
                const uint32_t q = p - 1;
                keys[index[k]] = previousKeys && k > 0 ? previousKeys[k - 1] << 8 : characterKey(text, q);
                positions[index[k]] = q;
            }
        }
    }

    class ClassSort {
     public:
        static uint32_t largestClass(const ClassLists& lists) {
            uint32_t largest = 0;

            for (int d = 0; d < kClasses; d++) {
                largest = std::max(largest, lists.start[d + 1] - lists.start[d]);
            }

            return largest;
        }

        static uint32_t largestNonSampleClass(const ClassLists& lists) {
            uint32_t largest = 0;

            for (int d = 1; d < kClasses; d++) {
                largest = std::max(largest, lists.start[d + 1] - lists.start[d]);
            }

            return largest;
        }

        // Scratch in 4-byte words: the previous class's sorted keys, the flags (scanned in place) for
        // the previous class plus one, of which the sample is the largest, then two keys and one value per element.
        // The sample's partition needs four arrays of m words. The merge output buffer is made at least this large.
        static size_t scratchWords(const ClassLists& lists) {
            const size_t m = lists.start[1] - lists.start[0];
            return std::max(4 * static_cast<size_t>(largestNonSampleClass(lists)) + m + 1, 4 * m);
        }

        static cudaError_t tempBytes(const ClassLists& lists, size_t& bytes) {
            const uint32_t largest = largestClass(lists);
            cub::DoubleBuffer<uint32_t> keys(nullptr, nullptr), values(nullptr, nullptr);
            size_t sort = 0, scan = 0;

            GPUSA_TRY(cub::DeviceRadixSort::SortPairs(nullptr, sort, keys, values, largest, 0, 32));
            GPUSA_TRY(cub::DeviceScan::ExclusiveSum(
                nullptr, scan, static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr), largest + 1));
            bytes = std::max(sort, scan);

            return cudaSuccess;
        }

        // Fills lists (n entries) with the sorted classes.
        // scratch holds scratchWords(layout) words.
        static cudaError_t run(Construction& c, const ClassLists& layout, uint32_t* lists, uint32_t* scratch) {
            size_t tempBytes = 0;
            uint8_t* temp = nullptr;

            GPUSA_TRY(ClassSort::tempBytes(layout, tempBytes));
            GPUSA_TRY(c.memory.allocate(temp, tempBytes));

            const cudaError_t error = sortAll(c, layout, lists, scratch, temp, tempBytes);
            c.memory.release(temp);

            return error;
        }

     private:
        static cudaError_t sortAll(Construction& c, const ClassLists& layout, uint32_t* lists, uint32_t* scratch, uint8_t* temp, size_t tempBytes) {
            if (c.m >= kPartitionThreshold) {
                uint32_t* const keys0 = scratch;
                uint32_t* const keys1 = keys0 + c.m;
                uint32_t* const values0 = keys1 + c.m;
                uint32_t* const values1 = values0 + c.m;
                cub::DoubleBuffer<uint32_t> keys(keys0, keys1), values(values0, values1);
                const int high = static_cast<int>(std::bit_width(c.m - 1));
                size_t bytes = tempBytes;

                samplePairs<<<blocksFor(c.m), kThreads>>>(c.ranks, c.m, keys0, values0);
                GPUSA_TRY(cub::DeviceRadixSort::SortPairs(
                    temp, bytes, keys, values, c.m, std::max(0, high - kPartitionBits), high));
                scatterPairs<<<blocksFor(c.m), kThreads>>>(keys.Current(), values.Current(), c.m, lists + layout.start[0]);
            } else if (c.m > 0) {
                scatterSample<<<blocksFor(c.m), kThreads>>>(c.ranks, c.m, lists + layout.start[0]);
            }

            for (int d = 1; d < kClasses; d++) {
                const uint32_t size = layout.start[d + 1] - layout.start[d];

                if (size == 0) {
                    continue;
                }

                const uint32_t* const previous = lists + layout.start[d - 1];
                const uint32_t count = layout.start[d] - layout.start[d - 1];
                uint32_t* const positions = lists + layout.start[d];
                uint32_t* const previousKeys = scratch;
                uint32_t* const index = previousKeys + largestNonSampleClass(layout);
                uint32_t* const keys0 = index + (layout.start[1] - layout.start[0]) + 1;
                uint32_t* const keys1 = keys0 + size;
                uint32_t* const values1 = keys1 + size;
                cub::DoubleBuffer<uint32_t> keys(keys0, keys1);
                cub::DoubleBuffer<uint32_t> values(positions, values1);
                size_t bytes = tempBytes;

                // Classes 1 and 5 read their keys from the text; the others shift the previous ones.
                const bool fromText = d % 4 == 1;

                flagPredecessors<<<blocksFor(count + 1ull), kThreads>>>(previous, count, c.n, d, index);
                GPUSA_TRY(cub::DeviceScan::ExclusiveSum(temp, bytes, index, index, count + 1));
                gatherPredecessors<<<blocksFor(count + 1ull), kThreads>>>(
                    c.text, previous, fromText ? nullptr : previousKeys, count, c.n, d, index, keys0, positions);

                bytes = tempBytes;
                GPUSA_TRY(cub::DeviceRadixSort::SortPairs(temp, bytes, keys, values, size, 24, 32));

                if (values.Current() != positions) {
                    GPUSA_TRY(cudaMemcpyAsync(
                        positions, values.Current(), size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                }

                if (d + 1 < kClasses && (d + 1) % 4 != 1) {
                    GPUSA_TRY(cudaMemcpyAsync(
                        previousKeys, keys.Current(), size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                }
            }

            return cudaGetLastError();
        }
    };
}
