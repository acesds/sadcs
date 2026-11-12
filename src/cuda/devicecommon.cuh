/*
 * devicecommon.cuh - What the phases of the GPU construction share.

 * Included only by gpusa.cu: the phases are one translation unit, so the __constant__ tables
 * need no relocatable device code.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "gpusa.h"

#define GPUSA_TRY(call)                       \
    do {                                      \
        const cudaError_t tryError_ = (call); \
        if (tryError_ != cudaSuccess) {       \
            return tryError_;                 \
        }                                     \
    } while (0)

namespace gpusa {
    constexpr int kCoverPeriod = 32;
    constexpr int kCoverSize = GpuCover::kSize;
    constexpr int kClasses = GpuCover::kNextSampleDistanceCount;  // d = 0 (the sample) to 6
    constexpr int kThreads = 256;

    static_assert(kCoverPeriod == 32 && kCoverSize == 7 && kClasses == 7, "the tables below assume the cover modulo 32");

    // Text words past the last one holding text, zeroed: 8-byte loads at up to n + 31 stay inside.
    constexpr size_t kTextPaddingWords = 8;

    // Rank entries past the last sample, zeroed: a window of kCoverSize ranks at any position <= n.
    constexpr size_t kRankPadding = 8;

    __constant__ uint8_t cBelow[kCoverPeriod];                     // cover residues below r
    __constant__ uint8_t cResidue[kCoverSize];                     // the cover
    __constant__ uint8_t cTieOffset[kCoverPeriod][kCoverPeriod];   // smallest k with a + k, b + k sampled
    __constant__ uint8_t cWindowSlot[kCoverPeriod][kCoverPeriod];  // [a][k]: rank of a + k in a's window
    __constant__ uint8_t cClassResidue[kClasses][kCoverPeriod];    // residues with distance d, ascending
    __constant__ uint8_t cClassResidueCount[kClasses];
    __constant__ uint8_t cClassOf[kCoverPeriod];                   // distance d of residue r

    // Host copies of the tables, for sizes computed on the host.
    struct CoverTables {
        uint8_t below[kCoverPeriod];
        uint8_t residue[kCoverSize];
        uint8_t tieOffset[kCoverPeriod][kCoverPeriod];
        uint8_t windowSlot[kCoverPeriod][kCoverPeriod];
        uint8_t classResidue[kClasses][kCoverPeriod];
        uint8_t classResidueCount[kClasses];
        uint8_t classOf[kCoverPeriod];

        CoverTables() : below{}, residue{}, tieOffset{}, windowSlot{}, classResidue{}, classResidueCount{}, classOf{} {
            for (int r = 0, k = 0; r < kCoverPeriod; r++) {
                below[r] = static_cast<uint8_t>(GpuCover::pack(r));

                if (GpuCover::isSample(r)) {
                    residue[k++] = static_cast<uint8_t>(r);
                }

                const int d = GpuCover::nextSampleDistance(r);

                classResidue[d][classResidueCount[d]++] = static_cast<uint8_t>(r);
                classOf[r] = static_cast<uint8_t>(d);
            }

            for (int a = 0; a < kCoverPeriod; a++) {
                for (int b = 0; b < kCoverPeriod; b++) {
                    int k = 0;

                    while (!GpuCover::isSample(a + k) || !GpuCover::isSample(b + k)) {
                        k++;
                    }

                    tieOffset[a][b] = static_cast<uint8_t>(k);
                }

                for (int k = 0; k < kCoverPeriod; k++) {
                    windowSlot[a][k] = static_cast<uint8_t>(GpuCover::pack(a + k) - GpuCover::pack(a));
                }
            }
        }

        cudaError_t upload() const {
            GPUSA_TRY(cudaMemcpyToSymbol(cBelow, below, sizeof below));
            GPUSA_TRY(cudaMemcpyToSymbol(cResidue, residue, sizeof residue));
            GPUSA_TRY(cudaMemcpyToSymbol(cTieOffset, tieOffset, sizeof tieOffset));
            GPUSA_TRY(cudaMemcpyToSymbol(cWindowSlot, windowSlot, sizeof windowSlot));
            GPUSA_TRY(cudaMemcpyToSymbol(cClassResidue, classResidue, sizeof classResidue));
            GPUSA_TRY(cudaMemcpyToSymbol(cClassResidueCount, classResidueCount, sizeof classResidueCount));
            return cudaMemcpyToSymbol(cClassOf, classOf, sizeof classOf);
        }

        // Number of positions below n in class d.
        int64_t classSize(int32_t n, int d) const {
            int64_t size = 0;

            for (int k = 0; k < classResidueCount[d]; k++) {
                const int r = classResidue[d][k];

                size += n / kCoverPeriod + (r < n % kCoverPeriod ? 1 : 0);
            }

            return size;
        }
    };

    // Random 4-byte writes into a large array cost a DRAM sector each.
    // Above this many, they're first ordered by the top kPartitionBits of their target index (one radix pass),
    // so each bucket's writes fall into a window the L2 combines into full sectors.
    constexpr uint32_t kPartitionThreshold = 1u << 22;
    constexpr int kPartitionBits = 8;

    inline size_t textWords(int32_t n) {
        return static_cast<size_t>(n) / 8 + kTextPaddingWords;
    }

    inline size_t rankEntries(int32_t n) {
        return static_cast<size_t>(gpuSampleSize(n)) + kRankPadding;
    }

    inline unsigned blocksFor(uint64_t count) {
        return static_cast<unsigned>((count + kThreads - 1) / kThreads);
    }

    // Index of the first sample position >= p in the compact rank array; for a sample position its own entry.
    __device__ __forceinline__ uint32_t compactIndex(uint32_t p) {
        return (p / kCoverPeriod) * kCoverSize + cBelow[p % kCoverPeriod];
    }

    __device__ __forceinline__ uint64_t byteSwap64(uint64_t x) {
        return (static_cast<uint64_t>(__byte_perm(static_cast<uint32_t>(x), 0, 0x0123)) << 32) |
               __byte_perm(static_cast<uint32_t>(x >> 32), 0, 0x0123);
    }

    // The 8 text bytes from p as a big-endian word, zero past the end of the text.
    __device__ __forceinline__ uint64_t load8(const uint64_t* __restrict__ words, uint32_t p) {
        const uint64_t lo = words[p / 8], hi = words[p / 8 + 1];
        const uint32_t shift = (p % 8) * 8;
        return byteSwap64(shift ? (lo >> shift) | (hi << (64 - shift)) : lo);
    }

    // Device allocations of one construction.
    // Tracks the bytes in use and their peak; frees what's left on destruction.
    class DeviceMemory {
     public:
        // With a pool, allocations come from it in stream order on the default stream,
        // so freeing costs nothing and the memory stays reserved for the next construction.
        explicit DeviceMemory(cudaMemPool_t pool = nullptr) : m_pool(pool) {}
        DeviceMemory(const DeviceMemory&) = delete;
        DeviceMemory& operator=(const DeviceMemory&) = delete;

        ~DeviceMemory() {
            for (const Block& block : m_blocks) {
                free(block.pointer);
            }
        }

        template <class T>
        cudaError_t allocate(T*& pointer, size_t count) {
            const size_t bytes = std::max<size_t>(count * sizeof(T), 1);
            void* raw = nullptr;

            GPUSA_TRY(m_pool ? cudaMallocFromPoolAsync(&raw, bytes, m_pool, 0) : cudaMalloc(&raw, bytes));
            pointer = static_cast<T*>(raw);
            m_blocks.push_back({raw, bytes});
            m_current += static_cast<int64_t>(bytes);
            m_peak = std::max(m_peak, m_current);

            return cudaSuccess;
        }

        template <class T>
        void release(T*& pointer) {
            if (!pointer) {
                return;
            }

            for (size_t i = 0; i < m_blocks.size(); i++) {
                if (m_blocks[i].pointer == pointer) {
                    free(pointer);
                    m_current -= static_cast<int64_t>(m_blocks[i].bytes);
                    m_blocks.erase(m_blocks.begin() + static_cast<std::ptrdiff_t>(i));
                    break;
                }
            }

            pointer = nullptr;
        }

        int64_t current() const {
            return m_current;
        }

        int64_t peak() const {
            return m_peak;
        }

        void resetPeak() {
            m_peak = m_current;
        }

     private:
        struct Block {
            void* pointer;
            size_t bytes;
        };

        cudaMemPool_t m_pool = nullptr;
        std::vector<Block> m_blocks;
        int64_t m_current = 0;
        int64_t m_peak = 0;

        void free(void* pointer) const {
            if (m_pool) {
                cudaFreeAsync(pointer, 0);
            } else {
                cudaFree(pointer);
            }
        }
    };

    // What the phases share: the text and the sample ranks stay on the device from the upload to the end of the merge.
    struct Construction {
        explicit Construction(cudaMemPool_t pool = nullptr) : memory(pool) {}

        DeviceMemory memory;
        CoverTables tables;
        uint32_t n = 0;
        uint32_t m = 0;              // sample size
        uint64_t* text = nullptr;    // textWords(n) words, zero past the text
        uint32_t* ranks = nullptr;   // rankEntries(n) entries, zero past the sample
    };
}
