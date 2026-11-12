/*
 * gpusa.h - The GPU variant's interface to the rest of sadcs (see docs/gpu-design.md).
 *
 * This is plain C++, no CUDA types, so the CLI compiles without the toolkit.
 * Builds without CUDA link gpusa_stub.cpp instead of gpusa.cu, where every call reports GpuStatus::Unavailable.
 * gpuSuffixArray() and gpuBwt() build on the device: the sample ranking, the class sorts and the multiway merge.
 * gpuRankSample() runs only the sample ranking, so that the selftest can check it on its own against ranks read
 * off a suffix array.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "differencecover.h"

enum class GpuStatus {
    Ok,
    Unavailable,      // built without CUDA, no device, or no usable driver
    NotEnoughMemory,  // the input doesn't fit in free device memory
    Failed,           // a CUDA call failed; gpuLastError() has the details
};

struct GpuDevice {
    char name[256];
    int computeMajor;
    int computeMinor;
    int64_t totalBytes;
    int64_t freeBytes;
};

struct GpuStats {
    int64_t deviceBytes = 0;        // peak device memory allocated
    int64_t rankingBytes = 0;       // peak of the sample ranking
    int64_t classSortBytes = 0;     // peak of the class sorts
    int64_t mergeBytes = 0;         // peak of the merge
    bool pinned = false;            // the transfers went through pinned staging buffers
    double setupSeconds = 0.0;      // without a context: creating the temporary one
    double uploadSeconds = 0.0;
    double deviceSeconds = 0.0;     // the phases below, including their allocations
    double rankingSeconds = 0.0;
    double classSortSeconds = 0.0;
    double mergeSeconds = 0.0;
    double bwtSeconds = 0.0;        // gpuBwt(): placing the sentinel's row after the merge
    double downloadSeconds = 0.0;
    int32_t characterPasses = 0;    // sample ranking: one pass of 15 characters, then 7 each
    int32_t doublingRounds = 0;     // sample ranking: prefix doubling rounds after those
};

using GpuCover = DifferenceCover<32>;

// Number of sample positions in a text of n bytes: the length of the rank array.
inline int64_t gpuSampleSize(int32_t n) {
    return GpuCover::pack(n);
}

// The sample ranks a suffix array implies:
// ranks[GpuCover::pack(p)] is the number of sample suffixes smaller than the one at sample position p.
// The reference for gpuRankSample().
inline void gpuSampleRanksFromSuffixArray(const int32_t* sa, int32_t n, uint32_t* ranks) {
    uint32_t rank = 0;

    for (int32_t i = 0; i < n; i++) {
        if (GpuCover::isSample(sa[i])) {
            ranks[GpuCover::pack(sa[i])] = rank++;
        }
    }
}

// The first CUDA device and its memory. Unavailable without one.
GpuStatus gpuQueryDevice(GpuDevice& device);

// Peak device memory gpuSuffixArray() needs for n bytes of text, or -1 if it can't be determined.
int64_t gpuRequiredBytes(int32_t n);

// Device memory left unused, so the Windows driver doesn't fall back to host memory when the display or other programs grow.
constexpr int64_t kGpuMemoryReserve = int64_t(256) << 20;

// Device memory and pinned staging buffers for inputs of up to capacity bytes, kept across calls:
// the device memory as a CUDA memory pool, reserved up front.
// Without one, each call creates a temp context: reserving and freeing the device memory, about 18 ms on enwik8 (100 MB).
struct GpuContext;

// Creates a context for inputs of up to capacity bytes.
// Refuses with NotEnoughMemory, without allocating anything, if gpuRequiredBytes(capacity) plus kGpuMemoryReserve
// exceeds the free device memory. Not thread-safe: use one context per thread, and only one at a time per device.
GpuStatus gpuCreateContext(int32_t capacity, GpuContext*& context);

// Frees a context; null is ignored.
void gpuDestroyContext(GpuContext* context);

// Builds the suffix array of text[0, n) on the device into sa[0, n).
// With a context, n must not exceed its capacity (NotEnoughMemory otherwise).
// Without one, refuses with NotEnoughMemory, without allocating anything,
// if gpuRequiredBytes(n) plus kGpuMemoryReserve exceeds the free device memory.
GpuStatus gpuSuffixArray(GpuContext* context, const uint8_t* text, int32_t n, int32_t* sa, GpuStats* stats = nullptr);
GpuStatus gpuSuffixArray(const uint8_t* text, int32_t n, int32_t* sa, GpuStats* stats = nullptr);

// Builds the Burrows-Wheeler transform of text[0, n) on the device into bwt[0, n),
// in buildBwt()'s convention (the BWT of text plus a sentinel, without the sentinel; primary is the sentinel's row).
// The merge writes the BWT directly, so only n bytes come back. Same device memory as gpuSuffixArray().
GpuStatus gpuBwt(GpuContext* context, const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuStats* stats = nullptr);
GpuStatus gpuBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuStats* stats = nullptr);

// Uploads text[0, n) and ranks the suffixes at the sample positions of GpuCover on the device:
// ranks[GpuCover::pack(p)] for each sample position p, gpuSampleSize(n) entries.
// stats->deviceBytes is the peak device memory of the phase.
// Refuses with NotEnoughMemory if that plus kGpuMemoryReserve exceeds the free device memory.
GpuStatus gpuRankSample(const uint8_t* text, int32_t n, uint32_t* ranks, GpuStats* stats = nullptr);

// Page-locked host memory, or null if unavailable.
// Text and output buffers allocated this way are copied directly, at full PCIe speed;
// other host memory goes through the context's pinned staging chunks, at about half that.
void* gpuAllocatePinned(size_t bytes);
void gpuFreePinned(void* pointer);

// A description of the last failure, or "" if there was none.
const char* gpuLastError();
