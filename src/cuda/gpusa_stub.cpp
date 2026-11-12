/*
 * gpusa_stub.cpp - gpusa.h for builds without CUDA: no device, every call is unavailable.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "gpusa.h"

GpuStatus gpuQueryDevice(GpuDevice&) {
    return GpuStatus::Unavailable;
}

int64_t gpuRequiredBytes(int32_t) {
    return 0;
}

GpuStatus gpuCreateContext(int32_t, GpuContext*& context) {
    context = nullptr;
    return GpuStatus::Unavailable;
}

void gpuDestroyContext(GpuContext*) {}

GpuStatus gpuSuffixArray(GpuContext*, const uint8_t* text, int32_t n, int32_t* sa, GpuStats* stats) {
    return gpuSuffixArray(text, n, sa, stats);
}

GpuStatus gpuBwt(GpuContext*, const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, GpuStats* stats) {
    return gpuBwt(text, n, bwt, primary, stats);
}

GpuStatus gpuSuffixArray(const uint8_t*, int32_t, int32_t*, GpuStats* stats) {
    if (stats) {
        *stats = GpuStats{};
    }

    return GpuStatus::Unavailable;
}

GpuStatus gpuBwt(const uint8_t*, int32_t, uint8_t*, int32_t& primary, GpuStats* stats) {
    primary = -1;

    if (stats) {
        *stats = GpuStats{};
    }

    return GpuStatus::Unavailable;
}

GpuStatus gpuRankSample(const uint8_t*, int32_t, uint32_t*, GpuStats* stats) {
    if (stats) {
        *stats = GpuStats{};
    }

    return GpuStatus::Unavailable;
}

const char* gpuLastError() {
    return "sadcs was built without CUDA (configure with the cuda preset)";
}

void* gpuAllocatePinned(size_t) {
    return nullptr;
}

void gpuFreePinned(void*) {}
