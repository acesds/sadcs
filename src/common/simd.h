/*
 * simd.h - Wide byte compares and big-endian word loads on the text.
 *
 * These read up to 31 bytes past what they logically look at.
 * Every text buffer carries kTextPadding zero bytes after its last real byte.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

constexpr int32_t kTextPadding = 64;

// Length of the common prefix of a and b, capped at limit.
inline int32_t commonPrefixLength(const uint8_t* a, const uint8_t* b, int32_t limit) {
#if defined(__AVX2__)
    constexpr int32_t kChunk = 32;
#else
    constexpr int32_t kChunk = 16;
#endif

    for (int32_t offset = 0; offset < limit; offset += kChunk) {
#if defined(__AVX2__)
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + offset));
        const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + offset));
        const uint32_t mismatches = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y)));
#else
        const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset));
        const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset));
        const uint32_t mismatches = ~static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(x, y))) & 0xFFFFu;
#endif

        if (mismatches != 0) {
            const int32_t length = offset + std::countr_zero(mismatches);
            return length < limit ? length : limit;
        }
    }

    return limit > 0 ? limit : 0;
}

// The 8 bytes at p as a big-endian number, so integer order is byte string order.
inline uint64_t loadBigEndian64(const uint8_t* p) {
    uint64_t word;
    memcpy(&word, p, sizeof(word));
#if defined(_MSC_VER)
    return _byteswap_uint64(word);
#else
    return __builtin_bswap64(word);
#endif
}
