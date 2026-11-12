/*
 * bucketsort.h - Parallel, stable counting sort of suffixes by their first two characters.
 *
 * For an 8-bit alphabet, the bucket key encodes the first two characters: T[p]*257+T[p+1]+1.
 * For the last suffix the key is T[p] * 257, placing it before all longer suffixes with the
 * same first character.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <omp.h>

constexpr int32_t kBucketCount = 256 * 257;

// Text bytes per thread below which more threads cost more in table work than they save.
constexpr int32_t kBucketSortThreadMinSize = 1 << 16;

// Relies on the zero padding after the text: T[n] reads as 0.
inline int32_t bucketOf(const uint8_t* text, int32_t n, int32_t position) {
    return text[position] * 257 + text[position + 1] + (position + 1 < n ? 1 : 0);
}

// Calls visit(p) for every p in [first, last) whose bit is set, in ascending order.
template <class Visit>
void forEachMarked(const uint32_t* bits, int32_t first, int32_t last, Visit visit) {
    if (first >= last) {
        return;
    }

    const int32_t firstWord = first >> 5;
    const int32_t lastWord = (last - 1) >> 5;

    for (int32_t word = firstWord; word <= lastWord; word++) {
        uint32_t marks = bits[word];

        if (word == firstWord) {
            marks &= ~uint32_t{0} << (first & 31);
        }

        if (word == lastWord) {
            marks &= ~uint32_t{0} >> (31 - ((last - 1) & 31));
        }

        for (; marks != 0; marks &= marks - 1) {
            visit(word * 32 + std::countr_zero(marks));
        }
    }
}

// Writes the positions that forEachIncluded(first, last, visit) visits into out, by bucket.
// forEachIncluded must visit the included positions of [first, last) in ASCENDING order.
// Returns the kBucketCount+1 bucket start offsets into out.
template <class ForEachIncluded>
std::vector<int32_t> bucketSort(const uint8_t* text, int32_t n, ForEachIncluded forEachIncluded, int32_t* out) {
    std::vector<int32_t> bucketStarts(kBucketCount + 1, 0);
    const int threads = static_cast<int>(std::clamp<int64_t>(n / kBucketSortThreadMinSize, 1, omp_get_max_threads()));
    const std::unique_ptr<int32_t[]> threadCursors(new int32_t[static_cast<size_t>(threads) * kBucketCount]);

#pragma omp parallel num_threads(threads)
    {
        const int thread = omp_get_thread_num();
        const int threadCount = omp_get_num_threads();

        // Chunk borders are multiples of 64, so no two threads share a word of a bitset.
        const int32_t chunkBegin = static_cast<int32_t>(static_cast<int64_t>(n) * thread / threadCount) & ~63;
        const int32_t chunkEnd =
            thread + 1 == threadCount ? n : static_cast<int32_t>(static_cast<int64_t>(n) * (thread + 1) / threadCount) & ~63;
        int32_t* cursor = threadCursors.get() + static_cast<size_t>(thread) * kBucketCount;

        std::fill(cursor, cursor + kBucketCount, 0);

        forEachIncluded(chunkBegin, chunkEnd, [&](int32_t p) { cursor[bucketOf(text, n, p)]++; });

#pragma omp barrier
        if (threadCount == 1) {
            // Counts to write cursors in one pass.
            int32_t offset = 0;

            for (int32_t bucket = 0; bucket < kBucketCount; bucket++) {
                const int32_t count = cursor[bucket];

                bucketStarts[bucket] = offset;
                cursor[bucket] = offset;
                offset += count;
            }

            bucketStarts[kBucketCount] = offset;
        } else {
            // Counts to write cursors: bucket by bucket, each thread's share in thread order.
#pragma omp for
            for (int32_t bucket = 0; bucket < kBucketCount; bucket++) {
                int32_t total = 0;

                for (int t = 0; t < threadCount; t++) {
                    total += threadCursors[static_cast<size_t>(t) * kBucketCount + bucket];
                }

                bucketStarts[bucket + 1] = total;
            }

#pragma omp single
            for (int32_t bucket = 0; bucket < kBucketCount; bucket++) {
                bucketStarts[bucket + 1] += bucketStarts[bucket];
            }

#pragma omp for
            for (int32_t bucket = 0; bucket < kBucketCount; bucket++) {
                int32_t offset = bucketStarts[bucket];

                for (int t = 0; t < threadCount; t++) {
                    int32_t& slot = threadCursors[static_cast<size_t>(t) * kBucketCount + bucket];
                    const int32_t count = slot;
                    slot = offset;
                    offset += count;
                }
            }
        }

        forEachIncluded(chunkBegin, chunkEnd, [&](int32_t p) { out[cursor[bucketOf(text, n, p)]++] = p; });
    }

    return bucketStarts;
}
