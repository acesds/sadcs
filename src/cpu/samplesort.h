/*
 * samplesort.h - Ranks the difference cover sample suffixes completely.
 *
 * This happens in three steps, so the caller can run the last two alongside other work:
 * 1) bucketSortSample() groups the sample positions by two characters
 * 2) stringSortSample() sorts them to V+1 characters with tasks in the caller's parallel region
 * 3) doublingSortSample() resolves the remaining ties with prefix doubling on packed sample indexes
 *
 * Sorting to V+1 rather than V characters guarantees every tied suffix is longer than V, so the doubling
 * never steps past the end of the sample.
 * Note that the rank array has one extra slot, rank[sampleSize] = -1, for the empty suffix at position n.
 * It's what pack(n) addresses, and it's smaller than every real suffix by definition.
 *
 * The caller provides both arrays, so they can live in otherwise unused parts of the output suffix array.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>
#include <vector>
#include "bucketsort.h"
#include "differencecover.h"
#include "stringsort.h"
#include "doubling.h"

// The sample's string sort and doubling are the longer path, so their tasks go before the main string sort's.
constexpr int kSampleSortPriority = 1;

struct SampleArrays {
    int32_t* rank;  // size + 1 entries
    int32_t* sa;    // size entries; the doubling's input, scratch afterwards
    int32_t size;
};

// Converts string-sorted sample positions (tie group members after the first complemented)
// into the doubling's input: finished runs become -(run length),
// tie group members become packed indexes ranked by their group's last slot.
template <int V>
void prepareDoublingSort(int32_t* sampleSa, int32_t* sampleRank, int32_t sampleSize) {
    using Cover = DifferenceCover<V>;

    for (int32_t i = sampleSize - 1; i >= 0; i--) {
        if (sampleSa[i] >= 0) {
            const int32_t runLast = i;

            do {
                sampleRank[Cover::pack(sampleSa[i])] = i;
                i--;
            } while (i >= 0 && sampleSa[i] >= 0);

            sampleSa[i + 1] = i - runLast;

            if (i < 0) {
                break;
            }
        }

        const int32_t groupLast = i;

        do {
            sampleSa[i] = Cover::pack(~sampleSa[i]);
            sampleRank[sampleSa[i]] = groupLast;
            i--;
        } while (sampleSa[i] < 0);

        sampleSa[i] = Cover::pack(sampleSa[i]);
        sampleRank[sampleSa[i]] = groupLast;
    }
}

// Writes the sample positions into sample.sa by bucket.
// sample.size must be DifferenceCover<V>::pack(n).
template <int V>
std::vector<int32_t> bucketSortSample(const uint8_t* text, int32_t n, const SampleArrays& sample) {
    using Cover = DifferenceCover<V>;

    auto forEachSample = [](int32_t first, int32_t last, auto visit) {
        for (int32_t base = first & ~Cover::kMask; base < last; base += V) {
            for (const int residue : CoverSet<V>::kElements) {
                const int32_t p = base + residue;

                if (p >= first && p < last) {
                    visit(p);
                }
            }
        }
    };

    return bucketSort(text, n, forEachSample, sample.sa);
}

// Returns once the sample is ready for doublingSortSample().
// With taskgroup support, call it from a task in a parallel region, so it can run alongside other tasks.
template <int V>
void stringSortSample(const uint8_t* text, int32_t n, const SampleArrays& sample, const std::vector<int32_t>& bucketStarts) {
#if SADCS_HAS_TASKGROUP
#pragma omp taskgroup
#else
#pragma omp parallel
#pragma omp single
#endif
    spawnBucketSorts(text, n, sample.sa, bucketStarts, V + 1, markTieGroup, kSampleSortPriority);

    prepareDoublingSort<V>(sample.sa, sample.rank, sample.size);
}

// Sequential; leaves the final ranks in sample.rank.
// Note that scratch is passed on to sortByPrefixDoubling().
template <int V>
void doublingSortSample(const SampleArrays& sample, int32_t* scratch, ptrdiff_t scratchSlots) {
    sortByPrefixDoubling(sample.rank, sample.sa, sample.size, DifferenceCover<V>::kSize, scratch, scratchSlots);
    sample.rank[sample.size] = -1;
}
