/*
 * stringsort.h - Depth-limited, task-parallel multikey quicksort of suffixes.
 *
 * Sorts suffixes that agree on their first `depth` characters, comparing up to `limit` characters.
 * Groups still equal there are tie groups and go to onTie(first, last), which may run concurrently
 * for disjoint ranges.
 *
 * A suffix sorts before any longer suffix it's a prefix of, so suffixes reaching the end of the text
 * never tie. Mid-sized groups first try a one-character MSD radix pass; groups it can't split well,
 * and the big ones, are partitioned on 7 characters at once. Small groups are insertion-sorted.
 * Parts of kTaskMinSize or more become OpenMP tasks; outside a parallel region they just run inline.
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
#include <utility>
#include <vector>
#include <immintrin.h>
#include "bucketsort.h"
#include "simd.h"

// MSVC's OpenMP (/openmp:llvm) has neither task priorities nor taskgroup.
#if defined(__clang__)
#define SADCS_TASK_PRIORITY(p) priority(p)
#define SADCS_HAS_TASKGROUP 1
#else
#define SADCS_TASK_PRIORITY(p)
#define SADCS_HAS_TASKGROUP 0
#endif

constexpr int32_t kKeyChars = 7;
constexpr ptrdiff_t kInsertionSortMaxSize = 32;
constexpr ptrdiff_t kTaskMinSize = 1 << 14;
constexpr int32_t kBucketBatchMinSize = 1 << 13;
constexpr ptrdiff_t kRadixMinSize = 256;
constexpr ptrdiff_t kRadixMaxSize = 1 << 18;

// Partitioning reads the text at random; loading the suffix this many places ahead hides most of the latency.
constexpr ptrdiff_t kPrefetchDistance = 16;

inline void prefetchText(const uint8_t* address) {
    _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
}

// The next 7 characters at the given position.
// Tis is coded as big-endian in the top 56 bits and the remaining length (capped at 7) in the low byte.
// Integer order is then suffix order and equal keys mean both suffixes continue with the same 7 characters.
inline uint64_t suffixKey(const uint8_t* text, int32_t n, int32_t position) {
    const int32_t remaining = n - position;
    const uint64_t characters = loadBigEndian64(text + position) & ~static_cast<uint64_t>(0xFF);

    if (remaining >= kKeyChars) {
        return characters | kKeyChars;
    }

    const uint64_t validMask = remaining == 0 ? 0 : ~static_cast<uint64_t>(0) << (64 - 8 * remaining);

    return (characters & validMask) | static_cast<uint64_t>(remaining);
}

struct KeyedSuffix {
    uint64_t key;
    int32_t position;
};

// Insertion sort for small groups on keys loaded once per suffix.
// Each suffix costs one random text access per 7 characters rather than one per comparison.
// Runs of equal keys go on 7 characters deeper.
template <class TieHandler>
void insertionSortSuffixes(const uint8_t* text, int32_t n, int32_t* first, int32_t* last, int32_t depth, int32_t limit, TieHandler& onTie) {
    KeyedSuffix keyed[kInsertionSortMaxSize];
    const ptrdiff_t size = last - first;

    for (ptrdiff_t i = 0; i < size; i++) {
        const KeyedSuffix current{suffixKey(text, n, first[i] + depth), first[i]};
        ptrdiff_t hole = i;

        for (; hole > 0 && current.key < keyed[hole - 1].key; hole--) {
            keyed[hole] = keyed[hole - 1];
        }

        keyed[hole] = current;
    }

    for (ptrdiff_t i = 0; i < size; i++) {
        first[i] = keyed[i].position;
    }

    const int32_t nextDepth = depth + kKeyChars;
    ptrdiff_t runFirst = 0;

    for (ptrdiff_t i = 1; i <= size; i++) {
        if (i < size && keyed[i].key == keyed[runFirst].key) {
            continue;
        }

        if (i - runFirst > 1) {
            if (nextDepth >= limit) {
                onTie(first + runFirst, first + i);
            } else {
                insertionSortSuffixes(text, n, first + runFirst, first + i, nextDepth, limit, onTie);
            }
        }

        runFirst = i;
    }
}

inline uint64_t medianOfThreeKeys(uint64_t a, uint64_t b, uint64_t c) {
    if (a > b) {
        std::swap(a, b);
    }

    return c <= a ? a : (c >= b ? b : c);
}

struct Pivot {
    uint64_t key;
    bool mostlyEqual;  // most samples equal the pivot: expect a big equal part
};

// Median of 3, or "Tukey's ninther" for big groups.
inline Pivot choosePivot(const uint8_t* text, int32_t n, const int32_t* first, ptrdiff_t size, int32_t depth) {
    uint64_t samples[9];
    int sampleCount;

    if (size <= 1024) {
        const ptrdiff_t at[] = {0, size / 2, size - 1};
        sampleCount = 3;

        for (int i = 0; i < sampleCount; i++) {
            samples[i] = suffixKey(text, n, first[at[i]] + depth);
        }
    } else {
        const ptrdiff_t step = size / 8;
        const ptrdiff_t middle = size / 2;
        const ptrdiff_t at[] = {0, step, 2 * step, middle - step, middle, middle + step, size - 1 - 2 * step, size - 1 - step, size - 1};

        sampleCount = 9;

        for (int i = 0; i < sampleCount; i++) {
            samples[i] = suffixKey(text, n, first[at[i]] + depth);
        }
    }

    const uint64_t key = sampleCount == 3
        ? medianOfThreeKeys(samples[0], samples[1], samples[2])
        : medianOfThreeKeys(medianOfThreeKeys(samples[0], samples[1], samples[2]),
              medianOfThreeKeys(samples[3], samples[4], samples[5]),
              medianOfThreeKeys(samples[6], samples[7], samples[8]));
    int equalCount = 0;

    for (int i = 0; i < sampleCount; i++) {
        equalCount += samples[i] == key;
    }

    return Pivot{key, 2 * equalCount > sampleCount};
}

constexpr int kRadixGroups = 257;

// One MSD radix pass on the character at depth (group 0: suffixes ending there, c+1 otherwise).
// Returns false, leaving the range untouched, if everything falls into one group.
inline bool radixPass(const uint8_t* text, int32_t n, int32_t* first, ptrdiff_t size, int32_t depth, int32_t (&groupStarts)[kRadixGroups + 1]) {
    thread_local uint16_t digits[kRadixMaxSize];
    thread_local int32_t scratch[kRadixMaxSize];

    int32_t counts[kRadixGroups] = {};

    for (ptrdiff_t i = 0; i < size; i++) {
        if (i + kPrefetchDistance < size) {
            prefetchText(text + first[i + kPrefetchDistance] + depth);
        }

        const int32_t position = first[i] + depth;
        const uint16_t digit = position < n ? static_cast<uint16_t>(text[position] + 1) : 0;

        digits[i] = digit;
        counts[digit]++;
    }

    // A pass that leaves most suffixes in one group (long runs) is worth less than a 7-character quicksort step.
    if (*std::max_element(counts, counts + kRadixGroups) > size - size / 8) {
        return false;
    }

    int32_t cursor[kRadixGroups];

    for (int g = 0, offset = 0; g < kRadixGroups; g++) {
        groupStarts[g] = cursor[g] = offset;
        offset += counts[g];
    }

    groupStarts[kRadixGroups] = static_cast<int32_t>(size);

    for (ptrdiff_t i = 0; i < size; i++) {
        scratch[cursor[digits[i]]++] = first[i];
    }

    std::copy(scratch, scratch + size, first);

    return true;
}

// As in introsort, a range may be partitioned about 2 log2(size) times without getting deeper into the text;
// the less and greater parts inherit what's left of that budget.
// A range that runs out is sorted on its keys with std::sort instead, so bad pivots can't make it quadratic.
template <class TieHandler>
void stringSort(const uint8_t* text, int32_t n, int32_t* first, int32_t* last, int32_t depth, int32_t limit, TieHandler onTie, int priority, int budget = -1);

#if defined(_MSC_VER)
#define SADCS_NOINLINE __declspec(noinline)
#else
#define SADCS_NOINLINE __attribute__((noinline))
#endif

// Fallback for a range out of partition budget.
// It hardly ever runs, so it's kept out of line, away from stringSort()'s partition loop.
template <class TieHandler>
SADCS_NOINLINE void keySortSuffixes(const uint8_t* text, int32_t n, int32_t* first, int32_t* last, int32_t depth, int32_t limit, TieHandler onTie, int priority) {
    std::sort(first, last, [text, n, depth](int32_t a, int32_t b) {
        return suffixKey(text, n, a + depth) < suffixKey(text, n, b + depth);
    });

    int32_t* runFirst = first;

    while (runFirst < last) {
        const uint64_t runKey = suffixKey(text, n, *runFirst + depth);
        int32_t* runLast = runFirst + 1;

        while (runLast < last && suffixKey(text, n, *runLast + depth) == runKey) {
            runLast++;
        }

        if (runLast - runFirst >= kTaskMinSize) {
#pragma omp task firstprivate(runFirst, runLast) SADCS_TASK_PRIORITY(priority)
            stringSort(text, n, runFirst, runLast, depth + kKeyChars, limit, onTie, priority);
        } else if (runLast - runFirst > 1) {
            stringSort(text, n, runFirst, runLast, depth + kKeyChars, limit, onTie, priority);
        }

        runFirst = runLast;
    }
}

template <class TieHandler>
void stringSort(const uint8_t* text, int32_t n, int32_t* first, int32_t* last, int32_t depth, int32_t limit, TieHandler onTie, int priority, int budget) {
    struct Part {
        int32_t* first;
        int32_t* last;
        int32_t depth;
        int budget;

        ptrdiff_t size() const {
            return last - first;
        }
    };

    bool radixHelps = true;

    while (true) {
        const ptrdiff_t size = last - first;

        if (size < 2) {
            return;
        }

        if (depth >= limit) {
            onTie(first, last);
            return;
        }

        if (size <= kInsertionSortMaxSize) {
            insertionSortSuffixes(text, n, first, last, depth, limit, onTie);
            return;
        }

        int32_t groupStarts[kRadixGroups + 1];
        bool radixSplit = false;

        if (radixHelps && size >= kRadixMinSize && size <= kRadixMaxSize) {
            radixSplit = radixPass(text, n, first, size, depth, groupStarts);
            radixHelps = radixSplit;
        }

        if (radixSplit) {
            for (int g = 1; g < kRadixGroups; g++) {
                int32_t* groupFirst = first + groupStarts[g];
                int32_t* groupLast = first + groupStarts[g + 1];

                if (groupLast - groupFirst >= kTaskMinSize) {
#pragma omp task firstprivate(groupFirst, groupLast) SADCS_TASK_PRIORITY(priority)
                    stringSort(text, n, groupFirst, groupLast, depth + 1, limit, onTie, priority);
                } else if (groupLast - groupFirst > 1) {
                    stringSort(text, n, groupFirst, groupLast, depth + 1, limit, onTie, priority);
                }
            }

            return;
        }

        if (budget < 0) {
            budget = 2 * static_cast<int>(std::bit_width(static_cast<size_t>(size)));
        }

        if (budget == 0) {
            keySortSuffixes(text, n, first, last, depth, limit, onTie, priority);
            return;
        }

        budget--;

        const Pivot pivot = choosePivot(text, n, first, size, depth);
        const uint64_t pivotKey = pivot.key;

        // Simple three-way partition:
        // If most keys equal the pivot (repetitive text), the branches are predictable and a classic Dutch flag pass
        // moves the fewest entries. Otherwise it's branch-free: a Lomuto pass on key<=pivot, with each smaller entry
        // also rotated to the end of the less part, so the comparisons only move indexes.
        ptrdiff_t lessCount = 0;
        ptrdiff_t notGreaterCount = 0;

        if (pivot.mostlyEqual) {
            int32_t* lessEnd = first;
            int32_t* greaterBegin = last;

            for (int32_t* it = first; it < greaterBegin;) {
                if (greaterBegin - it > kPrefetchDistance) {
                    prefetchText(text + it[kPrefetchDistance] + depth);
                    prefetchText(text + greaterBegin[-kPrefetchDistance] + depth);
                }

                const uint64_t key = suffixKey(text, n, *it + depth);

                if (key < pivotKey) {
                    std::swap(*lessEnd++, *it++);
                } else if (key > pivotKey) {
                    std::swap(*it, *--greaterBegin);
                } else {
                    it++;
                }
            }

            lessCount = lessEnd - first;
            notGreaterCount = greaterBegin - first;
        } else {
            for (ptrdiff_t i = 0; i < size; i++) {
                if (i + kPrefetchDistance < size) {
                    prefetchText(text + first[i + kPrefetchDistance] + depth);
                }

                const int32_t entry = first[i];
                const uint64_t key = suffixKey(text, n, entry + depth);

                first[i] = first[notGreaterCount];
                first[notGreaterCount] = entry;

                const ptrdiff_t target = key < pivotKey ? lessCount : notGreaterCount;

                first[notGreaterCount] = first[target];
                first[target] = entry;

                lessCount += key < pivotKey;
                notGreaterCount += key <= pivotKey;
            }
        }

        int32_t* const lessEnd = first + lessCount;
        int32_t* const greaterBegin = first + notGreaterCount;

        // Only the equal part gets deeper, so only it starts a fresh budget.
        const Part parts[3] = {{first, lessEnd, depth, budget}, {lessEnd, greaterBegin, depth + kKeyChars, -1},
            {greaterBegin, last, depth, budget}};

        // Loop on the biggest part (bounded stack depth), hand off or recurse on the others.
        int biggest = 0;

        for (int i = 1; i < 3; i++) {
            if (parts[i].size() > parts[biggest].size()) {
                biggest = i;
            }
        }

        for (int i = 0; i < 3; i++) {
            Part part = parts[i];

            if (i == biggest || part.size() < 2) {
                continue;
            }

            if (part.size() >= kTaskMinSize) {
#pragma omp task firstprivate(part) SADCS_TASK_PRIORITY(priority)
                stringSort(text, n, part.first, part.last, part.depth, limit, onTie, priority, part.budget);
            } else {
                stringSort(text, n, part.first, part.last, part.depth, limit, onTie, priority, part.budget);
            }
        }

        first = parts[biggest].first;
        last = parts[biggest].last;
        depth = parts[biggest].depth;
        budget = parts[biggest].budget;
    }
}

// A tie handler that leaves the group to be sorted later.
// All members but the first are complemented, so a group is a non-negative entry followed by negative ones.
inline void markTieGroup(int32_t* first, int32_t* last) {
    for (int32_t* it = first + 1; it < last; it++) {
        *it = ~*it;
    }
}

// Sorts every bucket of a bucketSort() result from depth 2 on, as tasks of the caller's parallel region.
// Small buckets are batched into one task, big ones get their own.
template <class TieHandler>
void spawnBucketSorts(const uint8_t* text, int32_t n, int32_t* sa, const std::vector<int32_t>& bucketStarts, int32_t limit, TieHandler onTie, int priority) {
    const int32_t* starts = bucketStarts.data();
    auto sortBucketRange = [=](int32_t firstBucket, int32_t lastBucket) {
        for (int32_t bucket = firstBucket; bucket < lastBucket; bucket++) {
            if (starts[bucket + 1] - starts[bucket] > 1) {
                stringSort(text, n, sa + starts[bucket], sa + starts[bucket + 1], 2, limit, onTie, priority);
            }
        }
    };

    int32_t batchFirst = 0;

    for (int32_t bucket = 0; bucket <= kBucketCount; bucket++) {
        const bool atEnd = bucket == kBucketCount;
        const bool isBig = !atEnd && starts[bucket + 1] - starts[bucket] >= kBucketBatchMinSize;
        const bool batchFull = starts[bucket] - starts[batchFirst] >= kBucketBatchMinSize;

        if (atEnd || isBig || batchFull) {
            const int32_t batchLast = bucket;

            if (starts[batchLast] - starts[batchFirst] > 1) {
#pragma omp task firstprivate(sortBucketRange, batchFirst, batchLast) SADCS_TASK_PRIORITY(priority)
                sortBucketRange(batchFirst, batchLast);
            }

            batchFirst = bucket;
        }

        if (isBig) {
#pragma omp task firstprivate(sortBucketRange, bucket) SADCS_TASK_PRIORITY(priority)
            sortBucketRange(bucket, bucket + 1);

            batchFirst = bucket + 1;
        }
    }
}
