/*
 * dcsort.cpp - Implementation of the difference cover SACA.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "dcsort.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>
#include <omp.h>
#include "bucketsort.h"
#include "differencecover.h"
#include "inducedsort.h"
#include "parallelsort.h"
#include "samplesort.h"
#include "stringsort.h"

namespace {
    constexpr int32_t kTieScanMinChunk = 1 << 16;
    constexpr ptrdiff_t kClassSortMinSize = 32;
    constexpr ptrdiff_t kParallelTieGroupMinSize = 1 << 15;
    constexpr ptrdiff_t kTieTaskSize = 1 << 14;
    constexpr ptrdiff_t kRadixSortMinSize = 256;
    constexpr ptrdiff_t kParallelClassSortMinSize = 1 << 16;
    constexpr double kThreadCapBytes = 8192.0;

    // Sorts keys by their upper 32 bits, which are below 2^rankBits, with LSD radix passes through buffer.
    // Larger inputs pay for wider digits and take fewer passes.
    void radixSortByRank(uint64_t* keys, uint64_t* buffer, ptrdiff_t size, int rankBits) {
        const int maxDigitBits = size >= 4096 ? 11 : 8;
        const int passCount = std::max(1, (rankBits + maxDigitBits - 1) / maxDigitBits);
        const int digitBits = (rankBits + passCount - 1) / passCount;
        const uint64_t mask = (uint64_t{1} << digitBits) - 1;
        int32_t counts[1 << 11];
        uint64_t* source = keys;
        uint64_t* target = buffer;

        for (int pass = 0; pass < passCount; pass++) {
            const int shift = 32 + pass * digitBits;
            std::fill(counts, counts + mask + 1, 0);

            for (ptrdiff_t i = 0; i < size; i++) {
                counts[(source[i] >> shift) & mask]++;
            }

            int32_t slot = 0;

            for (uint64_t digit = 0; digit <= mask; digit++) {
                const int32_t count = counts[digit];
                counts[digit] = slot;
                slot += count;
            }

            for (ptrdiff_t i = 0; i < size; i++) {
                target[counts[(source[i] >> shift) & mask]++] = source[i];
            }

            std::swap(source, target);
        }

        if (source != keys) {
            std::copy(source, source + size, keys);
        }
    }

    // Sorts the keys of one class, by radix if there are enough of them.
    void sortClass(uint64_t* keys, uint64_t* buffer, ptrdiff_t size, int rankBits) {
        if (size >= kRadixSortMinSize) {
            radixSortByRank(keys, buffer, size, rankBits);
        } else {
            std::sort(keys, keys + size);
        }
    }

    // Sorts one tie group.
    // Members at the same distance d from the next sample position compare by rank[pack(i+d)] alone,
    // so each such class is sorted on keys loaded once.
    // The few classes are then merged pairwise with the general comparison.
    // keyed and buffer hold size keys each; the merges use buffer as int32_t.
    template <int V, class Less>
    void sortTieGroup(const int32_t* rank, int rankBits, int32_t* first, int32_t* last, const Less& less, uint64_t* keyed, uint64_t* buffer) {
        using Cover = DifferenceCover<V>;
        constexpr int kClassCount = Cover::kNextSampleDistanceCount;
        const ptrdiff_t size = last - first;

        if (size < kClassSortMinSize) {
            std::sort(first, last, less);
            return;
        }

        int32_t classStarts[kClassCount + 1] = {};

        for (const int32_t* it = first; it < last; it++) {
            classStarts[Cover::nextSampleDistance(*it) + 1]++;
        }

        for (int d = 0; d < kClassCount; d++) {
            classStarts[d + 1] += classStarts[d];
        }

        int32_t cursor[kClassCount];
        std::copy(classStarts, classStarts + kClassCount, cursor);

        for (const int32_t* it = first; it < last; it++) {
            const int32_t d = Cover::nextSampleDistance(*it);
            const uint64_t key = static_cast<uint64_t>(rank[Cover::pack(*it + d)]);
            keyed[cursor[d]++] = (key << 32) | static_cast<uint32_t>(*it);
        }

        int32_t runBounds[kClassCount + 1] = {0};
        int runCount = 0;

        for (int d = 0; d < kClassCount; d++) {
            if (classStarts[d + 1] > classStarts[d]) {
                sortClass(keyed + classStarts[d], buffer + classStarts[d], classStarts[d + 1] - classStarts[d], rankBits);
                runBounds[++runCount] = classStarts[d + 1];
            }
        }

        for (ptrdiff_t i = 0; i < size; i++) {
            first[i] = static_cast<int32_t>(keyed[i] & 0xFFFFFFFFu);
        }

        int32_t* source = first;
        int32_t* target = reinterpret_cast<int32_t*>(buffer);

        while (runCount > 1) {
            int mergedCount = 0;

            for (int run = 0; run < runCount; run += 2) {
                const int32_t runFirst = runBounds[run];
                const int32_t runMiddle = runBounds[run + 1];
                const int32_t runLast = run + 2 <= runCount ? runBounds[run + 2] : runMiddle;

                std::merge(source + runFirst, source + runMiddle, source + runMiddle, source + runLast, target + runFirst, less);
                runBounds[++mergedCount] = runLast;
            }

            runCount = mergedCount;
            std::swap(source, target);
        }

        if (source != first) {
            std::copy(source, source + size, first);
        }
    }

    // Returns how many of the first k merged elements of the sorted runs a and b come from a.
    template <class Less>
    ptrdiff_t mergeSplit(const int32_t* a, ptrdiff_t aSize, const int32_t* b, ptrdiff_t bSize, ptrdiff_t k, const Less& less) {
        ptrdiff_t low = std::max<ptrdiff_t>(0, k - bSize);
        ptrdiff_t high = std::min(k, aSize);

        while (low < high) {
            const ptrdiff_t middle = (low + high) / 2;

            if (less(b[k - middle - 1], a[middle])) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }

        return low;
    }

    // Sorts one large tie group the way sortTieGroup() does, with tasks:
    // blocks of members are keyed in parallel, the classes are sorted in parallel,
    // and each merge is split into pieces along merge paths.
    template <int V, class Less>
    void sortLargeTieGroup(const int32_t* rank, int rankBits, int32_t* first, int32_t* last, const Less& less, uint64_t* keyed, uint64_t* buffer) {
        using Cover = DifferenceCover<V>;
        constexpr int kClassCount = Cover::kNextSampleDistanceCount;
        const ptrdiff_t size = last - first;
        const ptrdiff_t blockCount = std::max<ptrdiff_t>(1, size / kTieTaskSize);
        const ptrdiff_t blockSize = (size + blockCount - 1) / blockCount;

        // Each block's member count per class, then the slot of its first member per class.
        std::vector<int32_t> blockCursorStorage(static_cast<size_t>(blockCount) * kClassCount);
        int32_t* blockCursors = blockCursorStorage.data();

        for (ptrdiff_t block = 0; block < blockCount; block++) {
#pragma omp task firstprivate(block)
            {
                int32_t* counts = blockCursors + block * kClassCount;
                const int32_t* blockLast = first + std::min(size, (block + 1) * blockSize);

                for (const int32_t* it = first + block * blockSize; it < blockLast; it++) {
                    counts[Cover::nextSampleDistance(*it)]++;
                }
            }
        }

#pragma omp taskwait
        int32_t classStarts[kClassCount + 1];
        int32_t slot = 0;

        for (int d = 0; d < kClassCount; d++) {
            classStarts[d] = slot;

            for (ptrdiff_t block = 0; block < blockCount; block++) {
                const int32_t count = blockCursors[block * kClassCount + d];
                blockCursors[block * kClassCount + d] = slot;
                slot += count;
            }
        }

        classStarts[kClassCount] = slot;

        for (ptrdiff_t block = 0; block < blockCount; block++) {
#pragma omp task firstprivate(block)
            {
                int32_t* cursor = blockCursors + block * kClassCount;
                const int32_t* blockLast = first + std::min(size, (block + 1) * blockSize);

                for (const int32_t* it = first + block * blockSize; it < blockLast; it++) {
                    const int32_t d = Cover::nextSampleDistance(*it);
                    const uint64_t key = static_cast<uint64_t>(rank[Cover::pack(*it + d)]);
                    keyed[cursor[d]++] = (key << 32) | static_cast<uint32_t>(*it);
                }
            }
        }

#pragma omp taskwait
        int32_t runBounds[kClassCount + 1] = {0};
        int runCount = 0;

#pragma omp taskgroup
        for (int d = 0; d < kClassCount; d++) {
            uint64_t* classFirst = keyed + classStarts[d];
            uint64_t* classBuffer = buffer + classStarts[d];
            const ptrdiff_t classSize = classStarts[d + 1] - classStarts[d];

            if (classSize >= kParallelClassSortMinSize) {
                parallelSort(classFirst, classFirst + classSize, std::less<uint64_t>());
            } else if (classSize > 1) {
#pragma omp task firstprivate(classFirst, classBuffer, classSize)
                sortClass(classFirst, classBuffer, classSize, rankBits);
            }

            if (classSize > 0) {
                runBounds[++runCount] = classStarts[d + 1];
            }
        }

        for (ptrdiff_t block = 0; block < blockCount; block++) {
#pragma omp task firstprivate(block)
            {
                const ptrdiff_t blockLast = std::min(size, (block + 1) * blockSize);

                for (ptrdiff_t i = block * blockSize; i < blockLast; i++) {
                    first[i] = static_cast<int32_t>(keyed[i] & 0xFFFFFFFFu);
                }
            }
        }

#pragma omp taskwait
        int32_t* source = first;
        int32_t* target = reinterpret_cast<int32_t*>(buffer);

        while (runCount > 1) {
            int mergedCount = 0;

            for (int run = 0; run < runCount; run += 2) {
                const ptrdiff_t runFirst = runBounds[run];
                const ptrdiff_t runMiddle = runBounds[run + 1];
                const ptrdiff_t runLast = run + 2 <= runCount ? runBounds[run + 2] : runMiddle;

                // A run without a partner is copied, as a merge with an empty run.
                for (ptrdiff_t pieceFirst = runFirst; pieceFirst < runLast; pieceFirst += kTieTaskSize) {
#pragma omp task firstprivate(pieceFirst)
                    {
                        const ptrdiff_t pieceLast = std::min(pieceFirst + kTieTaskSize, runLast);
                        const int32_t* a = source + runFirst;
                        const int32_t* b = source + runMiddle;
                        const ptrdiff_t aSize = runMiddle - runFirst;
                        const ptrdiff_t bSize = runLast - runMiddle;
                        const ptrdiff_t aFirst = mergeSplit(a, aSize, b, bSize, pieceFirst - runFirst, less);
                        const ptrdiff_t aLast = mergeSplit(a, aSize, b, bSize, pieceLast - runFirst, less);

                        std::merge(a + aFirst, a + aLast, b + (pieceFirst - runFirst - aFirst), b + (pieceLast - runFirst - aLast),
                            target + pieceFirst, less);
                    }
                }

                runBounds[++mergedCount] = static_cast<int32_t>(runLast);
            }

#pragma omp taskwait
            runCount = mergedCount;
            std::swap(source, target);
        }

        if (source != first) {
            for (ptrdiff_t block = 0; block < blockCount; block++) {
#pragma omp task firstprivate(block)
                {
                    const ptrdiff_t blockLast = std::min(size, (block + 1) * blockSize);
                    std::copy(source + block * blockSize, source + blockLast, first + block * blockSize);
                }
            }

#pragma omp taskwait
        }
    }

    // Sorts the tie groups markTieGroup() left in sa.
    template <int V, class Less>
    void sortMarkedTieGroups(int32_t* sa, int32_t n, const int32_t* rank, int rankBits, Less less, int32_t* scratch, ptrdiff_t scratchSlots) {
        const int threadCount = omp_get_max_threads();
        const int32_t chunkSize = std::max(kTieScanMinChunk, n / (16 * threadCount));
        const int32_t chunkCount = (n + chunkSize - 1) / chunkSize;
        std::vector<int32_t> chunkStarts(static_cast<size_t>(chunkCount) + 1, n);

        // Aligns the scratch for the 64-bit keys; every slice of it is a multiple of 4 slots.
        const ptrdiff_t misalignment = (reinterpret_cast<uintptr_t>(scratch) % alignof(uint64_t)) / sizeof(int32_t);
        scratch += misalignment;
        scratchSlots -= misalignment;

        // Small groups are sorted one per thread, in a share of the scratch.
        // Large groups are collected and sorted afterwards with tasks, each in a slice of the whole scratch.
        const ptrdiff_t smallGroupMaxSize = std::min(kParallelTieGroupMinSize, scratchSlots / (4 * threadCount));
        std::vector<std::pair<int32_t, int32_t>> largeGroups;

        auto sortGroupsIn = [sa, n, rank, rankBits, less, scratch, smallGroupMaxSize, &largeGroups](int32_t first, int32_t last) {
            uint64_t* keyed = reinterpret_cast<uint64_t*>(scratch + omp_get_thread_num() * 4 * smallGroupMaxSize);
            uint64_t* buffer = keyed + smallGroupMaxSize;

            for (int32_t groupFirst = first; groupFirst < last;) {
                int32_t groupLast = groupFirst + 1;

                for (; groupLast < n && sa[groupLast] < 0; groupLast++) {
                    sa[groupLast] = ~sa[groupLast];
                }

                const ptrdiff_t size = groupLast - groupFirst;

                if (size >= kClassSortMinSize && size > smallGroupMaxSize) {
#pragma omp critical(sadcsLargeTieGroups)
                    largeGroups.emplace_back(groupFirst, groupLast);
                } else if (size > 1) {
                    sortTieGroup<V>(rank, rankBits, sa + groupFirst, sa + groupLast, less, keyed, buffer);
                }

                groupFirst = groupLast;
            }
        };

#pragma omp parallel
        {
#pragma omp for
            for (int32_t chunk = 0; chunk < chunkCount; chunk++) {
                int32_t start = chunk * chunkSize;

                while (start < n && sa[start] < 0) {
                    start++;
                }

                chunkStarts[chunk] = start;
            }

#pragma omp single
            for (int32_t chunk = 0; chunk < chunkCount; chunk++) {
#pragma omp task firstprivate(chunk)
                sortGroupsIn(chunkStarts[chunk], chunkStarts[chunk + 1]);
            }
        }

        if (largeGroups.empty()) {
            return;
        }

        // The groups go in batches whose scratch fits; a group too large for all of it is quicksorted.
#pragma omp parallel
#pragma omp single
        for (size_t group = 0; group < largeGroups.size();) {
            ptrdiff_t usedSlots = 0;

            for (; group < largeGroups.size(); group++) {
                const int32_t groupFirst = largeGroups[group].first;
                const int32_t groupLast = largeGroups[group].second;
                const ptrdiff_t slots = 4 * static_cast<ptrdiff_t>(groupLast - groupFirst);

                if (slots > scratchSlots) {
#pragma omp task firstprivate(groupFirst, groupLast)
                    {
#pragma omp taskgroup
                        parallelSort(sa + groupFirst, sa + groupLast, less);
                    }

                    continue;
                }

                if (usedSlots + slots > scratchSlots) {
                    break;
                }

                int32_t* groupScratch = scratch + usedSlots;
                usedSlots += slots;

#pragma omp task firstprivate(groupFirst, groupLast, groupScratch)
                sortLargeTieGroup<V>(rank, rankBits, sa + groupFirst, sa + groupLast, less, reinterpret_cast<uint64_t*>(groupScratch),
                    reinterpret_cast<uint64_t*>(groupScratch) + (groupLast - groupFirst));
            }

#pragma omp taskwait
        }
    }

    template <int V, bool Bwt>
    void build(const uint8_t* text, int32_t n, int32_t* sa, BuildStats& stats) {
        using Cover = DifferenceCover<V>;

        // Until the RMS suffixes move to their final slots, sa holds them at the front and
        // the working arrays at the back: the RMS bits, then the sample's suffix array and ranks.
        // Note that there are at most n/2 RMS suffixes, as no two RMS positions are adjacent.
        const int32_t sampleSize = Cover::pack(n);
        const int32_t rmsWords = (n + 31) / 32;
        const int32_t backSlots = rmsWords + 2 * sampleSize + 1;
        std::vector<int32_t> ownBackStorage;
        int32_t* back = sa + (n - backSlots);

        if (n - backSlots < n / 2) {
            ownBackStorage.resize(backSlots);
            back = ownBackStorage.data();
        }

        uint32_t* rmsBits = reinterpret_cast<uint32_t*>(back);
        const SampleArrays sample{back + rmsWords + sampleSize, back + rmsWords, sampleSize};

        stats.sampleSize = sampleSize;

        // With several threads, the sample is sorted alongside the RMS suffixes.
        // On one thread nothing runs alongside it, so the RMS suffixes go first
        // and the sample is only built if their sort leaves ties.
        const bool overlapSample = omp_get_max_threads() > 1;
        const double sampleSortStart = omp_get_wtime();
        std::vector<int32_t> sampleBucketStarts;

        if (overlapSample) {
            sampleBucketStarts = bucketSortSample<V>(text, n, sample);
#if !SADCS_HAS_TASKGROUP
            stringSortSample<V>(text, n, sample, sampleBucketStarts);
#endif
        }

        const double mainSortStart = omp_get_wtime();

        stats.sampleStringSortSeconds = mainSortStart - sampleSortStart;

        // Only the RMS suffixes are sorted directly, compacted at the front of sa;
        // the others are induced from them at the end.
        const SuffixClasses classes = classifySuffixes(text, n, rmsBits);
        auto forEachRms = [rmsBits](int32_t first, int32_t last, auto visit) {
            forEachMarked(rmsBits, first, last, visit);
        };
        const std::vector<int32_t> bucketStarts = bucketSort(text, n, forEachRms, sa);

        // The RMS string sort works in place, so behind the RMS suffixes
        // up to the end of the RMS bits is room for the doubling.
        int32_t* const doublingScratch = sa + classes.rmsCount;
        int32_t* const doublingScratchEnd = ownBackStorage.empty() ? back + rmsWords : sa + n;

        bool hasTies = true;

        if (overlapSample) {
            // Their string sort doesn't need sample ranks, only its tie groups do.
            // The sample's string sort and doubling run as one chain of tasks alongside it.
            // Ties are sorted once both have finished. The chain's tasks have priority.
#pragma omp parallel
#pragma omp single
            {
#pragma omp task SADCS_TASK_PRIORITY(kSampleSortPriority)
                {
#if SADCS_HAS_TASKGROUP
                    const double chainStart = omp_get_wtime();
                    stringSortSample<V>(text, n, sample, sampleBucketStarts);
                    stats.sampleStringSortSeconds += omp_get_wtime() - chainStart;
#endif
                    const double doublingStart = omp_get_wtime();
                    doublingSortSample<V>(sample, doublingScratch, doublingScratchEnd - doublingScratch);
                    stats.sampleDoublingSeconds = omp_get_wtime() - doublingStart;
                }

                spawnBucketSorts(text, n, sa, bucketStarts, V - 1, markTieGroup, 0);
            }
        } else {
            bool sawTie = false;
            auto markAndNoteTie = [&sawTie](int32_t* first, int32_t* last) {
                sawTie = true;
                markTieGroup(first, last);
            };

#pragma omp parallel
#pragma omp single
            spawnBucketSorts(text, n, sa, bucketStarts, V - 1, markAndNoteTie, 0);

            hasTies = sawTie;

            if (hasTies) {
                const double chainStart = omp_get_wtime();
                sampleBucketStarts = bucketSortSample<V>(text, n, sample);
#if SADCS_HAS_TASKGROUP
#pragma omp parallel
#pragma omp single
#endif
                stringSortSample<V>(text, n, sample, sampleBucketStarts);

                const double doublingStart = omp_get_wtime();
                stats.sampleStringSortSeconds += doublingStart - chainStart;
                doublingSortSample<V>(sample, doublingScratch, doublingScratchEnd - doublingScratch);
                stats.sampleDoublingSeconds = omp_get_wtime() - doublingStart;
            }
        }

        if (hasTies) {
            // Tied suffixes agree on V-1>=k characters, so i+k and j+k are at most n.
            const int32_t* rank = sample.rank;

            // With the doubling done, everything from behind the RMS suffixes up to the sample ranks is free scratch.
            const ptrdiff_t tieScratchSlots = ownBackStorage.empty() ? rank - doublingScratch : n - classes.rmsCount;

            sortMarkedTieGroups<V>(sa, classes.rmsCount, rank, std::bit_width(static_cast<uint32_t>(sampleSize)), [rank](int32_t i, int32_t j) {
                const int32_t k = Cover::tieOffset(i, j);
                return rank[Cover::pack(i + k)] < rank[Cover::pack(j + k)];
            }, doublingScratch, tieScratchSlots);
        }

        placeRmsSuffixes(sa, bucketStarts, classes);
        induceSuffixes<Bwt>(text, n, sa, classes);

        stats.mainSortSeconds = omp_get_wtime() - mainSortStart;
    }

    template <bool Bwt>
    void buildWithCover(const uint8_t* text, int32_t n, int32_t* sa, int cover, BuildStats& stats) {
        // Each thread adds fixed costs (tables, synchronization) while the work per thread shrinks
        // with the thread count, so small inputs run on about sqrt(n / kThreadCapBytes) threads.
        const int maxThreads = omp_get_max_threads();
        const int threads = static_cast<int>(std::sqrt(static_cast<double>(n) / kThreadCapBytes));

        omp_set_num_threads(std::clamp(threads, 1, maxThreads));

        struct RestoreThreads {
            int threads;
            ~RestoreThreads() { omp_set_num_threads(threads); }
        } restoreThreads{maxThreads};

        if (cover == 128) {
            build<128, Bwt>(text, n, sa, stats);
        } else if (cover == 256) {
            build<256, Bwt>(text, n, sa, stats);
        } else {
            build<64, Bwt>(text, n, sa, stats);
        }
    }

    bool isSupported(int32_t n, int cover) {
        return n >= 0 && n <= kMaxTextSize && (cover == 64 || cover == 128 || cover == 256);
    }

    // Turns what induction left in sa (~T[j-1] per suffix j, 0 for suffix 0) into the BWT
    // without the sentinel: the sentinel's row comes first, with T[n-1], and suffix 0's row,
    // whose character would be the sentinel, is left out.
    // Returns that row's index plus one.
    int32_t packBwt(const uint8_t* text, int32_t n, const int32_t* sa, uint8_t* bwt) {
        int32_t suffixZeroRow = n;

#pragma omp parallel for reduction(min : suffixZeroRow)
        for (int32_t i = 0; i < n; i++) {
            if (sa[i] == 0) {
                suffixZeroRow = std::min(suffixZeroRow, i);
            }
        }

        bwt[0] = text[n - 1];  // read BEFORE the loops below may overwrite it (if bwt is text)

#pragma omp parallel
        {
            // The rows before suffix 0's shift by one, behind the sentinel's.
#pragma omp for
            for (int32_t i = 0; i < suffixZeroRow; i++) {
                bwt[i + 1] = static_cast<uint8_t>(~sa[i]);
            }

#pragma omp for
            for (int32_t i = suffixZeroRow + 1; i < n; i++) {
                bwt[i] = static_cast<uint8_t>(~sa[i]);
            }
        }

        return suffixZeroRow + 1;
    }
}

bool buildSuffixArray(const uint8_t* text, int32_t n, int32_t* sa, int cover, BuildStats* stats) {
    BuildStats ignoredStats;
    BuildStats& out = stats ? *stats : ignoredStats;
    out = BuildStats{};

    if (!isSupported(n, cover)) {
        return false;
    }

    if (n <= 1) {
        if (n == 1) {
            sa[0] = 0;
        }

        return true;
    }

    buildWithCover<false>(text, n, sa, cover, out);

    return true;
}

int32_t buildBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t* sa, int cover, BuildStats* stats) {
    BuildStats ignoredStats;
    BuildStats& out = stats ? *stats : ignoredStats;
    out = BuildStats{};

    if (!isSupported(n, cover)) {
        return -1;
    }

    if (n <= 1) {
        if (n == 1) {
            bwt[0] = text[0];
        }

        return n;
    }

    buildWithCover<true>(text, n, sa, cover, out);

    return packBwt(text, n, sa, bwt);
}

int32_t setTaggedInductionMinSize(int32_t minSize) {
    return std::exchange(taggedInductionMinSize(), minSize);
}
