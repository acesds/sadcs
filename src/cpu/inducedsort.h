/*
 * inducedsort.h - Two-stage induced sorting (Itoh/Tanaka, as in libdivsufsort).

 * The enhanced two-stage sort works like this: A suffix is type S if it is lexicographically
 * smaller than the suffix after it, and type L if it is larger. An RMS (rightmost S) suffix
 * is an S-suffix followed by an L-suffix: the last S-suffix in a consecutive run of S-suffixes.
 * Only the RMS-suffixes (called B* in libdivsufsort) are sorted directly. Once they are sorted,
 * a right-to-left scan induces the remaining S-suffixes from their successors, inserting them
 * at the tails of their character buckets. A simple left-to-right scan then induces the L-suffixes,
 * inserting them at the heads of their buckets. Consequently, within each bucket for character c,
 * the L-suffixes come before the S-suffixes. The S-suffixes are grouped by their second character.
 * Among those sharing the prefix c0,c1, the RMS-suffixes come first: their successors start with
 * c1 and are L-type, which sorts before the S-type successors. The SA-IS algorithm uses a similar
 * induction scheme, but relies on LMS (leftmost S) suffixes instead.
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
#include <immintrin.h>
#include <omp.h>
#include "bucketsort.h"

constexpr int32_t kClassifyChunkMinSize = 1 << 16;
constexpr ptrdiff_t kInducePrefetchDistance = 32;
constexpr int32_t kInduceBlockSize = 1 << 18;
constexpr int32_t kInduceParallelMinSize = 1 << 13;

struct SuffixClasses {
    std::vector<int32_t> lFirst;     // by c: first slot of the L-suffixes starting with c
    std::vector<int32_t> sFirst;     // by c: first slot of the S-suffixes starting with c
    std::vector<int32_t> sPairFirst; // by c0 * 256 + c1: first slot of the S-suffixes starting with c0,c1
    int32_t rmsCount = 0;
};

// Finds the types of all suffixes in parallel chunks, right to left within each chunk.
// Marks the RMS positions in rmsBits, (n+31)/32 words.
inline SuffixClasses classifySuffixes(const uint8_t* text, int32_t n, uint32_t* rmsBits) {
    const int threadCount = omp_get_max_threads();
    const int32_t chunkSize = (std::max(kClassifyChunkMinSize, n / (4 * threadCount)) + 63) & ~63;
    const int32_t chunkCount = (n + chunkSize - 1) / chunkSize;

    SuffixClasses classes;

    // The type of each chunk's first suffix: +1=S, -1=L, 0 if the chunk is a single character.
    std::vector<int8_t> firstType(chunkCount);

#pragma omp parallel for
    for (int32_t chunk = 0; chunk < chunkCount; chunk++) {
        const int32_t first = chunk * chunkSize;
        const int32_t last = std::min(n, first + chunkSize);
        int32_t differing = first + 1;

        while (differing < last && text[differing] == text[first]) {
            differing++;
        }

        firstType[chunk] = differing == last ? 0 : (text[first] < text[differing] ? 1 : -1);
    }

    // the type of the suffix right after each chunk
    std::vector<uint8_t> nextIsS(chunkCount, 0);

    for (int32_t chunk = chunkCount - 1; chunk > 0; chunk--) {
        int8_t& type = firstType[chunk];

        if (type == 0) {
            const uint8_t c = text[chunk * chunkSize];
            const int32_t next = (chunk + 1) * chunkSize;

            if (next >= n) {
                type = -1;
            } else if (text[next] != c) {
                type = c < text[next] ? 1 : -1;
            } else {
                type = nextIsS[chunk] ? 1 : -1;
            }
        }

        nextIsS[chunk - 1] = type > 0;
    }

    // L-counts by first character, then S-counts by character pair.
    // One table per thread, and no more threads than chunks.
    constexpr int32_t kSPairCountsAt = 256;
    constexpr int32_t kCountsSize = kSPairCountsAt + 256 * 256;
    const int countThreads = std::min(threadCount, chunkCount);
    std::vector<int32_t> counts(kCountsSize);
    const std::unique_ptr<int32_t[]> threadCounts(new int32_t[static_cast<size_t>(countThreads) * kCountsSize]);
    int32_t rmsCount = 0;

#pragma omp parallel num_threads(countThreads) reduction(+ : rmsCount)
    {
        const int teamSize = omp_get_num_threads();
        int32_t* localCounts = threadCounts.get() + static_cast<size_t>(omp_get_thread_num()) * kCountsSize;
        int32_t localRms = 0;

        std::fill(localCounts, localCounts + kCountsSize, 0);

#pragma omp for schedule(dynamic, 1)
        for (int32_t chunk = 0; chunk < chunkCount; chunk++) {
            const int32_t first = chunk * chunkSize;
            const int32_t last = std::min(n, first + chunkSize);
            bool followerIsS = nextIsS[chunk];
            int32_t i = last - 1;

            if (last == n) {
                rmsBits[(n - 1) >> 5] = 0;  // the scan starts below n-1, maybe in an earlier word
                localCounts[text[n - 1]]++;
                followerIsS = false;
                i = n - 2;
            }

            // The RMS bits of a word collect in a register, which is stored when the
            // scan reaches the word's first position. Chunks start at multiples of 64.
            uint32_t wordBits = 0;

            for (; i >= first; i--) {
                const uint32_t c0 = text[i];
                const uint32_t c1 = text[i + 1];
                const bool isS = (c0 < c1) | ((c0 == c1) & followerIsS);

                localCounts[isS ? kSPairCountsAt + (c0 << 8 | c1) : c0]++;
                wordBits |= static_cast<uint32_t>(isS & !followerIsS) << (i & 31);
                followerIsS = isS;

                if ((i & 31) == 0) {
                    rmsBits[i >> 5] = wordBits;
                    localRms += std::popcount(wordBits);
                    wordBits = 0;
                }
            }
        }

        rmsCount += localRms;

        // The omp for above ends with a barrier, so all tables are complete.
#pragma omp for
        for (int32_t index = 0; index < kCountsSize; index++) {
            int32_t total = 0;

            for (int t = 0; t < teamSize; t++) {
                total += threadCounts[static_cast<size_t>(t) * kCountsSize + index];
            }

            counts[index] = total;
        }
    }

    const int32_t* lCount = counts.data();
    const int32_t* sPairCount = counts.data() + kSPairCountsAt;

    classes.lFirst.resize(256);
    classes.sFirst.resize(256);
    classes.sPairFirst.resize(256 * 256);
    classes.rmsCount = rmsCount;

    for (int32_t c0 = 0, offset = 0; c0 < 256; c0++) {
        classes.lFirst[c0] = offset;
        offset += lCount[c0];
        classes.sFirst[c0] = offset;

        for (int32_t c1 = 0; c1 < 256; c1++) {
            classes.sPairFirst[c0 * 256 + c1] = offset;
            offset += sPairCount[c0 * 256 + c1];
        }
    }

    return classes;
}

// For large texts, an entry j whose predecessor j-1 is L-type is stored as j|kPredecessorIsL
// during induction. Whoever places a suffix has just read its first character, and the one before
// it is nearly always in the same cache line, so the tag is cheap to set. The scans then skip the
// random text read for entries that induce nothing: the S-scan for tagged ones, the L-scan for
// untagged ones. The RMS suffixes are placed untagged. The L-scan clears the tags as it goes.
// Setting and clearing the tags costs more than the skipped reads save while the text fits in
// the cache, so smaller texts are induced untagged.
constexpr int32_t kPredecessorIsL = INT32_MIN;
constexpr int32_t kTaggedInductionMinSize = 1 << 26;

// The size from which induction is tagged; the selftest lowers it to cover the tagged scans.
inline int32_t& taggedInductionMinSize() {
    static int32_t minSize = kTaggedInductionMinSize;
    return minSize;
}

// Built as a BWT, an entry is replaced by its suffix's preceding character, stored as ~c,
// as soon as no scan needs it any more: by the S-scan once it has induced from it, by the
// L-scan for all the others. Suffix 0 has no preceding character; it stays 0 and marks the
// primary index required for the inverse BWT.
constexpr int32_t kBwtCharMin = -256;

inline int32_t bwtChar(uint8_t c) {
    return ~static_cast<int32_t>(c);
}

// Suffix k of type S, with ck = T[k]: k-1 is L if T[k-1]>T[k].
inline int32_t tagSSuffix(const uint8_t* text, int32_t k, uint8_t ck) {
    const bool predecessorIsL = (k > 0) & (text[k - (k > 0)] > ck);
    return k | (predecessorIsL ? kPredecessorIsL : 0);
}

// Suffix k of type L, with ck = T[k]: k-1 is L if T[k-1]>=T[k].
inline int32_t tagLSuffix(const uint8_t* text, int32_t k, uint8_t ck) {
    const bool predecessorIsL = (k > 0) & (text[k - (k > 0)] >= ck);
    return k | (predecessorIsL ? kPredecessorIsL : 0);
}

// Built as a BWT, an L-suffix k whose predecessor is S induces nothing in the L-scan,
// so it is placed as its BWT character T[k-1] right away.
inline int32_t tagLSuffixForBwt(const uint8_t* text, int32_t k, uint8_t ck) {
    const uint8_t preceding = text[k - (k > 0)];
    const int32_t tagged = k | kPredecessorIsL;
    return k == 0 ? 0 : (preceding >= ck ? tagged : bwtChar(preceding));
}

// Moves the sorted RMS suffixes from the front of sa, grouped by bucketSort(),
// to the front of their S-pair buckets.
inline void placeRmsSuffixes(int32_t* sa, const std::vector<int32_t>& bucketStarts, const SuffixClasses& classes) {
    for (int32_t pair = 256 * 256 - 1; pair >= 0; pair--) {
        const int32_t bucket = (pair >> 8) * 257 + (pair & 255) + 1;
        const int32_t* sourceFirst = sa + bucketStarts[bucket];
        const int32_t* sourceLast = sa + bucketStarts[bucket + 1];

        if (sourceFirst != sourceLast) {
            std::copy_backward(sourceFirst, sourceLast, sa + classes.sPairFirst[pair] + (sourceLast - sourceFirst));
        }
    }
}

inline void prefetchSuffix(const uint8_t* text, int32_t position) {
    _mm_prefetch(reinterpret_cast<const char*>(text + position - 1), _MM_HINT_T0);
}

struct InduceStep {
    int32_t first;
    int32_t last;
    bool parallel;
};

template <bool Tagged, bool Bwt>
void induceSSuffixes(const uint8_t* text, int32_t n, int32_t* sa, const SuffixClasses& classes) {
    // Cursors start at the end of each pair bucket; the last pair of c ends where c+1 begins.
    std::vector<int32_t> pairCursor(256 * 256);

    for (int32_t c0 = 0; c0 < 256; c0++) {
        for (int32_t c1 = 0; c1 < 255; c1++) {
            pairCursor[c0 * 256 + c1] = classes.sPairFirst[c0 * 256 + c1 + 1];
        }

        pairCursor[c0 * 256 + 255] = c0 < 255 ? classes.lFirst[c0 + 1] : n;
    }

    int32_t* cursor = pairCursor.data();
    const size_t bufferSize = static_cast<size_t>(std::min(kInduceBlockSize, n));
    const std::unique_ptr<int32_t[]> inducedSuffixes(new int32_t[bufferSize]);
    const std::unique_ptr<uint8_t[]> inducedChars(new uint8_t[bufferSize]);
    std::vector<int32_t> threadCursors(static_cast<size_t>(omp_get_max_threads()) * 256);
    InduceStep step{};

    // The predecessor whose character an entry needs, 0 if there is none to read: untagged entries other than 0 need it.
    auto source = [](int32_t entry) {
        return entry > 0 ? entry - 1 : 0;
    };

    auto induced = [text](int32_t k, uint8_t ck) {
        return Tagged ? tagSSuffix(text, k, ck) : k;
    };

    auto scanAlone = [&](int32_t c, int32_t first, int32_t last) {
        for (int32_t i = last - 1; i >= first; i--) {
            if (i - kInducePrefetchDistance >= 0) {
                prefetchSuffix(text, source(sa[i - kInducePrefetchDistance]) + 1);
            }

            const int32_t entry = sa[i];

            if (entry > 0) {
                const int32_t k = entry - 1;
                const uint8_t c0 = text[k];

                if (c0 <= c) {
                    sa[--cursor[c0 * 256 + c]] = induced(k, c0);

                    if constexpr (Bwt) {
                        sa[i] = bwtChar(c0);
                    }
                } else if (Tagged) {
                    sa[i] = entry | kPredecessorIsL;
                }
            }
        }
    };

#pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        const int threadCount = omp_get_num_threads();
        int32_t* myCursor = threadCursors.data() + static_cast<size_t>(thread) * 256;

        for (int32_t c = 255; c >= 0; c--) {
            const int32_t regionFirst = classes.sFirst[c];
            int32_t position = c < 255 ? classes.lFirst[c + 1] : n;

            while (position > regionFirst) {
#pragma omp single
                {
                    const int32_t filledFirst = std::max(regionFirst, cursor[c * 256 + c]);
                    const int32_t blockFirst = std::max(filledFirst, position - kInduceBlockSize);

                    step.parallel = position - blockFirst >= kInduceParallelMinSize;

                    if (step.parallel) {
                        step.first = blockFirst;
                    } else {
                        step.first = std::max(regionFirst, position - kInduceBlockSize);
                        scanAlone(c, step.first, position);
                    }
                }

                const InduceStep current = step;
#pragma omp barrier

                if (current.parallel) {
                    // Thread 0 takes the top share, as the scan runs downwards.
                    const int64_t size = position - current.first;
                    const int32_t shareLast = position - static_cast<int32_t>(size * thread / threadCount);
                    const int32_t shareFirst = position - static_cast<int32_t>(size * (thread + 1) / threadCount);
                    int32_t* mySuffixes = inducedSuffixes.get() + (shareFirst - current.first);
                    uint8_t* myChars = inducedChars.get() + (shareFirst - current.first);
                    int32_t inducedCount = 0;

                    std::fill(myCursor, myCursor + 256, 0);

                    for (int32_t i = shareLast - 1; i >= shareFirst; i--) {
                        if (i - kInducePrefetchDistance >= shareFirst) {
                            prefetchSuffix(text, source(sa[i - kInducePrefetchDistance]) + 1);
                        }

                        const int32_t entry = sa[i];
                        const int32_t k = source(entry);
                        const uint8_t c0 = text[k];
                        const int32_t induces = (entry > 0) & (c0 <= c);

                        if constexpr (Tagged || Bwt) {
                            const int32_t kept = Tagged ? entry | (((entry > 0) & (c0 > c)) ? kPredecessorIsL : 0) : entry;
                            sa[i] = Bwt && induces ? bwtChar(c0) : kept;
                        }

                        mySuffixes[inducedCount] = induced(k, c0);
                        myChars[inducedCount] = c0;
                        inducedCount += induces;
                        myCursor[c0] += induces;
                    }

#pragma omp barrier
#pragma omp single
                    for (int32_t c0 = 0; c0 <= c; c0++) {
                        for (int t = 0; t < threadCount; t++) {
                            int32_t& slot = threadCursors[static_cast<size_t>(t) * 256 + c0];
                            const int32_t count = slot;

                            slot = cursor[c0 * 256 + c];
                            cursor[c0 * 256 + c] -= count;
                        }
                    }

                    for (int32_t k = 0; k < inducedCount; k++) {
                        sa[--myCursor[myChars[k]]] = mySuffixes[k];
                    }

#pragma omp barrier
                }

                position = current.first;
            }
        }
    }
}

template <bool Tagged, bool Bwt>
void induceLSuffixes(const uint8_t* text, int32_t n, int32_t* sa, const SuffixClasses& classes) {
    std::vector<int32_t> bucketCursor(classes.lFirst);
    int32_t* cursor = bucketCursor.data();
    const size_t bufferSize = static_cast<size_t>(std::min(kInduceBlockSize, n));
    const std::unique_ptr<int32_t[]> inducedSuffixes(new int32_t[bufferSize]);
    const std::unique_ptr<uint8_t[]> inducedChars(new uint8_t[bufferSize]);
    std::vector<int32_t> threadCursors(static_cast<size_t>(omp_get_max_threads()) * 256);
    InduceStep step{};

    // Built as a BWT, characters are negative too, but above the tagged entries.
    auto isTagged = [](int32_t entry) {
        return entry < (Bwt ? kBwtCharMin : 0);
    };

    // The predecessor whose character an entry needs, 0 if there is none to read.
    // Only tagged entries need it (they're never 0).
    auto source = [=](int32_t entry) {
        if constexpr (Tagged) {
            return isTagged(entry) ? (entry & INT32_MAX) - 1 : 0;
        } else {
            return entry > 0 ? entry - 1 : 0;
        }
    };

    auto induced = [=](int32_t k, [[maybe_unused]] uint8_t ck) {
        if constexpr (Tagged) {
            return Bwt ? tagLSuffixForBwt(text, k, ck) : tagLSuffix(text, k, ck);
        } else {
            return k;
        }
    };

    sa[cursor[text[n - 1]]++] = induced(n - 1, text[n - 1]);

    auto scanAlone = [&]([[maybe_unused]] int32_t lowestInducing, int32_t first, int32_t last) {
        for (int32_t i = first; i < last; i++) {
            if (i + kInducePrefetchDistance < n) {
                prefetchSuffix(text, source(sa[i + kInducePrefetchDistance]) + 1);
            }

            const int32_t entry = sa[i];

            if constexpr (Tagged) {
                if constexpr (!Bwt) {
                    sa[i] = entry & INT32_MAX;
                }

                if (isTagged(entry)) {
                    const int32_t k = (entry & INT32_MAX) - 1;
                    const uint8_t c0 = text[k];

                    sa[cursor[c0]++] = induced(k, c0);

                    if constexpr (Bwt) {
                        sa[i] = bwtChar(c0);
                    }
                }
            } else if (entry > 0) {
                const int32_t k = entry - 1;
                const uint8_t c0 = text[k];

                if (c0 >= lowestInducing) {
                    sa[cursor[c0]++] = k;
                }

                if constexpr (Bwt) {
                    sa[i] = bwtChar(c0);
                }
            }
        }
    };

#pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        const int threadCount = omp_get_num_threads();
        int32_t* myCursor = threadCursors.data() + static_cast<size_t>(thread) * 256;

        for (int32_t part = 0; part < 2 * 256; part++) {
            const int32_t c = part / 2;
            const bool isLPart = part % 2 == 0;
            const int32_t partLast = isLPart ? classes.sFirst[c] : (c < 255 ? classes.lFirst[c + 1] : n);
            const int32_t lowestInducing = isLPart ? c : c + 1;
            int32_t position = isLPart ? classes.lFirst[c] : classes.sFirst[c];

            while (position < partLast) {
#pragma omp single
                {
                    const int32_t filledLast = isLPart ? std::min(partLast, cursor[c]) : partLast;
                    const int32_t blockLast = std::min(filledLast, position + kInduceBlockSize);

                    step.parallel = blockLast - position >= kInduceParallelMinSize;

                    if (step.parallel) {
                        step.last = blockLast;
                    } else {
                        step.last = std::min(partLast, position + kInduceBlockSize);
                        scanAlone(lowestInducing, position, step.last);
                    }
                }

                const InduceStep current = step;
#pragma omp barrier

                if (current.parallel) {
                    const int64_t size = current.last - position;
                    const int32_t shareFirst = position + static_cast<int32_t>(size * thread / threadCount);
                    const int32_t shareLast = position + static_cast<int32_t>(size * (thread + 1) / threadCount);
                    int32_t* mySuffixes = inducedSuffixes.get() + (shareFirst - position);
                    uint8_t* myChars = inducedChars.get() + (shareFirst - position);
                    int32_t inducedCount = 0;

                    std::fill(myCursor, myCursor + 256, 0);

                    for (int32_t i = shareFirst; i < shareLast; i++) {
                        if (i + kInducePrefetchDistance < shareLast) {
                            prefetchSuffix(text, source(sa[i + kInducePrefetchDistance]) + 1);
                        }

                        const int32_t entry = sa[i];
                        const int32_t k = source(entry);
                        const uint8_t c0 = text[k];
                        int32_t induces;

                        if constexpr (Tagged) {
                            induces = isTagged(entry);
                            sa[i] = Bwt ? (induces ? bwtChar(c0) : entry) : entry & INT32_MAX;
                        } else {
                            induces = (entry > 0) & (c0 >= lowestInducing);

                            if constexpr (Bwt) {
                                sa[i] = entry > 0 ? bwtChar(c0) : entry;
                            }
                        }

                        mySuffixes[inducedCount] = induced(k, c0);
                        myChars[inducedCount] = c0;
                        inducedCount += induces;
                        myCursor[c0] += induces;
                    }

#pragma omp barrier
#pragma omp single
                    for (int32_t c0 = lowestInducing; c0 < 256; c0++) {
                        for (int t = 0; t < threadCount; t++) {
                            int32_t& slot = threadCursors[static_cast<size_t>(t) * 256 + c0];
                            const int32_t count = slot;

                            slot = cursor[c0];
                            cursor[c0] += count;
                        }
                    }

                    for (int32_t k = 0; k < inducedCount; k++) {
                        sa[myCursor[myChars[k]]++] = mySuffixes[k];
                    }

#pragma omp barrier
                }

                position = current.last;
            }
        }
    }
}

// Induces all other suffixes from the placed RMS-suffixes.
// Built as a BWT, sa ends up holding ~T[j-1] in place of each suffix j (0 for suffix 0).
template <bool Bwt>
void induceSuffixes(const uint8_t* text, int32_t n, int32_t* sa, const SuffixClasses& classes) {
    if (n >= taggedInductionMinSize()) {
        induceSSuffixes<true, Bwt>(text, n, sa, classes);
        induceLSuffixes<true, Bwt>(text, n, sa, classes);
    } else {
        induceSSuffixes<false, Bwt>(text, n, sa, classes);
        induceLSuffixes<false, Bwt>(text, n, sa, classes);
    }
}
