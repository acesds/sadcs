/*
 * checker.cpp - Implementation of the suffix array and BWT checker.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "checker.h"
#include <algorithm>
#include <vector>
#include "simd.h"

namespace {
    constexpr int64_t kNoFailure = INT64_MAX;

    CheckResult pass() {
        return CheckResult{true, -1, nullptr};
    }

    CheckResult fail(int64_t index, const char* reason) {
        return CheckResult{false, index, reason};
    }

    CheckResult failIfFound(int64_t firstFailure, const char* reason) {
        return firstFailure == kNoFailure ? pass() : fail(firstFailure, reason);
    }

    CheckResult checkPermutation(int32_t n, const int32_t* sa) {
        std::vector<uint8_t> seen((static_cast<size_t>(n) + 7) / 8, 0);

        for (int32_t i = 0; i < n; i++) {
            const int32_t position = sa[i];

            if (position < 0 || position >= n) {
                return fail(i, "entry out of range");
            }

            uint8_t& byte = seen[static_cast<size_t>(position) >> 3];
            const uint8_t bit = static_cast<uint8_t>(1u << (position & 7));

            if (byte & bit) {
                return fail(i, "duplicate entry");
            }

            byte |= bit;
        }

        return pass();
    }

    CheckResult checkFast(const uint8_t* text, int32_t n, const int32_t* sa) {
        std::vector<int32_t> inverse(static_cast<size_t>(n));
        int32_t* isa = inverse.data();
        int64_t firstOutOfRange = kNoFailure;

#pragma omp parallel for reduction(min : firstOutOfRange)
        for (int32_t i = 0; i < n; i++) {
            const int32_t position = sa[i];

            if (position < 0 || position >= n) {
                firstOutOfRange = std::min(firstOutOfRange, static_cast<int64_t>(i));
            } else {
#pragma omp atomic write
                isa[position] = i;
            }
        }

        if (firstOutOfRange != kNoFailure) {
            return fail(firstOutOfRange, "entry out of range");
        }

        // Of two equal entries, only one can own the inverse slot.
        int64_t firstDuplicate = kNoFailure;

#pragma omp parallel for reduction(min : firstDuplicate)
        for (int32_t i = 0; i < n; i++) {
            if (isa[sa[i]] != i) {
                firstDuplicate = std::min(firstDuplicate, static_cast<int64_t>(i));
            }
        }

        if (firstDuplicate != kNoFailure) {
            return fail(firstDuplicate, "duplicate entry");
        }

        int64_t firstMisordered = kNoFailure;

#pragma omp parallel for reduction(min : firstMisordered)
        for (int32_t i = 0; i < n - 1; i++) {
            const int32_t left = sa[i];
            const int32_t right = sa[i + 1];
            bool inOrder = text[left] < text[right];

            if (text[left] == text[right]) {
                const int32_t emptySuffixRank = -1;
                const int32_t leftNextRank = left + 1 < n ? isa[left + 1] : emptySuffixRank;
                const int32_t rightNextRank = right + 1 < n ? isa[right + 1] : emptySuffixRank;
                inOrder = leftNextRank < rightNextRank;
            }

            if (!inOrder) {
                firstMisordered = std::min(firstMisordered, static_cast<int64_t>(i));
            }
        }

        return failIfFound(firstMisordered, "SA[index] and SA[index + 1] out of order");
    }

    // Walking SA in order, the predecessor of each suffix must be the next suffix of its first-character bucket.
    CheckResult checkLowMemory(const uint8_t* text, int32_t n, const int32_t* sa) {
        const CheckResult permutation = checkPermutation(n, sa);

        if (!permutation.ok) {
            return permutation;
        }

        for (int32_t i = 1; i < n; i++) {
            if (text[sa[i - 1]] > text[sa[i]]) {
                return fail(i - 1, "first characters out of order");
            }
        }

        constexpr int32_t kBucketExhausted = -1;
        int32_t nextSlot[256] = {};

        for (int32_t i = 0; i < n; i++) {
            nextSlot[text[i]]++;
        }

        for (int32_t c = 0, offset = 0; c < 256; c++) {
            const int32_t count = nextSlot[c];
            nextSlot[c] = offset;
            offset += count;
        }

        // The last suffix is the smallest of its bucket and has no successor to find it from.
        const int32_t lastSuffixSlot = nextSlot[text[n - 1]]++;

        for (int32_t i = 0; i < n; i++) {
            int32_t predecessor;
            int32_t expectedSlot;

            if (sa[i] > 0) {
                predecessor = sa[i] - 1;
                expectedSlot = nextSlot[text[predecessor]];
            } else {
                predecessor = n - 1;
                expectedSlot = lastSuffixSlot;
            }

            if (expectedSlot == kBucketExhausted || sa[expectedSlot] != predecessor) {
                return fail(i, "predecessor of SA[index] not where its bucket expects it");
            }

            if (expectedSlot != lastSuffixSlot) {
                const uint8_t bucket = text[predecessor];
                int32_t& slot = ++nextSlot[bucket];

                if (slot >= n || text[sa[slot]] != bucket) {
                    slot = kBucketExhausted;
                }
            }
        }

        return pass();
    }

    CheckResult checkNaive(const uint8_t* text, int32_t n, const int32_t* sa) {
        const CheckResult permutation = checkPermutation(n, sa);

        if (!permutation.ok) {
            return permutation;
        }

        int64_t firstMisordered = kNoFailure;

#pragma omp parallel for schedule(dynamic, 4096) reduction(min : firstMisordered)
        for (int32_t i = 0; i < n - 1; i++) {
            const int32_t left = sa[i];
            const int32_t right = sa[i + 1];
            const int32_t comparable = std::min(n - left, n - right);
            const int32_t common = commonPrefixLength(text + left, text + right, comparable);
            const bool inOrder = common < comparable ? text[left + common] < text[right + common] : left > right;

            if (!inOrder) {
                firstMisordered = std::min(firstMisordered, static_cast<int64_t>(i));
            }
        }

        return failIfFound(firstMisordered, "SA[index] and SA[index + 1] out of order");
    }
}

CheckResult checkBwt(const uint8_t* text, int32_t n, const uint8_t* bwt, int32_t primary) {
    if (n == 0) {
        return primary == 0 ? pass() : fail(-1, "primary index out of range");
    }

    if (primary < 1 || primary > n) {
        return fail(-1, "primary index out of range");
    }

    // With the sentinel, the matrix has n+1 rows.
    // Row r<primary holds bwt[r], row primary the sentinel, and row r>primary bwt[r - 1].
    // lf[u] is the row that the row of bwt[u] maps to: one for the sentinel's row, plus the
    // smaller characters, plus the earlier occurrences of the same one.
    std::vector<int32_t> lf(static_cast<size_t>(n));
    int32_t next[256] = {};

    for (int32_t u = 0; u < n; u++) {
        next[bwt[u]]++;
    }

    for (int32_t c = 0, sum = 1; c < 256; c++) {
        const int32_t count = next[c];
        next[c] = sum;
        sum += count;
    }

    for (int32_t u = 0; u < n; u++) {
        lf[u] = next[bwt[u]]++;
    }

    // Row 0 is the sentinel's suffix. Its character is T[n-1], and following the mapping
    // spells the text backwards until it reaches suffix 0, the row of the sentinel character.
    int32_t row = 0;

    for (int32_t i = n - 1; i >= 0; i--) {
        if (row == primary) {
            return fail(i, "inverse BWT ends early at text position index");
        }

        const int32_t u = row < primary ? row : row - 1;

        if (bwt[u] != text[i]) {
            return fail(i, "inverse BWT differs from the text at position index");
        }

        row = lf[u];
    }

    return row == primary ? pass() : fail(0, "inverse BWT doesn't end at the primary index");
}

CheckResult checkSuffixArray(const uint8_t* text, int32_t n, const int32_t* sa, CheckMode mode) {
    if (n == 0) {
        return pass();
    }

    switch (mode) {
        case CheckMode::LowMemory:
            return checkLowMemory(text, n, sa);
        case CheckMode::Naive:
            return checkNaive(text, n, sa);
        case CheckMode::Fast:
        default:
            return checkFast(text, n, sa);
    }
}
