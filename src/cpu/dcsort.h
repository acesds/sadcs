/*
 * dcsort.h - Suffix array construction with a difference cover sample (Burkhardt/Kärkkäinen).
 *
 * Ranks the sample suffixes, then bucket- and string-sorts the RMS suffixes to V-1 characters.
 * Suffixes i and j still tied are ordered by the sample ranks at i+k and j+k, with k from the cover.
 * The other suffixes are induced from the RMS ones. The working arrays live in the back half of sa,
 * which the RMS suffixes leave free, so apart from the text and SA only small per-thread buffers and
 * tables are allocated.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>

// Positions plus a few cover periods of lookahead must fit an int32_t.
constexpr int64_t kMaxTextSize = INT32_MAX - 1024;

struct BuildStats {
    int32_t sampleSize = 0;                 // number of sample positions
    double sampleStringSortSeconds = 0.0;   // sorting the sample suffixes by their first characters
    double sampleDoublingSeconds = 0.0;     // prefix doubling on the sample's remaining ties
    double mainSortSeconds = 0.0;           // everything else: RMS sort, tie groups and induction
};

// Builds the suffix array of text[0, n) into sa[0, n) with a difference cover modulo cover (64, 128 or 256).
// text must be followed by kTextPadding zero bytes (simd.h).
// Runs on omp_get_max_threads() threads and, apart from sa, allocates only small buffers and tables.
// Returns false for an unsupported cover or n over kMaxTextSize.
bool buildSuffixArray(const uint8_t* text, int32_t n, int32_t* sa, int cover, BuildStats* stats = nullptr);

// Builds the Burrows-Wheeler transform of text[0, n) into bwt[0, n), with sa (n entries) as working space.
// bwt may be text, and text must be followed by kTextPadding zero bytes.
// The convention is libsais's and divsufsort's: the BWT of text plus a sentinel, without the sentinel.
// Returns the primary index (the sentinel's row), or -1 for an unsupported cover or n over kMaxTextSize.
int32_t buildBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t* sa, int cover, BuildStats* stats = nullptr);

// For the selftest: the text size from which induction tags entries with their predecessor's type (usually 64 MiB).
// Returns the previous value.
int32_t setTaggedInductionMinSize(int32_t minSize);
