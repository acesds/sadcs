/*
 * checker.h - Suffix array and BWT verification.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstdint>

enum class CheckMode { Fast, LowMemory, Naive };

struct CheckResult {
    bool ok;
    int64_t index;       // SA index of the first failure, -1 if ok
    const char* reason;  // nullptr if ok
};

// Note that text must be followed by kTextPadding zero bytes.
CheckResult checkSuffixArray(const uint8_t* text, int32_t n, const int32_t* sa, CheckMode mode);

// Inverts a BWT in buildBwt()'s convention and compares it with text.
// Sequential, 4n extra bytes.
// On failure, index is the text position where the inverse goes wrong.
CheckResult checkBwt(const uint8_t* text, int32_t n, const uint8_t* bwt, int32_t primary);
