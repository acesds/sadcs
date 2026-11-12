/*
 * selftest.h - Built-in adversarial test suite for the construction and the checker.
 * Small inputs (empty, single bytes, runs, periods around the cover sizes, Fibonacci and Thue-Morse words,
 * random text over tiny to full alphabets) are compared against a naive std::sort suffix array for both covers.
 * Large ones (a few MB, enough to exercise the parallel task paths) are checked with the fast checker,
 * compared across covers, and compared against a single-threaded build.
 * Finally the checker itself has to catch planted errors in every mode.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

// Runs every test, prints failures (and each case with verbose).
// Returns the failure count.
int runSelfTest(bool verbose);
