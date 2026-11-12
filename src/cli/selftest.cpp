/*
 * selftest.cpp - The guts of selftest.h.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "selftest.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>
#include "checker.h"
#include "dcsort.h"
#include "gpusa.h"
#include "simd.h"

namespace {
    constexpr int kCovers[] = {128, 64, 256};  // the default first: it gets the full check
    constexpr int32_t kLargeTextSize = 1 << 22;

    // splitmix64 with fixed seeds, so every run tests the same inputs.
    struct Rng {
        uint64_t state;

        uint64_t next() {
            uint64_t z = (state += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }

        uint32_t nextBelow(uint32_t bound) {
            return static_cast<uint32_t>(next() % bound);
        }
    };

    struct Suite {
        bool verbose;
        int cases = 0;
        int failures = 0;

        void reportFailure(const std::string& name, const char* what, int64_t index) {
            failures++;
            printf("  FAIL %s: %s (index %lld)\n", name.c_str(), what, static_cast<long long>(index));
        }
    };

    std::vector<uint8_t> withPadding(const std::vector<uint8_t>& input) {
        std::vector<uint8_t> padded(input);
        padded.resize(input.size() + kTextPadding, 0);
        return padded;
    }

    std::vector<uint8_t> randomText(Rng& rng, int32_t n, uint32_t alphabetSize) {
        std::vector<uint8_t> text(n);

        for (uint8_t& c : text) {
            c = static_cast<uint8_t>(alphabetSize == 256 ? rng.nextBelow(256) : 'a' + rng.nextBelow(alphabetSize));
        }

        return text;
    }

    std::vector<uint8_t> periodicText(Rng& rng, int32_t n, int32_t period) {
        const std::vector<uint8_t> pattern = randomText(rng, period, 4);
        std::vector<uint8_t> text(n);

        for (int32_t i = 0; i < n; i++) {
            text[i] = pattern[i % period];
        }

        return text;
    }

    std::vector<uint8_t> fibonacciText(int32_t n, uint8_t a, uint8_t b) {
        std::vector<uint8_t> shorter{a};
        std::vector<uint8_t> longer{a, b};

        while (static_cast<int32_t>(longer.size()) < n) {
            std::vector<uint8_t> next(longer);
            next.insert(next.end(), shorter.begin(), shorter.end());
            shorter = std::move(longer);
            longer = std::move(next);
        }

        longer.resize(n);

        return longer;
    }

    std::vector<uint8_t> thueMorseText(int32_t n) {
        std::vector<uint8_t> text(n);

        for (int32_t i = 0; i < n; i++) {
            text[i] = static_cast<uint8_t>('a' + (std::popcount(static_cast<uint32_t>(i)) & 1));
        }

        return text;
    }

    std::vector<uint8_t> textOf(const char* s) {
        return std::vector<uint8_t>(s, s + strlen(s));
    }

    std::vector<int32_t> naiveSuffixArray(const uint8_t* text, int32_t n) {
        std::vector<int32_t> sa(n);

        std::iota(sa.begin(), sa.end(), 0);
        std::sort(sa.begin(), sa.end(), [&](int32_t p, int32_t q) {
            const int32_t comparable = std::min(n - p, n - q);
            const int32_t common = commonPrefixLength(text + p, text + q, comparable);
            return common < comparable ? text[p + common] < text[q + common] : p > q;
        });

        return sa;
    }

    // Index of the first differing entry, -1 if equal.
    template <class T>
    int64_t firstMismatch(const std::vector<T>& a, const std::vector<T>& b) {
        const auto mismatch = std::mismatch(a.begin(), a.end(), b.begin());
        return mismatch.first == a.end() ? -1 : mismatch.first - a.begin();
    }

    struct Bwt {
        std::vector<uint8_t> bytes;
        int32_t primary = 0;
    };

    // The BWT in buildBwt()'s convention, read off a suffix array: the sentinel's row first,
    // then one row per suffix, except suffix 0's, whose index plus one is the primary index.
    Bwt bwtFromSuffixArray(const uint8_t* text, int32_t n, const std::vector<int32_t>& sa) {
        Bwt bwt;

        if (n == 0) {
            return bwt;
        }

        bwt.bytes.push_back(text[n - 1]);

        for (int32_t row = 0; row < n; row++) {
            if (sa[row] == 0) {
                bwt.primary = row + 1;
            } else {
                bwt.bytes.push_back(text[sa[row] - 1]);
            }
        }

        return bwt;
    }

    // buildBwt() into a buffer of its own, or in place of a copy of the text.
    Bwt buildBwtOf(const std::vector<uint8_t>& padded, int32_t n, int cover, bool inPlace = false) {
        std::vector<int32_t> sa(n);
        Bwt bwt;

        if (inPlace) {
            std::vector<uint8_t> text(padded);
            bwt.primary = buildBwt(text.data(), n, text.data(), sa.data(), cover);
            bwt.bytes.assign(text.begin(), text.begin() + n);
        } else {
            bwt.bytes.resize(n);
            bwt.primary = buildBwt(padded.data(), n, bwt.bytes.data(), sa.data(), cover);
        }

        return bwt;
    }

    // Reports a difference from expected; true if there is none.
    bool compareBwt(Suite& suite, const std::string& name, const Bwt& expected, const Bwt& actual) {
        if (actual.primary != expected.primary) {
            suite.reportFailure(name, "BWT has the wrong primary index", actual.primary);
            return false;
        }

        if (const int64_t mismatch = firstMismatch(expected.bytes, actual.bytes); mismatch >= 0) {
            suite.reportFailure(name, "BWT differs", mismatch);
            return false;
        }

        return true;
    }

    // Every cover against the naive suffix array, plus the two cheap checkers.
    void testSmall(Suite& suite, const std::string& name, const std::vector<uint8_t>& input) {
        const std::vector<uint8_t> padded = withPadding(input);
        const int32_t n = static_cast<int32_t>(input.size());
        const std::vector<int32_t> expected = naiveSuffixArray(padded.data(), n);

        for (const int cover : kCovers) {
            const std::string label = name + " cover " + std::to_string(cover);
            std::vector<int32_t> sa(n);

            suite.cases++;

            if (!buildSuffixArray(padded.data(), n, sa.data(), cover)) {
                suite.reportFailure(label, "build refused", -1);
                continue;
            }

            if (const int64_t mismatch = firstMismatch(sa, expected); mismatch >= 0) {
                suite.reportFailure(label, "differs from naive suffix array", mismatch);
                continue;
            }

            const CheckResult fast = checkSuffixArray(padded.data(), n, sa.data(), CheckMode::Fast);
            const CheckResult lowMemory = checkSuffixArray(padded.data(), n, sa.data(), CheckMode::LowMemory);

            if (!fast.ok || !lowMemory.ok) {
                suite.reportFailure(
                    label, "checker rejects the correct suffix array", fast.ok ? lowMemory.index : fast.index);
            }
        }

        const Bwt expectedBwt = bwtFromSuffixArray(padded.data(), n, expected);

        for (const int cover : kCovers) {
            const std::string label = name + " BWT cover " + std::to_string(cover);
            const Bwt bwt = buildBwtOf(padded, n, cover, cover == kCovers[0]);

            suite.cases++;

            if (compareBwt(suite, label, expectedBwt, bwt)) {
                if (const CheckResult check = checkBwt(padded.data(), n, bwt.bytes.data(), bwt.primary); !check.ok) {
                    suite.reportFailure(label, "BWT check rejects the correct BWT", check.index);
                }
            }
        }

        if (suite.verbose) {
            printf("  ok   %s (n=%d)\n", name.c_str(), n);
        }
    }

    // Fast checker on the first cover; the other covers and a single-threaded build must agree,
    // and so must the BWT builds and the build with tagged induction, which normally only texts of 64 MiB and more get.
    void testLarge(Suite& suite, const std::string& name, const std::vector<uint8_t>& input) {
        const std::vector<uint8_t> padded = withPadding(input);
        const int32_t n = static_cast<int32_t>(input.size());
        std::vector<int32_t> expected(n);
        std::vector<int32_t> other(n);

        suite.cases++;

        buildSuffixArray(padded.data(), n, expected.data(), kCovers[0]);
        const CheckResult check = checkSuffixArray(padded.data(), n, expected.data(), CheckMode::Fast);

        if (!check.ok) {
            suite.reportFailure(name, check.reason, check.index);
            return;
        }

        for (const int cover : kCovers) {
            if (cover == kCovers[0]) {
                continue;
            }

            buildSuffixArray(padded.data(), n, other.data(), cover);

            if (const int64_t mismatch = firstMismatch(expected, other); mismatch >= 0) {
                suite.reportFailure(name + " cover " + std::to_string(cover), "disagrees with the first cover", mismatch);
                return;
            }
        }

        const int threads = omp_get_max_threads();
        omp_set_num_threads(1);
        buildSuffixArray(padded.data(), n, other.data(), kCovers[0]);
        const Bwt singleThreadBwt = buildBwtOf(padded, n, kCovers[0]);
        omp_set_num_threads(threads);

        if (const int64_t mismatch = firstMismatch(expected, other); mismatch >= 0) {
            suite.reportFailure(name, "single-threaded build disagrees", mismatch);
            return;
        }

        const int32_t untaggedMinSize = setTaggedInductionMinSize(0);
        buildSuffixArray(padded.data(), n, other.data(), kCovers[0]);
        const Bwt taggedBwt = buildBwtOf(padded, n, kCovers[0]);
        setTaggedInductionMinSize(untaggedMinSize);

        if (const int64_t mismatch = firstMismatch(expected, other); mismatch >= 0) {
            suite.reportFailure(name, "tagged induction disagrees", mismatch);
            return;
        }

        const Bwt expectedBwt = bwtFromSuffixArray(padded.data(), n, expected);
        const bool bwtOk = compareBwt(suite, name + " BWT", expectedBwt, buildBwtOf(padded, n, kCovers[0])) &&
                           compareBwt(suite, name + " BWT cover " + std::to_string(kCovers[1]), expectedBwt,
                               buildBwtOf(padded, n, kCovers[1], true)) &&
                           compareBwt(suite, name + " BWT single-threaded", expectedBwt, singleThreadBwt) &&
                           compareBwt(suite, name + " BWT tagged", expectedBwt, taggedBwt);

        if (bwtOk && suite.verbose) {
            printf("  ok   %s (n=%d)\n", name.c_str(), n);
        }
    }

    // Every check mode has to reject each planted error.
    void testCheckerCatches(Suite& suite) {
        Rng rng{7};
        const std::vector<uint8_t> input = randomText(rng, 3000, 3);
        const std::vector<uint8_t> padded = withPadding(input);
        const int32_t n = static_cast<int32_t>(input.size());
        const std::vector<int32_t> good = naiveSuffixArray(padded.data(), n);

        struct Mutation {
            const char* name;
            void (*apply)(std::vector<int32_t>&);
        };

        const Mutation mutations[] = {
            {"swapped neighbours", [](std::vector<int32_t>& sa) { std::swap(sa[1000], sa[1001]); }},
            {"swapped far apart", [](std::vector<int32_t>& sa) { std::swap(sa[10], sa[2900]); }},
            {"duplicate entry", [](std::vector<int32_t>& sa) { sa[500] = sa[1500]; }},
            {"out of range", [](std::vector<int32_t>& sa) { sa[42] = static_cast<int32_t>(sa.size()); }},
            {"negative entry", [](std::vector<int32_t>& sa) { sa[0] = -1; }},
        };
        const struct {
            const char* name;
            CheckMode mode;
        } modes[] = {{"fast", CheckMode::Fast}, {"lowmem", CheckMode::LowMemory}, {"naive", CheckMode::Naive}};

        for (const auto& mode : modes) {
            suite.cases++;

            if (!checkSuffixArray(padded.data(), n, good.data(), mode.mode).ok) {
                suite.reportFailure(std::string("checker ") + mode.name, "rejects a correct suffix array", -1);
            }

            for (const Mutation& mutation : mutations) {
                std::vector<int32_t> bad(good);

                mutation.apply(bad);
                suite.cases++;

                if (checkSuffixArray(padded.data(), n, bad.data(), mode.mode).ok) {
                    suite.reportFailure(std::string("checker ") + mode.name, mutation.name, -1);
                }
            }
        }

        const Bwt goodBwt = bwtFromSuffixArray(padded.data(), n, good);

        suite.cases++;

        if (!checkBwt(padded.data(), n, goodBwt.bytes.data(), goodBwt.primary).ok) {
            suite.reportFailure("BWT checker", "rejects a correct BWT", -1);
        }

        struct BwtMutation {
            const char* name;
            void (*apply)(Bwt&);
        };

        const BwtMutation bwtMutations[] = {
            {"swapped neighbours", [](Bwt& bwt) {
                 auto it = std::adjacent_find(bwt.bytes.begin() + 1000, bwt.bytes.end(), std::not_equal_to<>());
                 std::iter_swap(it, it + 1);
             }},
            {"changed character", [](Bwt& bwt) { bwt.bytes[1234] ^= 1; }},
            {"primary index off by one", [](Bwt& bwt) { bwt.primary += bwt.primary < 3000 ? 1 : -1; }},
            {"primary index 0", [](Bwt& bwt) { bwt.primary = 0; }},
        };

        for (const BwtMutation& mutation : bwtMutations) {
            Bwt bad(goodBwt);

            mutation.apply(bad);
            suite.cases++;

            if (checkBwt(padded.data(), n, bad.bytes.data(), bad.primary).ok) {
                suite.reportFailure("BWT checker", mutation.name, -1);
            }
        }
    }

    void runSmallCases(Suite& suite) {
        const char* words[] = {"", "a", "ab", "ba", "aa", "aab", "banana", "mississippi", "abracadabra", "yabbadabbado"};

        for (const char* word : words) {
            testSmall(suite, std::string("\"") + word + "\"", textOf(word));
        }

        // Lengths around all cover periods and their tie limits.
        for (int32_t n = 1; n <= 270; n++) {
            testSmall(suite, "run of a, n=" + std::to_string(n), std::vector<uint8_t>(n, 'a'));
        }

        // Zero bytes look like the padding but must sort as real characters.
        testSmall(suite, "zeros", std::vector<uint8_t>(1000, 0));
        testSmall(suite, "run then b", [] {
            std::vector<uint8_t> text(2000, 'a');
            text.back() = 'b';
            return text;
        }());
        testSmall(suite, "b then run", [] {
            std::vector<uint8_t> text(2000, 'a');
            text.front() = 'b';
            return text;
        }());

        Rng rng{1};

        for (const int32_t period : {1, 2, 3, 5, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000}) {
            testSmall(suite, "period " + std::to_string(period), periodicText(rng, 5000, period));
        }

        testSmall(suite, "fibonacci", fibonacciText(5000, 'a', 'b'));
        testSmall(suite, "fibonacci 0/1", fibonacciText(5000, 0, 1));
        testSmall(suite, "thue-morse", thueMorseText(4096));

        for (const uint32_t alphabetSize : {1u, 2u, 3u, 4u, 26u, 256u}) {
            for (int i = 0; i < 25; i++) {
                const int32_t n = 1 + static_cast<int32_t>(rng.nextBelow(4000));
                testSmall(suite, "random sigma=" + std::to_string(alphabetSize) + " #" + std::to_string(i),
                    randomText(rng, n, alphabetSize));
            }
        }
    }

    // The GPU construction must return the CPU's suffix array, with a temporary context or the given one.
    void testGpu(Suite& suite, const std::string& name, const std::vector<uint8_t>& input, GpuContext* context = nullptr) {
        const std::vector<uint8_t> padded = withPadding(input);
        const int32_t n = static_cast<int32_t>(input.size());
        std::vector<int32_t> expected(n);
        std::vector<int32_t> actual(n, -1);

        suite.cases++;
        buildSuffixArray(padded.data(), n, expected.data(), kCovers[0]);

        if (gpuSuffixArray(context, padded.data(), n, actual.data()) != GpuStatus::Ok) {
            suite.reportFailure(name + " GPU", gpuLastError(), -1);
            return;
        }

        if (const int64_t mismatch = firstMismatch(expected, actual); mismatch >= 0) {
            suite.reportFailure(name + " GPU", "differs from the CPU suffix array", mismatch);
            return;
        }

        // The BWT, in place of a copy of the text as sadcs build --bwt --gpu does it.
        std::vector<uint8_t> text(padded);
        Bwt bwt;

        suite.cases++;

        if (gpuBwt(context, text.data(), n, text.data(), bwt.primary) != GpuStatus::Ok) {
            suite.reportFailure(name + " GPU BWT", gpuLastError(), -1);
            return;
        }

        bwt.bytes.assign(text.begin(), text.begin() + n);

        if (!compareBwt(suite, name + " GPU BWT", bwtFromSuffixArray(padded.data(), n, expected), bwt)) {
            return;
        }

        if (suite.verbose) {
            printf("  ok   %s GPU, suffix array and BWT (n=%d)\n", name.c_str(), n);
        }
    }

    // The GPU sample ranking must match the ranks the CPU's suffix array implies.
    void testGpuSampleRanks(Suite& suite, const std::string& name, const std::vector<uint8_t>& input) {
        const std::vector<uint8_t> padded = withPadding(input);
        const int32_t n = static_cast<int32_t>(input.size());
        std::vector<int32_t> sa(n);
        std::vector<uint32_t> expected(gpuSampleSize(n));
        std::vector<uint32_t> actual(expected.size());
        GpuStats stats;

        suite.cases++;
        buildSuffixArray(padded.data(), n, sa.data(), kCovers[0]);
        gpuSampleRanksFromSuffixArray(sa.data(), n, expected.data());

        if (gpuRankSample(padded.data(), n, actual.data(), &stats) != GpuStatus::Ok) {
            suite.reportFailure(name + " GPU sample", gpuLastError(), -1);
            return;
        }

        if (const int64_t mismatch = firstMismatch(expected, actual); mismatch >= 0) {
            suite.reportFailure(name + " GPU sample", "ranks differ from the CPU suffix array's", mismatch);
            return;
        }

        if (suite.verbose) {
            printf("  ok   %s GPU sample (n=%d, %d character passes, %d doubling rounds)\n", name.c_str(), n,
                stats.characterPasses, stats.doublingRounds);
        }
    }

    // One context reused for inputs of different sizes, largest first,
    // so each construction runs in memory an earlier one left dirty.
    // The text of the largest takes several staging chunks.
    void runGpuContextCases(Suite& suite, Rng& rng) {
        constexpr int32_t kCapacity = 20000000;
        GpuContext* context = nullptr;

        suite.cases++;

        if (gpuCreateContext(kCapacity, context) != GpuStatus::Ok) {
            suite.reportFailure("GPU context", gpuLastError(), -1);
            return;
        }

        testGpu(suite, "context, random sigma=4, n=20M", randomText(rng, kCapacity, 4), context);
        testGpu(suite, "context, large random sigma=256", randomText(rng, kLargeTextSize, 256), context);
        testGpu(suite, "context, \"banana\"", textOf("banana"), context);
        testGpu(suite, "context, \"\"", {}, context);
        testGpu(suite, "context, \"a\"", textOf("a"), context);
        testGpu(suite, "context, large fibonacci", fibonacciText(kLargeTextSize, 'a', 'b'), context);
        testGpu(suite, "context, period 33", periodicText(rng, 5000, 33), context);

        for (int i = 0; i < 5; i++) {
            const int32_t n = 1 + static_cast<int32_t>(rng.nextBelow(20000));
            testGpu(suite, "context, random sigma=2 #" + std::to_string(i), randomText(rng, n, 2), context);
        }

        // Larger than the capacity: refused, and the context still works afterwards.
        const std::vector<uint8_t> tooLarge = withPadding(std::vector<uint8_t>(kCapacity + 1, 'a'));
        std::vector<int32_t> sa(kCapacity + 1);

        suite.cases++;

        if (gpuSuffixArray(context, tooLarge.data(), kCapacity + 1, sa.data()) != GpuStatus::NotEnoughMemory) {
            suite.reportFailure("GPU context, over capacity", "not refused with NotEnoughMemory", -1);
        }

        testGpu(suite, "context, after refusal", randomText(rng, 5000, 26), context);
        gpuDestroyContext(context);
    }

    void runGpuCases(Suite& suite) {
        GpuDevice device;

        if (gpuQueryDevice(device) != GpuStatus::Ok) {
            printf("selftest: GPU cases skipped (%s)\n", gpuLastError());
            return;
        }

        printf("selftest: GPU vs. CPU on %s (sm_%d%d, %lld MiB free)...\n", device.name, device.computeMajor,
            device.computeMinor, static_cast<long long>(device.freeBytes >> 20));
        Rng rng{3};

        testGpu(suite, "\"\"", {});
        testGpu(suite, "\"a\"", textOf("a"));
        testGpu(suite, "\"banana\"", textOf("banana"));
        testGpu(suite, "\"mississippi\"", textOf("mississippi"));
        testGpu(suite, "run of a", std::vector<uint8_t>(1000, 'a'));
        testGpu(suite, "fibonacci", fibonacciText(5000, 'a', 'b'));
        testGpu(suite, "fibonacci 0/1", fibonacciText(5000, 0, 1));
        testGpu(suite, "thue-morse", thueMorseText(4096));

        for (const int32_t period : {1, 2, 3, 5, 7, 31, 32, 33, 63, 64, 65, 1000}) {
            testGpu(suite, "period " + std::to_string(period), periodicText(rng, 5000, period));
        }

        for (int32_t n = 0; n <= 80; n++) {
            testGpu(suite, "run of a, n=" + std::to_string(n), std::vector<uint8_t>(n, 'a'));
            testGpu(suite, "random sigma=2, n=" + std::to_string(n), randomText(rng, n, 2));
            testGpuSampleRanks(suite, "run of a, n=" + std::to_string(n), std::vector<uint8_t>(n, 'a'));
            testGpuSampleRanks(suite, "random sigma=2, n=" + std::to_string(n), randomText(rng, n, 2));
        }

        for (const uint32_t alphabetSize : {1u, 2u, 4u, 26u, 256u}) {
            for (int i = 0; i < 10; i++) {
                const int32_t n = 1 + static_cast<int32_t>(rng.nextBelow(20000));
                testGpu(suite, "random sigma=" + std::to_string(alphabetSize) + " #" + std::to_string(i),
                    randomText(rng, n, alphabetSize));
            }
        }

        testGpu(suite, "large run of a", std::vector<uint8_t>(kLargeTextSize, 'a'));
        testGpu(suite, "large period 33", periodicText(rng, kLargeTextSize, 33));
        testGpu(suite, "large fibonacci", fibonacciText(kLargeTextSize, 'a', 'b'));
        testGpu(suite, "large thue-morse", thueMorseText(kLargeTextSize));
        testGpu(suite, "large random sigma=4", randomText(rng, kLargeTextSize, 4));
        testGpu(suite, "large random sigma=256", randomText(rng, kLargeTextSize, 256));

        runGpuContextCases(suite, rng);

        for (const int32_t period : {1, 2, 3, 7, 31, 32, 33, 64, 65, 1000}) {
            testGpuSampleRanks(suite, "period " + std::to_string(period), periodicText(rng, 5000, period));
        }

        testGpuSampleRanks(suite, "\"banana\"", textOf("banana"));
        testGpuSampleRanks(suite, "fibonacci 0/1", fibonacciText(5000, 0, 1));
        testGpuSampleRanks(suite, "thue-morse", thueMorseText(4096));
        testGpuSampleRanks(suite, "random sigma=256", randomText(rng, 5000, 256));
        testGpuSampleRanks(suite, "large run of a", std::vector<uint8_t>(kLargeTextSize, 'a'));
        testGpuSampleRanks(suite, "large period 33", periodicText(rng, kLargeTextSize, 33));
        testGpuSampleRanks(suite, "large fibonacci", fibonacciText(kLargeTextSize, 'a', 'b'));
        testGpuSampleRanks(suite, "large thue-morse", thueMorseText(kLargeTextSize));
        testGpuSampleRanks(suite, "large random sigma=4", randomText(rng, kLargeTextSize, 4));
        testGpuSampleRanks(suite, "large random sigma=256", randomText(rng, kLargeTextSize, 256));
    }

    void runLargeCases(Suite& suite) {
        Rng rng{2};

        testLarge(suite, "large run of a", std::vector<uint8_t>(kLargeTextSize, 'a'));
        testLarge(suite, "large period 33", periodicText(rng, kLargeTextSize, 33));
        testLarge(suite, "large period 64", periodicText(rng, kLargeTextSize, 64));
        testLarge(suite, "large period 256", periodicText(rng, kLargeTextSize, 256));
        testLarge(suite, "large period 5000", periodicText(rng, kLargeTextSize, 5000));
        testLarge(suite, "large fibonacci", fibonacciText(kLargeTextSize, 'a', 'b'));
        testLarge(suite, "large thue-morse", thueMorseText(kLargeTextSize));
        testLarge(suite, "large random sigma=4", randomText(rng, kLargeTextSize, 4));
        testLarge(suite, "large random sigma=256", randomText(rng, kLargeTextSize, 256));
    }
}

int runSelfTest(bool verbose) {
    Suite suite{verbose};

    printf("selftest: small inputs vs. naive suffix array...\n");
    runSmallCases(suite);
    printf("selftest: large inputs (%d threads)...\n", omp_get_max_threads());
    runLargeCases(suite);
    printf("selftest: checker catches planted errors...\n");
    testCheckerCatches(suite);
    runGpuCases(suite);

    printf("selftest: %d of %d cases passed\n", suite.cases - suite.failures, suite.cases);

    return suite.failures;
}
