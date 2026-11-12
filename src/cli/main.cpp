/*
 * main.cpp - sadcs CLI: build/check/test/selftest commands.
 * Suffix array files are raw little-endian int32 (divsufsort's mksary format).
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include <omp.h>
#include "checker.h"
#include "dcsort.h"
#include "fileio.h"
#include "gpusa.h"
#include "selftest.h"
#include "simd.h"

namespace {
    constexpr int kExitOk = 0;
    constexpr int kExitError = 1;
    constexpr int kMaxPaths = 2;
    constexpr int kPreviewBytes = 24;

    struct Options {
        const char* paths[kMaxPaths] = {};
        int pathCount = 0;
        int cover = 128;
        CheckMode checkMode = CheckMode::Fast;
        bool bwt = false;
        bool gpu = false;
        bool verbose = false;
        bool recursive = false;
        int runs = 5;
        bool context = false;
        bool pinned = false;
    };

    // One build + check of one file, or the sum over several.
    struct TestRun {
        int64_t bytes = 0;
        BuildStats stats;
        double buildSeconds = 0.0;
        double gpuSeconds = 0.0;  // GPU construction, including transfers
        GpuStats gpu;
        double checkSeconds = 0.0;
        double peakPerByte = 0.0;  // GPU peak device memory per input byte; the total row shows the largest
        bool ok = true;

        void add(const TestRun& other) {
            bytes += other.bytes;
            stats.sampleStringSortSeconds += other.stats.sampleStringSortSeconds;
            stats.sampleDoublingSeconds += other.stats.sampleDoublingSeconds;
            stats.mainSortSeconds += other.stats.mainSortSeconds;
            buildSeconds += other.buildSeconds;
            gpuSeconds += other.gpuSeconds;
            gpu.deviceBytes = std::max(gpu.deviceBytes, other.gpu.deviceBytes);
            peakPerByte = std::max(peakPerByte, other.peakPerByte);
            checkSeconds += other.checkSeconds;
        }
    };

    void printUsage() {
        printf(
            "sadcs - difference-cover suffix array construction\n"
            "\n"
            "usage:\n"
            "  sadcs build|-b <in> <out.sa> [--cover 128|64|256] [--threads N] [--gpu]\n"
            "  sadcs build|-b --bwt <in> <out.bwt> [--cover 128|64|256] [--threads N] [--gpu]\n"
            "  sadcs check|-c <in> <in.sa> [--lowmem | --naive] [--threads N]\n"
            "  sadcs check|-c --bwt <in> <in.bwt>\n"
            "  sadcs test|-t <in|dir> [--bwt] [--gpu] [--cover 128|64|256] [--threads N] [--recursive]\n"
            "  sadcs bench <in|dir> [--bwt] [--runs N] [--context] [--pinned] [--recursive]\n"
            "  sadcs selftest [--verbose] [--threads N]\n"
            "\n"
            "The .sa file is n little-endian int32 entries (divsufsort's mksary format).\n"
            "The .bwt file is divsufsort's bwt format as one block: n, the primary index, the n bytes.\n"
            "--bwt builds the Burrows-Wheeler transform instead and checks it by inverting it.\n"
            "test on a directory runs every file in it (--recursive: and below) and prints a timing table.\n"
            "--gpu builds the suffix array or BWT on the GPU instead (cover 32, --cover and --threads don't\n"
            "apply). test --gpu builds it on both, compares the two and checks the GPU's.\n"
            "bench times the GPU build: a warmup, then the best of --runs (default 5) host-to-host runs,\n"
            "with each phase of the best one. It doesn't check the output; test --gpu does. --context\n"
            "reuses one GPU context per file, created before the runs and not timed, as a caller building\n"
            "many inputs would. --pinned uses page-locked host buffers, allocated before the runs\n"
            "(--context --pinned is how libcubwt's benchmark measures).\n");
    }

    bool parseOptions(int argc, char** argv, Options& options) {
        for (int i = 2; i < argc; i++) {
            const char* arg = argv[i];

            if (strcmp(arg, "--cover") == 0 && i + 1 < argc) {
                options.cover = atoi(argv[++i]);

                if (options.cover != 64 && options.cover != 128 && options.cover != 256) {
                    printf("sadcs: --cover must be 64, 128 or 256\n");
                    return false;
                }
            } else if (strcmp(arg, "--threads") == 0 && i + 1 < argc) {
                const int threads = atoi(argv[++i]);

                if (threads < 1) {
                    printf("sadcs: --threads must be at least 1\n");
                    return false;
                }

                omp_set_num_threads(threads);
            } else if (strcmp(arg, "--lowmem") == 0) {
                options.checkMode = CheckMode::LowMemory;
            } else if (strcmp(arg, "--naive") == 0) {
                options.checkMode = CheckMode::Naive;
            } else if (strcmp(arg, "--bwt") == 0) {
                options.bwt = true;
            } else if (strcmp(arg, "--gpu") == 0) {
                options.gpu = true;
            } else if (strcmp(arg, "--runs") == 0 && i + 1 < argc) {
                options.runs = atoi(argv[++i]);

                if (options.runs < 1) {
                    printf("sadcs: --runs must be at least 1\n");
                    return false;
                }
            } else if (strcmp(arg, "--context") == 0) {
                options.context = true;
            } else if (strcmp(arg, "--pinned") == 0) {
                options.pinned = true;
            } else if (strcmp(arg, "--recursive") == 0) {
                options.recursive = true;
            } else if (strcmp(arg, "--verbose") == 0) {
                options.verbose = true;
            } else if (strncmp(arg, "--", 2) == 0) {
                printf("sadcs: unknown option '%s'\n", arg);
                return false;
            } else if (options.pathCount < kMaxPaths) {
                options.paths[options.pathCount++] = arg;
            } else {
                printf("sadcs: unexpected argument '%s'\n", arg);
                return false;
            }
        }

        return true;
    }

    bool loadText(const char* path, std::vector<uint8_t>& text, int32_t& n) {
        if (!readWholeFile(path, text, kTextPadding)) {
            printf("sadcs: cannot read '%s'\n", path);
            return false;
        }

        const int64_t size = static_cast<int64_t>(text.size()) - kTextPadding;

        if (size > kMaxTextSize) {
            printf("sadcs: '%s' is too large (%lld bytes, at most %lld supported)\n", path, static_cast<long long>(size),
                static_cast<long long>(kMaxTextSize));
            return false;
        }

        n = static_cast<int32_t>(size);

        return true;
    }

    double megabytesPerSecond(int64_t bytes, double seconds) {
        return seconds > 0.0 ? static_cast<double>(bytes) / (1024.0 * 1024.0) / seconds : 0.0;
    }

    void printSuffixPreview(const uint8_t* text, int32_t n, int32_t position) {
        if (position < 0 || position >= n) {
            printf("(out of range)");
            return;
        }

        printf("\"");

        for (int32_t i = position; i < n && i < position + kPreviewBytes; i++) {
            const uint8_t c = text[i];

            if (c >= 0x20 && c < 0x7F && c != '"' && c != '\\') {
                printf("%c", c);
            } else {
                printf("\\x%02X", c);
            }
        }

        printf(n - position > kPreviewBytes ? "\"..." : "\"");
    }

    void printCheckFailure(const uint8_t* text, int32_t n, const int32_t* sa, const CheckResult& result) {
        printf("sadcs: check FAILED at SA[%lld]: %s\n", static_cast<long long>(result.index), result.reason);

        for (int64_t i = result.index; i >= 0 && i < n && i <= result.index + 1; i++) {
            printf("  SA[%lld] = %d: ", static_cast<long long>(i), sa[i]);
            printSuffixPreview(text, n, sa[i]);
            printf("\n");
        }
    }

    bool reportGpu(GpuStatus status) {
        if (status != GpuStatus::Ok) {
            printf("sadcs: GPU %s: %s\n",
                status == GpuStatus::Unavailable       ? "unavailable"
                : status == GpuStatus::NotEnoughMemory ? "out of memory"
                                                       : "failed",
                gpuLastError());
            return false;
        }

        return true;
    }

    bool runGpu(const uint8_t* text, int32_t n, int32_t* sa, double& seconds, GpuStats& stats) {
        const GpuStatus status = gpuSuffixArray(text, n, sa, &stats);
        seconds = stats.uploadSeconds + stats.deviceSeconds + stats.downloadSeconds;
        return reportGpu(status);
    }

    // The BWT on the GPU; bwt may be text.
    bool runGpuBwt(const uint8_t* text, int32_t n, uint8_t* bwt, int32_t& primary, double& seconds, GpuStats& stats) {
        const GpuStatus status = gpuBwt(text, n, bwt, primary, &stats);
        seconds = stats.uploadSeconds + stats.deviceSeconds + stats.downloadSeconds;
        return reportGpu(status);
    }

    bool runCheck(const uint8_t* text, int32_t n, const int32_t* sa, CheckMode mode, double& seconds) {
        const double start = omp_get_wtime();
        const CheckResult result = checkSuffixArray(text, n, sa, mode);
        seconds = omp_get_wtime() - start;

        if (!result.ok) {
            printCheckFailure(text, n, sa, result);
        }

        return result.ok;
    }

    bool runBwtCheck(const uint8_t* text, int32_t n, const uint8_t* bwt, int32_t primary, double& seconds) {
        const double start = omp_get_wtime();
        const CheckResult result = checkBwt(text, n, bwt, primary);
        seconds = omp_get_wtime() - start;

        if (!result.ok) {
            printf("sadcs: BWT check FAILED (primary index %d): %s, index %lld\n", primary, result.reason,
                static_cast<long long>(result.index));
        }

        return result.ok;
    }

    TestRun runTest(const std::vector<uint8_t>& text, int32_t n, const Options& options) {
        TestRun run;

        run.bytes = n;

        if (options.bwt) {
            std::vector<uint8_t> bwt(n);
            int32_t primary;

            {
                std::vector<int32_t> sa(n);
                const double start = omp_get_wtime();
                primary = buildBwt(text.data(), n, bwt.data(), sa.data(), options.cover, &run.stats);
                run.buildSeconds = omp_get_wtime() - start;
            }

            if (options.gpu) {
                std::vector<uint8_t> gpuBwtBytes(n);
                int32_t gpuPrimary = -1;

                if (!runGpuBwt(text.data(), n, gpuBwtBytes.data(), gpuPrimary, run.gpuSeconds, run.gpu)) {
                    run.ok = false;
                    return run;
                }

                run.peakPerByte = n > 0 ? static_cast<double>(run.gpu.deviceBytes) / n : 0.0;
                const auto mismatch = std::mismatch(bwt.begin(), bwt.end(), gpuBwtBytes.begin());

                if (gpuPrimary != primary) {
                    printf("sadcs: GPU BWT primary index %d, the CPU's %d\n", gpuPrimary, primary);
                } else if (mismatch.first != bwt.end()) {
                    printf("sadcs: GPU BWT differs from the CPU's at byte %lld\n",
                        static_cast<long long>(mismatch.first - bwt.begin()));
                }

                bwt.swap(gpuBwtBytes);  // check the GPU's
                primary = gpuPrimary;
            }

            run.ok = runBwtCheck(text.data(), n, bwt.data(), primary, run.checkSeconds);

            return run;
        }

        std::vector<int32_t> sa(n);
        const double start = omp_get_wtime();
        buildSuffixArray(text.data(), n, sa.data(), options.cover, &run.stats);
        run.buildSeconds = omp_get_wtime() - start;

        if (options.gpu) {
            std::vector<int32_t> gpuSa(n);

            if (!runGpu(text.data(), n, gpuSa.data(), run.gpuSeconds, run.gpu)) {
                run.ok = false;
                return run;
            }

            run.peakPerByte = n > 0 ? static_cast<double>(run.gpu.deviceBytes) / n : 0.0;
            const auto mismatch = std::mismatch(sa.begin(), sa.end(), gpuSa.begin());

            if (mismatch.first != sa.end()) {
                const int64_t index = mismatch.first - sa.begin();

                printf("sadcs: GPU suffix array differs from the CPU's at SA[%lld]: %d, expected %d\n",
                    static_cast<long long>(index), gpuSa[index], sa[index]);
            }

            sa.swap(gpuSa);  // check the GPU's
        }

        run.ok = runCheck(text.data(), n, sa.data(), options.checkMode, run.checkSeconds);

        return run;
    }

    int cmdBuild(const Options& options) {
        if (options.pathCount != 2) {
            printUsage();
            return kExitError;
        }

        const char* inPath = options.paths[0];
        const char* outPath = options.paths[1];
        std::vector<uint8_t> text;
        int32_t n;

        if (!loadText(inPath, text, n)) {
            return kExitError;
        }

        // The GPU's BWT needs no suffix array on the host: n bytes in all, in place of the text.
        std::vector<int32_t> sa(options.gpu && options.bwt ? 0 : n);
        const double start = omp_get_wtime();
        int32_t primary = 0;
        double gpuSeconds = 0.0;
        GpuStats gpuStats;

        if (options.gpu && options.bwt) {
            if (!runGpuBwt(text.data(), n, text.data(), primary, gpuSeconds, gpuStats)) {
                return kExitError;
            }
        } else if (options.gpu) {
            if (!runGpu(text.data(), n, sa.data(), gpuSeconds, gpuStats)) {
                return kExitError;
            }
        } else if (options.bwt) {
            // In place of the text: 5n bytes in all.
            primary = buildBwt(text.data(), n, text.data(), sa.data(), options.cover);
        } else {
            buildSuffixArray(text.data(), n, sa.data(), options.cover);
        }

        const double seconds = omp_get_wtime() - start;
        const bool written = options.bwt ? writeBwtFile(outPath, text.data(), n, primary)
                                         : writeWholeFile(outPath, sa.data(), sa.size() * sizeof(int32_t));

        if (!written) {
            printf("sadcs: cannot write '%s'\n", outPath);
            return kExitError;
        }

        if (options.gpu) {
            printf("%s -> %s: %d bytes, GPU, %.3fs (%.1f MB/s), peak device memory %.1f MiB (%.2f bytes per input byte)\n",
                inPath, outPath, n, seconds, megabytesPerSecond(n, seconds), gpuStats.deviceBytes / (1024.0 * 1024.0),
                n > 0 ? static_cast<double>(gpuStats.deviceBytes) / n : 0.0);
        } else {
            printf("%s -> %s: %d bytes, cover %d, %.3fs (%.1f MB/s, %d threads)\n", inPath, outPath, n, options.cover,
                seconds, megabytesPerSecond(n, seconds), omp_get_max_threads());
        }

        return kExitOk;
    }

    int cmdCheck(const Options& options) {
        if (options.pathCount != 2) {
            printUsage();
            return kExitError;
        }

        const char* textPath = options.paths[0];
        const char* saPath = options.paths[1];
        std::vector<uint8_t> text;
        int32_t n;

        if (!loadText(textPath, text, n)) {
            return kExitError;
        }

        if (options.bwt) {
            std::vector<uint8_t> bwt;
            int32_t primary;

            if (!readBwtFile(saPath, bwt, primary)) {
                printf("sadcs: cannot read '%s' as a single-block BWT file\n", saPath);
                return kExitError;
            }

            if (static_cast<int64_t>(bwt.size()) != n) {
                printf("sadcs: '%s' has %zu bytes of BWT, but '%s' has %d bytes\n", saPath, bwt.size(), textPath, n);
                return kExitError;
            }

            double seconds;

            if (!runBwtCheck(text.data(), n, bwt.data(), primary, seconds)) {
                return kExitError;
            }

            printf("%s: BWT OK (%d bytes, primary index %d, %.3fs)\n", saPath, n, primary, seconds);

            return kExitOk;
        }

        std::vector<int32_t> sa;

        if (!readSuffixArray(saPath, sa)) {
            printf("sadcs: cannot read '%s' as a suffix array\n", saPath);
            return kExitError;
        }

        if (static_cast<int64_t>(sa.size()) != n) {
            printf("sadcs: '%s' has %zu entries, but '%s' has %d bytes\n", saPath, sa.size(), textPath, n);
            return kExitError;
        }

        double seconds;

        if (!runCheck(text.data(), n, sa.data(), options.checkMode, seconds)) {
            return kExitError;
        }

        printf("%s: suffix array OK (%d entries, %.3fs)\n", saPath, n, seconds);

        return kExitOk;
    }

    int testFile(const Options& options) {
        const char* path = options.paths[0];
        std::vector<uint8_t> text;
        int32_t n;

        if (!loadText(path, text, n)) {
            return kExitError;
        }

        printf("%s: %d bytes, %s, cover %d, %d threads\n", path, n, options.bwt ? "BWT" : "suffix array",
            options.cover, omp_get_max_threads());

        const TestRun run = runTest(text, n, options);

        printf("  sample          %d suffixes (%.1f%%)\n", run.stats.sampleSize,
            n > 0 ? 100.0 * run.stats.sampleSize / n : 0.0);
        printf("  sample sort     %.3fs\n", run.stats.sampleStringSortSeconds);
        printf("  doubling        %.3fs\n", run.stats.sampleDoublingSeconds);
        printf("  main sort       %.3fs\n", run.stats.mainSortSeconds);
        printf("  total           %.3fs (%.1f MB/s)\n", run.buildSeconds, megabytesPerSecond(n, run.buildSeconds));

        if (options.gpu && run.gpu.deviceBytes > 0) {
            const auto perByte = [n](int64_t bytes) { return n > 0 ? static_cast<double>(bytes) / n : 0.0; };

            printf("  GPU             %.3fs (upload %.3fs, device %.3fs, download %.3fs, %s host memory; %d character "
                   "passes, %d doubling rounds)\n",
                run.gpuSeconds, run.gpu.uploadSeconds, run.gpu.deviceSeconds, run.gpu.downloadSeconds,
                run.gpu.pinned ? "pinned" : "pageable", run.gpu.characterPasses, run.gpu.doublingRounds);
            printf("  GPU memory      peak %.1f MiB, %.2f bytes per input byte (sample ranking %.2f, class sorts %.2f, "
                   "merge %.2f)\n",
                run.gpu.deviceBytes / (1024.0 * 1024.0), perByte(run.gpu.deviceBytes), perByte(run.gpu.rankingBytes),
                perByte(run.gpu.classSortBytes), perByte(run.gpu.mergeBytes));
        }

        if (!run.ok) {
            return kExitError;
        }

        printf("  check           %.3fs, OK\n", run.checkSeconds);

        return kExitOk;
    }

    // Regular files under dir, sorted.
    // Uses the error_code overloads: sadcs builds without exceptions.
    bool listFiles(const std::filesystem::path& dir, bool recursive, std::vector<std::filesystem::path>& files) {
        std::error_code error;
        auto collect = [&](auto iterator) {
            for (const decltype(iterator) end{}; !error && iterator != end; iterator.increment(error)) {
                std::error_code ignored;

                if (iterator->is_regular_file(ignored)) {
                    files.push_back(iterator->path());
                }
            }
        };

        if (recursive) {
            collect(std::filesystem::recursive_directory_iterator(dir, error));
        } else {
            collect(std::filesystem::directory_iterator(dir, error));
        }

        std::sort(files.begin(), files.end());

        return !error;
    }

    void printTableRow(const char* name, int nameWidth, const TestRun& run, bool gpu, const char* verdict) {
        printf("%-*s %12lld %8.3f %8.3f %8.3f %8.3f %8.1f", nameWidth, name, static_cast<long long>(run.bytes),
            run.stats.sampleStringSortSeconds, run.stats.sampleDoublingSeconds, run.stats.mainSortSeconds,
            run.buildSeconds, megabytesPerSecond(run.bytes, run.buildSeconds));

        if (gpu) {
            printf(" %8.3f %6.2f", run.gpuSeconds, run.peakPerByte);
        }

        printf(" %8.3f  %s\n", run.checkSeconds, verdict);
    }

    int testDirectory(const Options& options) {
        const std::filesystem::path dir(options.paths[0]);
        std::vector<std::filesystem::path> files;

        if (!listFiles(dir, options.recursive, files)) {
            printf("sadcs: cannot list directory '%s'\n", options.paths[0]);
            return kExitError;
        }

        std::vector<std::string> names;
        int nameWidth = 5;

        for (const auto& file : files) {
            names.push_back(file.lexically_relative(dir).generic_string());
            nameWidth = std::max(nameWidth, static_cast<int>(names.back().size()));
        }

        printf("%s: %zu files, %s, cover %d, %d threads (times in seconds)\n", options.paths[0], files.size(),
            options.bwt ? "BWT" : "suffix array", options.cover, omp_get_max_threads());
        printf("%-*s %12s %8s %8s %8s %8s %8s", nameWidth, "file", "bytes", "sample", "doubling", "main", "total", "MB/s");

        if (options.gpu) {
            printf(" %8s %6s", "gpu", "peak/n");
        }

        printf(" %8s\n", "check");

        TestRun total;
        size_t passed = 0;

        for (size_t i = 0; i < files.size(); i++) {
            std::vector<uint8_t> text;
            int32_t n;

            if (!loadText(files[i].string().c_str(), text, n)) {
                continue;
            }

            const TestRun run = runTest(text, n, options);

            printTableRow(names[i].c_str(), nameWidth, run, options.gpu, run.ok ? "OK" : "FAILED");
            total.add(run);
            passed += run.ok ? 1 : 0;
        }

        const std::string verdict = std::to_string(passed) + "/" + std::to_string(files.size()) + " OK";
        printTableRow("total", nameWidth, total, options.gpu, verdict.c_str());

        return passed == files.size() ? kExitOk : kExitError;
    }

    // The best of options.runs GPU builds of one file, after a warmup.
    // seconds is host to host, including the setup (pinning, allocations); stats are the best run's.
    struct BenchRun {
        int64_t bytes = 0;
        double seconds = 0.0;
        GpuStats stats;
        double peakPerByte = 0.0;  // the largest over the files

        void add(const BenchRun& other) {
            bytes += other.bytes;
            seconds += other.seconds;
            stats.setupSeconds += other.stats.setupSeconds;
            stats.uploadSeconds += other.stats.uploadSeconds;
            stats.rankingSeconds += other.stats.rankingSeconds;
            stats.classSortSeconds += other.stats.classSortSeconds;
            stats.mergeSeconds += other.stats.mergeSeconds;
            stats.bwtSeconds += other.stats.bwtSeconds;
            stats.downloadSeconds += other.stats.downloadSeconds;
            peakPerByte = std::max(peakPerByte, other.peakPerByte);
        }
    };

    bool benchGpu(const uint8_t* text, int32_t n, const Options& options, BenchRun& best) {
        std::vector<int32_t> sa(options.bwt || options.pinned ? 0 : n);
        std::vector<uint8_t> bwt(options.bwt && !options.pinned ? n : 0);
        GpuContext* context = nullptr;

        best = BenchRun{};
        best.bytes = n;

        if (options.context && !reportGpu(gpuCreateContext(n, context))) {
            return false;
        }

        const std::unique_ptr<GpuContext, void (*)(GpuContext*)> owner(context, gpuDestroyContext);

        // --pinned: page-locked copies of the text and the output, allocated before the runs.
        using Pinned = std::unique_ptr<void, void (*)(void*)>;
        const size_t outputBytes = options.bwt ? static_cast<size_t>(n) : static_cast<size_t>(n) * sizeof(int32_t);
        const Pinned pinnedText(options.pinned ? gpuAllocatePinned(static_cast<size_t>(n)) : nullptr, gpuFreePinned);
        const Pinned pinnedOutput(options.pinned ? gpuAllocatePinned(outputBytes) : nullptr, gpuFreePinned);
        const uint8_t* input = text;
        int32_t* saOut = sa.data();
        uint8_t* bwtOut = bwt.data();

        if (options.pinned) {
            if (!pinnedText || !pinnedOutput) {
                printf("sadcs: cannot allocate pinned host memory\n");
                return false;
            }

            memcpy(pinnedText.get(), text, static_cast<size_t>(n));
            input = static_cast<const uint8_t*>(pinnedText.get());
            saOut = static_cast<int32_t*>(pinnedOutput.get());
            bwtOut = static_cast<uint8_t*>(pinnedOutput.get());
        }

        for (int run = 0; run <= options.runs; run++) {
            GpuStats stats;
            int32_t primary = 0;
            const double start = omp_get_wtime();
            const GpuStatus status = options.bwt ? gpuBwt(context, input, n, bwtOut, primary, &stats)
                                                 : gpuSuffixArray(context, input, n, saOut, &stats);
            const double seconds = omp_get_wtime() - start;

            if (!reportGpu(status)) {
                return false;
            }

            if (run > 0 && (run == 1 || seconds < best.seconds)) {  // run 0 is the warmup
                best.seconds = seconds;
                best.stats = stats;
                best.peakPerByte = n > 0 ? static_cast<double>(stats.deviceBytes) / n : 0.0;
            }
        }

        return true;
    }

    void printBenchHeader(int nameWidth) {
        printf("%-*s %12s %7s %7s %7s %7s %7s %7s %7s %8s %7s %6s\n", nameWidth, "file", "bytes", "setup", "upload",
            "rank", "class", "merge", "bwt", "down", "total", "MB/s", "peak/n");
    }

    void printBenchRow(const char* name, int nameWidth, const BenchRun& run) {
        const auto ms = [](double seconds) { return seconds * 1000.0; };
        const GpuStats& s = run.stats;

        printf("%-*s %12lld %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %8.1f %7.1f %6.2f\n", nameWidth, name,
            static_cast<long long>(run.bytes), ms(s.setupSeconds), ms(s.uploadSeconds), ms(s.rankingSeconds),
            ms(s.classSortSeconds), ms(s.mergeSeconds), ms(s.bwtSeconds), ms(s.downloadSeconds), ms(run.seconds),
            megabytesPerSecond(run.bytes, run.seconds), run.peakPerByte);
    }

    int cmdBench(const Options& options) {
        if (options.pathCount != 1) {
            printUsage();
            return kExitError;
        }

        const std::filesystem::path path(options.paths[0]);
        std::vector<std::filesystem::path> files;
        std::error_code error;
        const bool directory = std::filesystem::is_directory(path, error);

        if (directory && !listFiles(path, options.recursive, files)) {
            printf("sadcs: cannot list directory '%s'\n", options.paths[0]);
            return kExitError;
        }

        if (!directory) {
            files.push_back(path);
        }

        std::vector<std::string> names;
        int nameWidth = 5;

        for (const auto& file : files) {
            names.push_back(directory ? file.lexically_relative(path).generic_string() : file.filename().string());
            nameWidth = std::max(nameWidth, static_cast<int>(names.back().size()));
        }

        GpuDevice device;

        if (!reportGpu(gpuQueryDevice(device))) {
            return kExitError;
        }

        printf("%s: %zu file%s, GPU %s, %s, best of %d runs%s%s (times in ms)\n", options.paths[0], files.size(),
            files.size() == 1 ? "" : "s", device.name, options.bwt ? "BWT" : "suffix array", options.runs,
            options.context ? ", context reused" : "", options.pinned ? ", pinned host buffers" : "");
        printBenchHeader(nameWidth);

        BenchRun total;

        for (size_t i = 0; i < files.size(); i++) {
            std::vector<uint8_t> text;
            int32_t n;
            BenchRun run;

            if (!loadText(files[i].string().c_str(), text, n) || !benchGpu(text.data(), n, options, run)) {
                return kExitError;
            }

            printBenchRow(names[i].c_str(), nameWidth, run);
            total.add(run);
        }

        if (files.size() > 1) {
            printBenchRow("total", nameWidth, total);
        }

        return kExitOk;
    }

    int cmdTest(const Options& options) {
        if (options.pathCount != 1) {
            printUsage();
            return kExitError;
        }

        std::error_code error;

        if (std::filesystem::is_directory(options.paths[0], error)) {
            return testDirectory(options);
        }

        return testFile(options);
    }
}

int main(int argc, char** argv) {
    // The sample sort's tasks run ahead of the main sort's (kSampleSortPriority);
    // libomp ignores task priorities unless this is set before it initializes.
    if (!std::getenv("OMP_MAX_TASK_PRIORITY")) {
        _putenv_s("OMP_MAX_TASK_PRIORITY", "1");
    }

    if (argc < 2) {
        printUsage();
        return kExitError;
    }

    const char* command = argv[1];
    Options options;

    if (!parseOptions(argc, argv, options)) {
        return kExitError;
    }

    if (strcmp(command, "build") == 0 || strcmp(command, "-b") == 0) {
        return cmdBuild(options);
    }

    if (strcmp(command, "check") == 0 || strcmp(command, "-c") == 0) {
        return cmdCheck(options);
    }

    if (strcmp(command, "test") == 0 || strcmp(command, "-t") == 0) {
        return cmdTest(options);
    }

    if (strcmp(command, "bench") == 0) {
        return cmdBench(options);
    }

    if (strcmp(command, "selftest") == 0) {
        return runSelfTest(options.verbose) == 0 ? kExitOk : kExitError;
    }

    printUsage();

    return kExitError;
}
