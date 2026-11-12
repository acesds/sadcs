/*
 * bench_cubwt.cu - Times libcubwt end to end: host buffers in, host buffers out, transfers included.
 * Build against 1.6.3 for the BWT, against 1.0.0 with -DCUBWT_SA for the suffix array.
 * Usage: bench_cubwt [--runs N] [--out DIR] [--prefix BYTES] file...
 * With --out it writes <name>.sa (little-endian int32, divsufsort's mksary format)
 * or <name>.bwt (divsufsort's bwt format: n, the primary index, the bytes).
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "libcubwt.cuh"

namespace {
#ifdef CUBWT_SA
    constexpr const char* kWhat = "SA";
    constexpr size_t kOutputBytesPerByte = 4;
#else
    constexpr const char* kWhat = "BWT";
    constexpr size_t kOutputBytesPerByte = 1;
#endif

    // The first limit bytes of the file at path (all of it if limit is 0), or nothing on failure.
    std::vector<uint8_t> readFile(const char* path, int64_t limit) {
        std::vector<uint8_t> data;
        FILE* fp = fopen(path, "rb");

        if (!fp) {
            return data;
        }

        _fseeki64(fp, 0, SEEK_END);
        int64_t size = _ftelli64(fp);
        _fseeki64(fp, 0, SEEK_SET);

        if (limit > 0 && size > limit) {
            size = limit;
        }

        data.resize(static_cast<size_t>(size));

        if (size > 0 && fread(data.data(), 1, data.size(), fp) != data.size()) {
            data.clear();
        }

        fclose(fp);

        return data;
    }

    std::string baseName(const char* path) {
        const std::string name(path);
        const size_t slash = name.find_last_of("/\\");

        return slash == std::string::npos ? name : name.substr(slash + 1);
    }

    double secondsSince(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    void writeOutput(const char* outDir, const std::string& name, const void* output, int64_t n, int64_t primary) {
        const std::string path = std::string(outDir) + "/" + name + (kOutputBytesPerByte == 4 ? ".sa" : ".bwt");
        FILE* fp = fopen(path.c_str(), "wb");

        if (!fp) {
            fprintf(stderr, "cannot write %s\n", path.c_str());
            return;
        }

        if constexpr (kOutputBytesPerByte == 1) {
            const int32_t header[2] = {static_cast<int32_t>(n), static_cast<int32_t>(primary)};
            fwrite(header, sizeof(int32_t), 2, fp);
        }

        fwrite(output, kOutputBytesPerByte, static_cast<size_t>(n), fp);
        fclose(fp);
    }
}

int main(int argc, char** argv) {
    int runs = 3;
    int64_t prefix = 0;
    const char* outDir = nullptr;
    std::vector<const char*> files;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            outDir = argv[++i];
        } else if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
            prefix = _atoi64(argv[++i]);
        } else {
            files.push_back(argv[i]);
        }
    }

    if (files.empty()) {
        fprintf(stderr, "usage: bench_cubwt [--runs N] [--out DIR] [--prefix BYTES] file...\n");
        return 1;
    }

    std::vector<std::vector<uint8_t>> texts;
    int64_t maxN = 0;

    for (const char* file : files) {
        texts.push_back(readFile(file, prefix));

        if (texts.back().empty()) {
            fprintf(stderr, "cannot read %s\n", file);
            return 1;
        }

        maxN = std::max(maxN, static_cast<int64_t>(texts.back().size()));
    }

    // Creates the CUDA context outside the timings.
    cudaFree(nullptr);

    const auto allocStart = std::chrono::steady_clock::now();
    void* storage = nullptr;
    const int64_t error = libcubwt_allocate_device_storage(&storage, maxN);
    const double allocSeconds = secondsSince(allocStart);

    if (error != LIBCUBWT_NO_ERROR) {
        fprintf(stderr, "libcubwt_allocate_device_storage(%lld) failed: %lld\n", static_cast<long long>(maxN),
            static_cast<long long>(error));
        return 1;
    }

    size_t freeBytes = 0, totalBytes = 0;
    uint8_t* hostText = nullptr;
    void* hostOutput = nullptr;

    cudaMemGetInfo(&freeBytes, &totalBytes);

    if (cudaMallocHost(reinterpret_cast<void**>(&hostText), static_cast<size_t>(maxN)) != cudaSuccess ||
        cudaMallocHost(&hostOutput, static_cast<size_t>(maxN) * kOutputBytesPerByte) != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed\n");
        return 1;
    }

    printf("libcubwt %s, %s, best of %d (+1 warmup), storage for %lld bytes: alloc %.3fs, VRAM free after alloc "
           "%.0f MiB of %.0f MiB\n",
        LIBCUBWT_VERSION_STRING, kWhat, runs, static_cast<long long>(maxN), allocSeconds, freeBytes / 1048576.0,
        totalBytes / 1048576.0);
    printf("%-16s %12s %9s %9s\n", "file", "bytes", "seconds", "MB/s");

    int64_t sumBytes = 0;
    double sumSeconds = 0.0;

    for (size_t k = 0; k < files.size(); k++) {
        const int64_t n = static_cast<int64_t>(texts[k].size());
        double best = 1e30;
        int64_t result = 0;

        memcpy(hostText, texts[k].data(), static_cast<size_t>(n));

        for (int run = 0; run <= runs; run++) {
            const auto start = std::chrono::steady_clock::now();
#ifdef CUBWT_SA
            result = libcubwt_sa(storage, hostText, static_cast<uint32_t*>(hostOutput), n);
#else
            result = libcubwt_bwt(storage, hostText, static_cast<uint8_t*>(hostOutput), n);
#endif
            const double seconds = secondsSince(start);

            if (result < 0) {
                fprintf(stderr, "%s: libcubwt error %lld\n", files[k], static_cast<long long>(result));
                return 1;
            }

            if (run > 0) {  // run 0 is the warmup
                best = std::min(best, seconds);
            }
        }

        const std::string name = baseName(files[k]);

        sumBytes += n;
        sumSeconds += best;
        printf("%-16s %12lld %9.4f %9.1f\n", name.c_str(), static_cast<long long>(n), best, n / best / 1e6);
        fflush(stdout);

        if (outDir) {
            writeOutput(outDir, name, hostOutput, n, result);
        }
    }

    if (files.size() > 1) {
        printf("%-16s %12lld %9.4f %9.1f\n", "total", static_cast<long long>(sumBytes), sumSeconds,
            sumBytes / sumSeconds / 1e6);
    }

    cudaFreeHost(hostText);
    cudaFreeHost(hostOutput);
    libcubwt_free_device_storage(storage);

    return 0;
}
