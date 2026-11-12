/*
 * fileio.cpp - File read and write helpers for the CLI.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#include "fileio.h"
#include <cstdio>

namespace {
    // File size in bytes, or -1; leaves fp at the start.
    int64_t fileSize(FILE* fp) {
        _fseeki64(fp, 0, SEEK_END);
        const int64_t size = _ftelli64(fp);
        _fseeki64(fp, 0, SEEK_SET);
        return size;
    }
}

bool readWholeFile(const char* path, std::vector<uint8_t>& out, size_t padding) {
    FILE* fp = fopen(path, "rb");

    if (!fp) {
        return false;
    }

    const int64_t size = fileSize(fp);

    if (size < 0) {
        fclose(fp);
        return false;
    }

    out.assign(static_cast<size_t>(size) + padding, 0);
    const size_t bytesRead = fread(out.data(), 1, static_cast<size_t>(size), fp);
    fclose(fp);

    return bytesRead == static_cast<size_t>(size);
}

bool readSuffixArray(const char* path, std::vector<int32_t>& out) {
    FILE* fp = fopen(path, "rb");

    if (!fp) {
        return false;
    }

    const int64_t size = fileSize(fp);

    if (size < 0 || size % static_cast<int64_t>(sizeof(int32_t)) != 0) {
        fclose(fp);
        return false;
    }

    out.resize(static_cast<size_t>(size) / sizeof(int32_t));
    const size_t entriesRead = fread(out.data(), sizeof(int32_t), out.size(), fp);
    fclose(fp);

    return entriesRead == out.size();
}

bool writeWholeFile(const char* path, const void* data, size_t size) {
    FILE* fp = fopen(path, "wb");

    if (!fp) {
        return false;
    }

    const size_t bytesWritten = fwrite(data, 1, size, fp);
    const bool closed = fclose(fp) == 0;

    return bytesWritten == size && closed;
}

bool readBwtFile(const char* path, std::vector<uint8_t>& out, int32_t& primary) {
    FILE* fp = fopen(path, "rb");

    if (!fp) {
        return false;
    }

    const int64_t size = fileSize(fp);
    int32_t blockSize = -1;
    primary = 0;
    bool ok = size >= 4 && fread(&blockSize, sizeof(blockSize), 1, fp) == 1 && blockSize >= 0;

    // A single block: the whole text, or nothing at all for an empty one.
    if (ok && blockSize == 0) {
        ok = size == 4;
        out.clear();
    } else if (ok) {
        ok = size == 8 + static_cast<int64_t>(blockSize) && fread(&primary, sizeof(primary), 1, fp) == 1;

        if (ok) {
            out.resize(static_cast<size_t>(blockSize));
            ok = fread(out.data(), 1, out.size(), fp) == out.size();
        }
    }

    fclose(fp);

    return ok;
}

bool writeBwtFile(const char* path, const uint8_t* bwt, int32_t n, int32_t primary) {
    FILE* fp = fopen(path, "wb");

    if (!fp) {
        return false;
    }

    bool ok = fwrite(&n, sizeof(n), 1, fp) == 1;

    if (ok && n > 0) {
        ok = fwrite(&primary, sizeof(primary), 1, fp) == 1 &&
             fwrite(bwt, 1, static_cast<size_t>(n), fp) == static_cast<size_t>(n);
    }

    const bool closed = fclose(fp) == 0;

    return ok && closed;
}
