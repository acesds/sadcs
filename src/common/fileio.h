/*
 * fileio.h - File read and write helpers for the CLI.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Reads the whole file at path into out, followed by padding zero bytes (out.size() is the file size plus padding).
bool readWholeFile(const char* path, std::vector<uint8_t>& out, size_t padding = 0);

// Reads a raw little-endian int32 suffix array (the divsufsort mksary format).
bool readSuffixArray(const char* path, std::vector<int32_t>& out);

// Writes size bytes to path, replacing any existing file.
bool writeWholeFile(const char* path, const void* data, size_t size);

// Reads a BWT file in divsufsort's bwt/unbwt format with a single block.
bool readBwtFile(const char* path, std::vector<uint8_t>& out, int32_t& primary);

// Writes a BWT file in divsufsort's bwt/unbwt format with a single block.
bool writeBwtFile(const char* path, const uint8_t* bwt, int32_t n, int32_t primary);
