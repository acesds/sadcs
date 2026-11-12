/*
 * doubling.h - Prefix doubling: refines a partially sorted suffix array to a full one.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

// Refines sa to the full order of its n entries, in libdivsufsort's convention.
// On entry, the entries are sorted by their first depth symbols, where symbol j of entry s is isa[s+j].
// A run of finished entries starts with -(run length), and each of them has isa = its own slot.
// A tied group lists its entries as they are, and each has isa = the slot of the group's last entry.
// On return, isa[s] is the final rank of every entry, and sa is left as scratch.
// scratch (scratchSlots slots, any contents, 4 MB at most used) lets large groups be radix sorted; it may be empty.
void sortByPrefixDoubling(int32_t* isa, int32_t* sa, int32_t n, int32_t depth, int32_t* scratch, ptrdiff_t scratchSlots);
