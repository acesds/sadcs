/*
 * doubling.cpp - Larsson-Sadakane prefix doubling with tandem repeat handling.
 *
 * A tied group is sorted by isa[s+depth]; groups still tied go round again with the depth doubled.
 * The control flow follows libdivsufsort's trsort.c one to one, so it can be checked against it.
 * There's only one addition here: groups of 256 to 2^18 entries are radix sorted on cached keys
 * instead of partitioned.
 *
 * Tandem repeats: In periodic text members s of a group with isa[s+depth] == isa[s] point back into it,
 * which would take log(run/period) more rounds. Their run (keyed by the group's own rank) is its middle part.
 * The sort ends that run with an entry whose s+depth lies in the run or behind it, which marks the repeat
 *  The middle part is then ordered by simply copying from the finished parts on either side.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 *
 * Derived from trsort.c of libdivsufsort 2.0.2:
 *
 *   Copyright (c) 2003-2008 Yuta Mori All Rights Reserved.
 *
 *   Permission is hereby granted, free of charge, to any person
 *   obtaining a copy of this software and associated documentation
 *   files (the "Software"), to deal in the Software without
 *   restriction, including without limitation the rights to use,
 *   copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the
 *   Software is furnished to do so, subject to the following
 *   conditions:
 *
 *   The above copyright notice and this permission notice shall be
 *   included in all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *   OTHER DEALINGS IN THE SOFTWARE.
 */

#include "doubling.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <utility>

namespace {
    constexpr ptrdiff_t kInsertionSortMaxSize = 8;
    constexpr int kStackSize = 64;

    // Groups in this size range are radix sorted on cached keys instead of partitioned,
    // as far as the caller's scratch allows.
    constexpr ptrdiff_t kRadixMinSize = 256;
    constexpr ptrdiff_t kRadixMaxSize = 1 << 18;
    constexpr int kRadixMaxBits = 11;

    // sortGroup's limit: >= 0 is the remaining introsort depth, these are pending actions.
    constexpr int32_t kTandemPartition = -1;
    constexpr int32_t kTandemCopy = -2;
    constexpr int32_t kSortedPartition = -3;

    // A tandem repeat's middle-part frame keeps its copy mode in the limit field.
    constexpr int32_t kFullCopy = 0;
    constexpr int32_t kPartialCopy = -1;

    // floor(log2(n)), -1 for 0.
    int32_t floorLog2(ptrdiff_t n) {
        return 31 - std::countl_zero(static_cast<uint32_t>(n));
    }

    int32_t slotOf(const int32_t* sa, const int32_t* p) {
        return static_cast<int32_t>(p - sa);
    }

    struct Budget {
        int32_t refillsLeft;
        int32_t remaining;
        int32_t refillSize;
        int32_t deferred;  // total size of the groups left for the next round

        bool spend(ptrdiff_t size) {
            const int32_t cost = static_cast<int32_t>(size);

            if (cost <= remaining) {
                remaining -= cost;
                return true;
            }

            if (refillsLeft == 0) {
                deferred += cost;
                return false;
            }

            remaining += refillSize - cost;
            refillsLeft--;
            return true;
        }
    };

    // Ties are marked by complementing the earlier entry, so marked entries move with their run.
    void insertionSortByKey(const int32_t* keys, int32_t* first, int32_t* last) {
        for (int32_t* current = first + 1; current < last; current++) {
            const int32_t entry = *current;
            int32_t* it = current - 1;
            int32_t order = keys[entry] - keys[*it];

            while (order < 0) {
                do {
                    *(it + 1) = *it;
                    it--;
                } while (it >= first && *it < 0);

                if (it < first) {
                    break;
                }

                order = keys[entry] - keys[*it];
            }

            if (order == 0) {
                *it = ~*it;
            }

            *(it + 1) = entry;
        }
    }

    void siftDown(const int32_t* keys, int32_t* sa, int32_t parent, int32_t size) {
        const int32_t entry = sa[parent];
        const int32_t entryKey = keys[entry];

        for (;;) {
            const int32_t left = 2 * parent + 1;

            if (left >= size) {
                break;
            }

            const int32_t right = left + 1;
            int32_t larger = left;
            int32_t largerKey = keys[sa[left]];
            const int32_t rightKey = keys[sa[right]];

            if (largerKey < rightKey) {
                larger = right;
                largerKey = rightKey;
            }

            if (largerKey <= entryKey) {
                break;
            }

            sa[parent] = sa[larger];
            parent = larger;
        }

        sa[parent] = entry;
    }

    void heapSortByKey(const int32_t* keys, int32_t* sa, int32_t size) {
        int32_t heapSize = size;

        if ((size % 2) == 0) {
            heapSize--;

            if (keys[sa[heapSize / 2]] < keys[sa[heapSize]]) {
                std::swap(sa[heapSize], sa[heapSize / 2]);
            }
        }

        for (int32_t parent = heapSize / 2 - 1; parent >= 0; parent--) {
            siftDown(keys, sa, parent, heapSize);
        }

        if ((size % 2) == 0) {
            std::swap(sa[0], sa[heapSize]);
            siftDown(keys, sa, 0, heapSize);
        }

        for (int32_t end = heapSize - 1; end > 0; end--) {
            const int32_t largest = sa[0];

            sa[0] = sa[end];
            siftDown(keys, sa, 0, end);
            sa[end] = largest;
        }
    }

    struct RadixBuffer {
        uint64_t* entries;  // 2 * maxSize
        ptrdiff_t maxSize;
    };

    // LSD radix sort of the group [first, last) by keys through buffer (2 * size entries),
    // with the same tie marking as insertionSortByKey. Each entry is cached with its key,
    // so the passes run sequentially instead of looking the key up again.
    // Returns false, leaving the group as it was, if all keys are equal.
    // Like the partition, it marks a tandem repeat (see the top of the file).
    // Kept out of sortGroup, whose hot loop over tiny groups gets slower with it inlined.
#if defined(_MSC_VER)
    __declspec(noinline)
#else
    __attribute__((noinline))
#endif
    bool radixSortByKey(const int32_t* isa, const int32_t* keys, int32_t* sa, int32_t* first, int32_t* last, uint64_t* buffer) {
        const ptrdiff_t size = last - first;
        uint64_t* from = buffer;
        uint64_t* to = buffer + size;
        uint32_t minKey = UINT32_MAX;
        uint32_t maxKey = 0;

        for (ptrdiff_t i = 0; i < size; i++) {
            const uint32_t key = static_cast<uint32_t>(keys[first[i]]) ^ 0x80000000u;

            minKey = std::min(minKey, key);
            maxKey = std::max(maxKey, key);
            from[i] = (static_cast<uint64_t>(key) << 32) | static_cast<uint32_t>(first[i]);
        }

        if (minKey == maxKey) {
            return false;
        }

        // Only the bits in which the keys differ, in as few passes as small count tables allow.
        const int keyBits = 32 - std::countl_zero(maxKey - minKey);
        const int maxBits = std::clamp(floorLog2(size), 4, kRadixMaxBits);
        const int passes = (keyBits + maxBits - 1) / maxBits;
        const int bits = (keyBits + passes - 1) / passes;
        const uint32_t mask = (1u << bits) - 1;
        const uint64_t bias = static_cast<uint64_t>(minKey) << 32;
        int32_t counts[1 << kRadixMaxBits];

        for (int pass = 0; pass < passes; pass++) {
            const int shift = 32 + pass * bits;

            std::fill_n(counts, mask + 1, 0);

            for (ptrdiff_t i = 0; i < size; i++) {
                counts[((from[i] - bias) >> shift) & mask]++;
            }

            int32_t sum = 0;

            for (uint32_t digit = 0; digit <= mask; digit++) {
                const int32_t count = counts[digit];
                counts[digit] = sum;
                sum += count;
            }

            for (ptrdiff_t i = 0; i < size; i++) {
                to[counts[((from[i] - bias) >> shift) & mask]++] = from[i];
            }

            std::swap(from, to);
        }

        const int32_t ownRank = slotOf(sa, last) - 1;
        const uint64_t ownKey = static_cast<uint32_t>(ownRank) ^ 0x80000000u;
        const ptrdiff_t depth = keys - isa;
        ptrdiff_t tandemFirst = -1;

        for (ptrdiff_t i = 0; i < size - 1; i++) {
            const int32_t entry = static_cast<int32_t>(from[i]);
            const uint64_t key = from[i] >> 32;
            const bool tied = key == (from[i + 1] >> 32);

            first[i] = tied ? ~entry : entry;

            if (tied && key == ownKey && tandemFirst < 0) {
                tandemFirst = i;
            }
        }

        first[size - 1] = static_cast<int32_t>(from[size - 1]);

        if (tandemFirst >= 0) {
            ptrdiff_t tandemLast = tandemFirst;

            while (first[tandemLast] < 0) {
                tandemLast++;
            }

            for (ptrdiff_t i = tandemLast; tandemFirst <= i; i--) {
                const int32_t entry = first[i] < 0 ? ~first[i] : first[i];

                if (ownRank <= keys[entry + depth]) {
                    if (i != tandemLast) {
                        first[i] = ~first[tandemLast];
                        first[tandemLast] = entry;
                    }

                    break;
                }
            }
        }

        return true;
    }

    int32_t* medianOfThree(const int32_t* keys, int32_t* v1, int32_t* v2, int32_t* v3) {
        if (keys[*v1] > keys[*v2]) {
            std::swap(v1, v2);
        }

        if (keys[*v2] > keys[*v3]) {
            return keys[*v1] > keys[*v3] ? v1 : v3;
        }

        return v2;
    }

    int32_t* medianOfFive(const int32_t* keys, int32_t* v1, int32_t* v2, int32_t* v3, int32_t* v4, int32_t* v5) {
        if (keys[*v2] > keys[*v3]) {
            std::swap(v2, v3);
        }

        if (keys[*v4] > keys[*v5]) {
            std::swap(v4, v5);
        }

        if (keys[*v2] > keys[*v4]) {
            std::swap(v2, v4);
            std::swap(v3, v5);
        }

        if (keys[*v1] > keys[*v3]) {
            std::swap(v1, v3);
        }

        if (keys[*v1] > keys[*v4]) {
            std::swap(v1, v4);
            std::swap(v3, v5);
        }

        return keys[*v3] > keys[*v4] ? v4 : v3;
    }

    int32_t* choosePivot(const int32_t* keys, int32_t* first, int32_t* last) {
        ptrdiff_t step = last - first;
        int32_t* middle = first + step / 2;

        if (step <= 32) {
            return medianOfThree(keys, first, middle, last - 1);
        }

        if (step <= 512) {
            step >>= 2;
            return medianOfFive(keys, first, first + step, middle, last - 1 - step, last - 1);
        }

        step >>= 3;
        first = medianOfThree(keys, first, first + step, first + (step << 1));
        middle = medianOfThree(keys, middle - step, middle, middle + step);
        last = medianOfThree(keys, last - 1 - (step << 1), last - 1 - step, last - 1);
        return medianOfThree(keys, first, middle, last);
    }

    // Three-way partition of [first, last) around pivotKey, where [first, middle) is known to equal it.
    // Equal keys collect at both ends during the scan and are swapped to the middle at the end.
    // Returns the equal part as [*equalFirst, *equalLast).
    void partitionByKey(const int32_t* keys, int32_t* first, int32_t* middle, int32_t* last, int32_t** equalFirst, int32_t** equalLast, int32_t pivotKey) {
        int32_t* lo = middle;
        int32_t* hi = last - 1;
        int32_t* equalLeftEnd;
        int32_t* equalRightEdge;

        // Moves lo up to the first key above the pivot (or to end), collecting equal keys at equalLeftEnd.
        auto scanUp = [&](int32_t* end) {
            for (; lo < end; lo++) {
                const int32_t key = keys[*lo];

                if (key > pivotKey) {
                    break;
                }

                if (key == pivotKey) {
                    std::swap(*lo, *equalLeftEnd++);
                }
            }
        };

        // Moves hi down to the first key below the pivot (or to lo), collecting equal keys at equalRightEdge.
        auto scanDown = [&]() {
            for (; lo < hi; hi--) {
                const int32_t key = keys[*hi];

                if (key < pivotKey) {
                    break;
                }

                if (key == pivotKey) {
                    std::swap(*hi, *equalRightEdge--);
                }
            }
        };

        while (lo < last && keys[*lo] == pivotKey) {
            lo++;
        }

        equalLeftEnd = lo;

        if (lo < last && keys[*lo] < pivotKey) {
            lo++;
            scanUp(last);
        }

        while (lo < hi && keys[*hi] == pivotKey) {
            hi--;
        }

        equalRightEdge = hi;

        if (lo < hi && keys[*hi] > pivotKey) {
            hi--;
            scanDown();
        }

        while (lo < hi) {
            std::swap(*lo, *hi);
            lo++;
            scanUp(hi);
            hi--;
            scanDown();
        }

        if (equalLeftEnd <= equalRightEdge) {
            hi = lo - 1;

            const ptrdiff_t leftCount = std::min(equalLeftEnd - first, lo - equalLeftEnd);
            const ptrdiff_t rightCount = std::min(equalRightEdge - hi, last - equalRightEdge - 1);

            std::swap_ranges(first, first + leftCount, lo - leftCount);
            std::swap_ranges(lo, lo + rightCount, last - rightCount);
            first += lo - equalLeftEnd;
            last -= equalRightEdge - hi;
        }

        *equalFirst = first;
        *equalLast = last;
    }

    // Orders the middle part [middleFirst, middleLast) of a tandem repeat.
    // This is done by walking the finished left part forwards and the right part backwards.
    // Every entry whose predecessor is still in the middle group places that predecessor next.
    void copyTandemOrder(int32_t* isa, int32_t* sa, int32_t* first, int32_t* middleFirst, int32_t* middleLast, int32_t* last, int32_t depth) {
        const int32_t middleRank = slotOf(sa, middleLast) - 1;
        int32_t* target = middleFirst - 1;

        for (int32_t* source = first; source <= target; source++) {
            const int32_t predecessor = *source - depth;

            if (predecessor >= 0 && isa[predecessor] == middleRank) {
                *++target = predecessor;
                isa[predecessor] = slotOf(sa, target);
            }
        }

        int32_t* const filledEnd = target + 1;

        target = middleLast;

        for (int32_t* source = last - 1; filledEnd < target; source--) {
            const int32_t predecessor = *source - depth;

            if (predecessor >= 0 && isa[predecessor] == middleRank) {
                *--target = predecessor;
                isa[predecessor] = slotOf(sa, target);
            }
        }
    }

    // copyTandemOrder for when the outer parts aren't fully sorted yet:
    // copied entries get group ranks (by their next key) instead of final ones.
    void copyTandemOrderPartial(int32_t* isa, int32_t* sa, int32_t* first, int32_t* middleFirst, int32_t* middleLast, int32_t* last, int32_t depth) {
        const int32_t middleRank = slotOf(sa, middleLast) - 1;
        int32_t previousKey = -1;
        int32_t newRank = -1;
        int32_t* target = middleFirst - 1;

        for (int32_t* source = first; source <= target; source++) {
            const int32_t predecessor = *source - depth;

            if (predecessor >= 0 && isa[predecessor] == middleRank) {
                *++target = predecessor;
                const int32_t key = isa[predecessor + depth];

                if (previousKey != key) {
                    previousKey = key;
                    newRank = slotOf(sa, target);
                }

                isa[predecessor] = newRank;
            }
        }

        previousKey = -1;

        for (int32_t* it = target; first <= it; it--) {
            const int32_t rank = isa[*it];

            if (previousKey != rank) {
                previousKey = rank;
                newRank = slotOf(sa, it);
            }

            if (newRank != rank) {
                isa[*it] = newRank;
            }
        }

        previousKey = -1;
        int32_t* const filledEnd = target + 1;

        target = middleLast;

        for (int32_t* source = last - 1; filledEnd < target; source--) {
            const int32_t predecessor = *source - depth;

            if (predecessor >= 0 && isa[predecessor] == middleRank) {
                *--target = predecessor;
                const int32_t key = isa[predecessor + depth];

                if (previousKey != key) {
                    previousKey = key;
                    newRank = slotOf(sa, target);
                }

                isa[predecessor] = newRank;
            }
        }
    }

    // Introsort of one tied group [first, last) by keys,
    // descending into tied subgroups with the keys one depth step further, with an explicit stack.
    void sortGroup(int32_t* isa, const int32_t* keys, int32_t* sa, int32_t* first, int32_t* last, Budget& budget, const RadixBuffer& radix) {
        struct Frame {
            const int32_t* keys;
            int32_t* first;
            int32_t* last;
            int32_t limit;
            int32_t tandemLink;
        };

        Frame stack[kStackSize];
        int stackSize = 0;
        const int32_t depthStep = static_cast<int32_t>(keys - isa);
        int32_t limit = floorLog2(last - first);
        int32_t tandemLink = -1;  // stack index of the enclosing tandem repeat's middle part
        int32_t* equalFirst;
        int32_t* equalLast;

        auto push = [&](const int32_t* frameKeys, int32_t* frameFirst, int32_t* frameLast, int32_t frameLimit, int32_t frameLink) {
            assert(stackSize < kStackSize);
            stack[stackSize++] = Frame{frameKeys, frameFirst, frameLast, frameLimit, frameLink};
        };

        auto pop = [&]() {
            if (stackSize == 0) {
                return false;
            }

            const Frame& frame = stack[--stackSize];

            keys = frame.keys;
            first = frame.first;
            last = frame.last;
            limit = frame.limit;
            tandemLink = frame.tandemLink;

            return true;
        };

        auto assignRank = [&](int32_t* from, int32_t* to, int32_t rank) {
            for (; from < to; from++) {
                isa[*from] = rank;
            }
        };

        auto markTandemCopyPartial = [&]() {
            if (tandemLink >= 0) {
                stack[tandemLink].limit = kPartialCopy;
            }
        };

        auto descendInto = [&](int32_t* groupFirst, int32_t* groupLast, int32_t groupLimit) {
            keys += depthStep;
            first = groupFirst;
            last = groupLast;
            limit = groupLimit;
        };

        for (;;) {
            if (limit == kTandemPartition) {
                partitionByKey(keys - depthStep, first, first, last, &equalFirst, &equalLast, slotOf(sa, last) - 1);

                if (equalFirst < last) {
                    assignRank(first, equalFirst, slotOf(sa, equalFirst) - 1);
                }

                if (equalLast < last) {
                    assignRank(equalFirst, equalLast, slotOf(sa, equalLast) - 1);
                }

                if (equalLast - equalFirst > 1) {
                    push(nullptr, equalFirst, equalLast, kFullCopy, 0);
                    push(keys - depthStep, first, last, kTandemCopy, tandemLink);
                    tandemLink = stackSize - 2;
                }

                if ((equalFirst - first) <= (last - equalLast)) {
                    if (equalFirst - first > 1) {
                        push(keys, equalLast, last, floorLog2(last - equalLast), tandemLink);
                        last = equalFirst;
                        limit = floorLog2(equalFirst - first);
                    } else if (last - equalLast > 1) {
                        first = equalLast;
                        limit = floorLog2(last - equalLast);
                    } else if (!pop()) {
                        return;
                    }
                } else {
                    if (last - equalLast > 1) {
                        push(keys, first, equalFirst, floorLog2(equalFirst - first), tandemLink);
                        first = equalLast;
                        limit = floorLog2(last - equalLast);
                    } else if (equalFirst - first > 1) {
                        last = equalFirst;
                        limit = floorLog2(equalFirst - first);
                    } else if (!pop()) {
                        return;
                    }
                }

                continue;
            }

            if (limit == kTandemCopy) {
                const Frame& middle = stack[--stackSize];
                const int32_t depth = static_cast<int32_t>(keys - isa);

                if (middle.limit == kFullCopy) {
                    copyTandemOrder(isa, sa, first, middle.first, middle.last, last, depth);
                } else {
                    markTandemCopyPartial();
                    copyTandemOrderPartial(isa, sa, first, middle.first, middle.last, last, depth);
                }

                if (!pop()) {
                    return;
                }

                continue;
            }

            if (limit == kSortedPartition) {
                if (*first >= 0) {
                    int32_t* it = first;

                    do {
                        isa[*it] = slotOf(sa, it);
                        it++;
                    } while (it < last && *it >= 0);

                    first = it;
                }

                if (last <= first) {
                    if (!pop()) {
                        return;
                    }

                    continue;
                }

                int32_t* groupLast = first;

                do {
                    *groupLast = ~*groupLast;
                    groupLast++;
                } while (*groupLast < 0);

                const int32_t nextLimit =
                    isa[*groupLast] != keys[*groupLast] ? floorLog2(groupLast - first + 1) : kTandemPartition;
                int32_t* const groupEnd = groupLast + 1;

                if (groupEnd < last) {
                    assignRank(first, groupEnd, slotOf(sa, groupEnd) - 1);
                }

                if (budget.spend(groupEnd - first)) {
                    if ((groupEnd - first) <= (last - groupEnd)) {
                        push(keys, groupEnd, last, kSortedPartition, tandemLink);
                        descendInto(first, groupEnd, nextLimit);
                    } else if (last - groupEnd > 1) {
                        push(keys + depthStep, first, groupEnd, nextLimit, tandemLink);
                        first = groupEnd;
                        limit = kSortedPartition;
                    } else {
                        descendInto(first, groupEnd, nextLimit);
                    }
                } else {
                    markTandemCopyPartial();

                    if (last - groupEnd > 1) {
                        first = groupEnd;
                        limit = kSortedPartition;
                    } else if (!pop()) {
                        return;
                    }
                }

                continue;
            }

            if (last - first <= kInsertionSortMaxSize) {
                insertionSortByKey(keys, first, last);
                limit = kSortedPartition;
                continue;
            }

            const ptrdiff_t size = last - first;

            if (size >= kRadixMinSize && size <= radix.maxSize) {
                if (radixSortByKey(isa, keys, sa, first, last, radix.entries)) {
                    limit = kSortedPartition;
                } else if (budget.spend(last - first)) {
                    limit = floorLog2(last - first);
                    keys += depthStep;
                } else {
                    markTandemCopyPartial();

                    if (!pop()) {
                        return;
                    }
                }

                continue;
            }

            if (limit == 0) {
                heapSortByKey(keys, first, static_cast<int32_t>(last - first));

                for (int32_t* it = last - 1; first < it;) {
                    const int32_t key = keys[*it];
                    int32_t* before = it - 1;

                    for (; first <= before && keys[*before] == key; before--) {
                        *before = ~*before;
                    }

                    it = before;
                }

                limit = kSortedPartition;
                continue;
            }

            limit--;
            std::swap(*first, *choosePivot(keys, first, last));

            const int32_t pivotKey = keys[*first];

            partitionByKey(keys, first, first + 1, last, &equalFirst, &equalLast, pivotKey);

            if (last - first == equalLast - equalFirst) {
                if (budget.spend(last - first)) {
                    limit = floorLog2(last - first);
                    keys += depthStep;
                } else {
                    markTandemCopyPartial();

                    if (!pop()) {
                        return;
                    }
                }

                continue;
            }

            const int32_t nextLimit = isa[*equalFirst] != pivotKey ? floorLog2(equalLast - equalFirst) : kTandemPartition;

            assignRank(first, equalFirst, slotOf(sa, equalFirst) - 1);

            if (equalLast < last) {
                assignRank(equalFirst, equalLast, slotOf(sa, equalLast) - 1);
            }

            const ptrdiff_t lessSize = equalFirst - first;
            const ptrdiff_t equalSize = equalLast - equalFirst;
            const ptrdiff_t greaterSize = last - equalLast;

            if (equalSize > 1 && budget.spend(equalSize)) {
                if (lessSize <= greaterSize) {
                    if (greaterSize <= equalSize) {
                        if (lessSize > 1) {
                            push(keys + depthStep, equalFirst, equalLast, nextLimit, tandemLink);
                            push(keys, equalLast, last, limit, tandemLink);
                            last = equalFirst;
                        } else if (greaterSize > 1) {
                            push(keys + depthStep, equalFirst, equalLast, nextLimit, tandemLink);
                            first = equalLast;
                        } else {
                            descendInto(equalFirst, equalLast, nextLimit);
                        }
                    } else if (lessSize <= equalSize) {
                        if (lessSize > 1) {
                            push(keys, equalLast, last, limit, tandemLink);
                            push(keys + depthStep, equalFirst, equalLast, nextLimit, tandemLink);
                            last = equalFirst;
                        } else {
                            push(keys, equalLast, last, limit, tandemLink);
                            descendInto(equalFirst, equalLast, nextLimit);
                        }
                    } else {
                        push(keys, equalLast, last, limit, tandemLink);
                        push(keys, first, equalFirst, limit, tandemLink);
                        descendInto(equalFirst, equalLast, nextLimit);
                    }
                } else {
                    if (lessSize <= equalSize) {
                        if (greaterSize > 1) {
                            push(keys + depthStep, equalFirst, equalLast, nextLimit, tandemLink);
                            push(keys, first, equalFirst, limit, tandemLink);
                            first = equalLast;
                        } else if (lessSize > 1) {
                            push(keys + depthStep, equalFirst, equalLast, nextLimit, tandemLink);
                            last = equalFirst;
                        } else {
                            descendInto(equalFirst, equalLast, nextLimit);
                        }
                    } else if (greaterSize <= equalSize) {
                        if (greaterSize > 1) {
                            push(keys, first, equalFirst, limit, tandemLink);
                            push(keys + depthStep, equalFirst, equalLast, nextLimit, tandemLink);
                            first = equalLast;
                        } else {
                            push(keys, first, equalFirst, limit, tandemLink);
                            descendInto(equalFirst, equalLast, nextLimit);
                        }
                    } else {
                        push(keys, first, equalFirst, limit, tandemLink);
                        push(keys, equalLast, last, limit, tandemLink);
                        descendInto(equalFirst, equalLast, nextLimit);
                    }
                }
            } else {
                if (equalSize > 1) {
                    markTandemCopyPartial();
                }

                if (lessSize <= greaterSize) {
                    if (lessSize > 1) {
                        push(keys, equalLast, last, limit, tandemLink);
                        last = equalFirst;
                    } else if (greaterSize > 1) {
                        first = equalLast;
                    } else if (!pop()) {
                        return;
                    }
                } else {
                    if (greaterSize > 1) {
                        push(keys, first, equalFirst, limit, tandemLink);
                        first = equalLast;
                    } else if (lessSize > 1) {
                        last = equalFirst;
                    } else if (!pop()) {
                        return;
                    }
                }
            }
        }
    }
}

void sortByPrefixDoubling(int32_t* isa, int32_t* sa, int32_t n, int32_t depth, int32_t* scratch, ptrdiff_t scratchSlots) {
    if (n <= 1) {
        if (n == 1) {
            isa[0] = 0;
        }

        return;
    }

    Budget budget{floorLog2(n) * 2 / 3, n, n, 0};
    const ptrdiff_t misalignment = (reinterpret_cast<uintptr_t>(scratch) % alignof(uint64_t)) / sizeof(int32_t);
    const RadixBuffer radix{reinterpret_cast<uint64_t*>(scratch + misalignment),
        std::min<ptrdiff_t>({kRadixMaxSize, n, std::max<ptrdiff_t>(scratchSlots - misalignment, 0) / 4})};

    for (const int32_t* keys = isa + depth; *sa > -n; keys += keys - isa) {
        int32_t* first = sa;
        int32_t finishedRun = 0;  // minus the length of the finished run being collected
        int32_t deferredTotal = 0;

        do {
            const int32_t entry = *first;

            if (entry < 0) {
                first -= entry;
                finishedRun += entry;
                continue;
            }

            if (finishedRun != 0) {
                *(first + finishedRun) = finishedRun;
                finishedRun = 0;
            }

            int32_t* last = sa + isa[entry] + 1;

            if (last - first > 1) {
                budget.deferred = 0;
                sortGroup(isa, keys, sa, first, last, budget, radix);

                if (budget.deferred != 0) {
                    deferredTotal += budget.deferred;
                } else {
                    finishedRun = static_cast<int32_t>(first - last);
                }
            } else if (last - first == 1) {
                finishedRun = -1;
            }

            first = last;
        } while (first < sa + n);

        if (finishedRun != 0) {
            *(first + finishedRun) = finishedRun;
        }

        if (deferredTotal == 0) {
            break;
        }
    }
}
