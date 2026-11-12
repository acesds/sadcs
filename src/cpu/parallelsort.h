/*
 * parallelsort.h - Task-parallel quicksort for distinct elements under a custom comparator.
 *
 * Used for tie groups too large for the scratch, and for large classes within tie groups.
 * Partitioning is sequential, the halves become tasks, small ranges go to std::sort.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

constexpr ptrdiff_t kSequentialSortMaxSize = 1 << 14;

template <class T, class Less>
T* medianOfThree(T* a, T* b, T* c, Less& less) {
    if (less(*b, *a)) {
        std::swap(a, b);
    }

    if (less(*c, *b)) {
        return less(*c, *a) ? a : c;
    }

    return b;
}

// As in introsort, after about 2 log2(size) partitions the rest goes to std::sort,
// so bad pivots can't make it quadratic. The smaller halves inherit what's left of the budget.
template <class T, class Less>
void parallelSort(T* first, T* last, Less less, int budget = -1) {
    if (budget < 0) {
        budget = 2 * static_cast<int>(std::bit_width(static_cast<size_t>(last - first)));
    }

    while (last - first > kSequentialSortMaxSize && budget > 0) {
        budget--;

        const ptrdiff_t size = last - first;
        const ptrdiff_t step = size / 8;
        T* middle = first + size / 2;
        T* pivot = medianOfThree(medianOfThree(first, first + step, first + 2 * step, less),
            medianOfThree(middle - step, middle, middle + step, less),
            medianOfThree(last - 1 - 2 * step, last - 1 - step, last - 1, less), less);

        std::swap(*first, *pivot);

        const T pivotValue = *first;
        T* greaterBegin = std::partition(first + 1, last, [&](const T& x) { return less(x, pivotValue); });
        T* lessEnd = greaterBegin - 1;

        std::swap(*first, *lessEnd);

        if (lessEnd - first < last - greaterBegin) {
            if (lessEnd - first > 1) {
#pragma omp task firstprivate(first, lessEnd, budget)
                parallelSort(first, lessEnd, less, budget);
            }

            first = greaterBegin;
        } else {
            if (last - greaterBegin > 1) {
#pragma omp task firstprivate(greaterBegin, last, budget)
                parallelSort(greaterBegin, last, less, budget);
            }

            last = lessEnd;
        }
    }

    std::sort(first, last, less);
}
