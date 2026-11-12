/*
 * differencecover.h - Difference covers and the lookup tables built from them.
 *
 * A difference cover D modulo V is a set of residues such that every d in [0,V) equals (b-a) mod V for
 * some a, b in D. The sample is every text position i with (i mod V) in D. For any two positions i and j
 * there's then a k<V with i+k and j+k both sampled, so two suffixes agreeing on k characters are ordered
 * by the sample ranks at i+k and j+k. Covers modulo 32, 64 and 128 have the minimum size (7, 9 and 13);
 * for 256, whether 20 is minimal hasn't been settled by an exhaustive search.
 * The tables are generated from them at compile time.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

template <int V>
struct CoverSet;

// The GPU variant's cover (docs/gpu-design.md, "Choice of cover"); the CPU doesn't build with it.
template <>
struct CoverSet<32> {
    static constexpr std::array<int, 7> kElements{0, 1, 3, 8, 12, 19, 25};
};

template <>
struct CoverSet<64> {
    static constexpr std::array<int, 9> kElements{0, 1, 2, 5, 14, 16, 34, 42, 59};
};

template <>
struct CoverSet<128> {
    static constexpr std::array<int, 13> kElements{0, 6, 11, 26, 30, 34, 42, 43, 44, 75, 82, 104, 107};
};

template <>
struct CoverSet<256> {
    static constexpr std::array<int, 20> kElements{
        0, 24, 31, 34, 36, 54, 58, 123, 139, 147, 154, 165, 168, 186, 203, 219, 228, 237, 243, 244};
};

namespace cover_detail {
    constexpr uint16_t kNoAnchor = 0xFFFF;

    template <int V, size_t N>
    constexpr std::array<bool, V> makeMembership(const std::array<int, N>& elements) {
        std::array<bool, V> isMember{};

        for (const int element : elements) {
            isMember[element] = true;
        }

        return isMember;
    }

    // elementsBelow[r] = number of cover elements smaller than r.
    template <int V, size_t N>
    constexpr std::array<uint8_t, V> makeElementsBelow(const std::array<int, N>& elements) {
        std::array<uint8_t, V> elementsBelow{};

        for (int residue = 0; residue < V; residue++) {
            int count = 0;

            for (const int element : elements) {
                count += element < residue ? 1 : 0;
            }

            elementsBelow[residue] = static_cast<uint8_t>(count);
        }

        return elementsBelow;
    }

    // anchor[d] = some a in D with (a+d) mod V in D, or kNoAnchor.
    template <int V, size_t N>
    constexpr std::array<uint16_t, V> makeAnchors(const std::array<int, N>& elements) {
        const std::array<bool, V> isMember = makeMembership<V>(elements);
        std::array<uint16_t, V> anchor{};

        for (int distance = 0; distance < V; distance++) {
            anchor[distance] = kNoAnchor;

            for (const int element : elements) {
                if (isMember[(element + distance) & (V - 1)]) {
                    anchor[distance] = static_cast<uint16_t>(element);
                    break;
                }
            }
        }

        return anchor;
    }

    // nextSample[r] = distance from residue r to the next cover element (0 if r is one).
    template <int V, size_t N>
    constexpr std::array<uint16_t, V> makeNextSampleDistances(const std::array<int, N>& elements) {
        const std::array<bool, V> isMember = makeMembership<V>(elements);
        std::array<uint16_t, V> distance{};

        for (int residue = 0; residue < V; residue++) {
            int d = 0;

            while (!isMember[(residue + d) & (V - 1)]) {
                d++;
            }

            distance[residue] = static_cast<uint16_t>(d);
        }

        return distance;
    }

    template <size_t N>
    constexpr int maxValue(const std::array<uint16_t, N>& values) {
        int largest = 0;

        for (const uint16_t value : values) {
            largest = value > largest ? value : largest;
        }

        return largest;
    }

    template <int V, size_t N>
    constexpr bool isSortedAndUnique(const std::array<int, N>& elements) {
        for (size_t i = 1; i < N; i++) {
            if (elements[i - 1] >= elements[i]) {
                return false;
            }
        }

        return elements[0] == 0 && elements[N - 1] < V;
    }

    template <int V>
    constexpr bool coversEveryDistance(const std::array<uint16_t, V>& anchors) {
        for (const uint16_t anchor : anchors) {
            if (anchor == kNoAnchor) {
                return false;
            }
        }

        return true;
    }
}

template <int V>
class DifferenceCover {
 public:
    static_assert(std::has_single_bit(static_cast<unsigned>(V)), "the period must be a power of two");

    static constexpr int kMask = V - 1;
    static constexpr int kLog2 = std::countr_zero(static_cast<unsigned>(V));
    static constexpr int kSize = static_cast<int>(CoverSet<V>::kElements.size());

    static constexpr bool isSample(int32_t position) {
        return kIsMember[position & kMask];
    }

    // Number of sample positions below position: a sample position's index in the packed sample arrays,
    // and pack(n) is the sample size. pack(p+V) == pack(p)+kSize.
    static constexpr int32_t pack(int32_t position) {
        return (position >> kLog2) * kSize + kElementsBelow[position & kMask];
    }

    // The k in [0,V) with both i+k and j+k sampled.
    static constexpr int32_t tieOffset(int32_t i, int32_t j) {
        return (kAnchors[(j - i) & kMask] - i) & kMask;
    }

    // The distance from position to the next sample position, in [0,kNextSampleDistanceCount).
    static constexpr int32_t nextSampleDistance(int32_t position) {
        return kNextSampleDistances[position & kMask];
    }

 private:
    static constexpr std::array<bool, V> kIsMember = cover_detail::makeMembership<V>(CoverSet<V>::kElements);
    static constexpr std::array<uint8_t, V> kElementsBelow = cover_detail::makeElementsBelow<V>(CoverSet<V>::kElements);
    static constexpr std::array<uint16_t, V> kAnchors = cover_detail::makeAnchors<V>(CoverSet<V>::kElements);
    static constexpr std::array<uint16_t, V> kNextSampleDistances = cover_detail::makeNextSampleDistances<V>(CoverSet<V>::kElements);

 public:
    static constexpr int kNextSampleDistanceCount = cover_detail::maxValue(kNextSampleDistances) + 1;

    static_assert(cover_detail::isSortedAndUnique<V>(CoverSet<V>::kElements), "cover elements must be sorted, unique, start at 0");
    static_assert(cover_detail::coversEveryDistance<V>(kAnchors), "the cover set is not a difference cover");
};
