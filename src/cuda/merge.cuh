/*
 * merge.cuh - Step 4 of docs/gpu-design.md: one multiway merge of the 7 sorted class lists.
 *
 *   1. Every G-th element of each list is a splitter; the splitters are sorted by this merge one level down.
 *   2. Binary searches in all lists give each splitter its output position (searchLists()).
 *   3. The output is cut into tiles of about T elements at splitters.
 *   4. One block per tile merges its sublists in shared memory (mergeTile(), mergeOverflow()).
 *
 * Steps 1 to 3 of the top level work in the output buffer, so they don't add to the peak.
 *
 * Part of sadcs, a suffix array and BWT construction library based on a difference cover.
 * Copyright (c) 2026 Alex Schumacher
 * Licensed under the MIT License; see the LICENSE file for details.
 * https://opensource.org/licenses/MIT
 */

#pragma once

#include <algorithm>
#include <memory>
#include <cub/cub.cuh>
#include <cuda/iterator>
#include "classsort.cuh"
#include "devicecommon.cuh"

namespace gpusa {
    // Per splitter: its position in each of the 7 lists, then their sum.
    constexpr int kPositionStride = 8;

    struct MergeInput {
        const uint64_t* text;   // padded with zero words
        const uint32_t* ranks;  // compact, padded
        const uint32_t* lists;  // text positions, the sorted lists concatenated
        uint32_t n;
        ClassLists layout;
    };

    __device__ __forceinline__ int listOf(const ClassLists& layout, uint32_t index) {
        int l = 0;

        while (index >= layout.start[l + 1]) {
            l++;
        }

        return l;
    }

    __device__ __forceinline__ uint64_t prefixMask(uint32_t remaining) {
        return remaining >= 8 ? ~uint64_t(0) : ~uint64_t(0) << (64 - 8 * remaining);
    }

    // Suffix i < suffix j, reading straight from global memory.
    __device__ bool lessGlobal(const MergeInput& in, uint32_t i, uint32_t j) {
        const uint32_t k = cTieOffset[i % kCoverPeriod][j % kCoverPeriod];
        const uint32_t common = min(k, min(in.n - i, in.n - j));

        for (uint32_t o = 0; o < common; o += 8) {
            const uint64_t mask = prefixMask(common - o);
            const uint64_t a = load8(in.text, i + o) & mask, b = load8(in.text, j + o) & mask;

            if (a != b) {
                return a < b;
            }
        }

        if (common < k || i + k == in.n || j + k == in.n) {
            return i > j;  // one is a prefix of the other
        }

        return in.ranks[compactIndex(i + k)] < in.ranks[compactIndex(j + k)];
    }

    // One element's comparison data:
    // its first 16 characters (comparisons reaching further, 5 to 15% of them, read the rest from global memory)
    // and the ranks of the 7 sample positions in its window, all a comparison can need.
    constexpr int kRecordWords = 2;

    struct __align__(16) MergeRecord {
        uint64_t words[kRecordWords];
        uint32_t ranks[kCoverSize];
        uint32_t position;
    };

    static_assert(sizeof(MergeRecord) == 48, "a merge record is 48 bytes");

    __device__ __forceinline__ void loadRecord(const MergeInput& in, uint32_t p, MergeRecord& record) {
#pragma unroll
        for (int w = 0; w < kRecordWords; w++) {
            record.words[w] = load8(in.text, p + 8 * w);
        }

        const uint32_t base = compactIndex(p);

#pragma unroll
        for (int s = 0; s < kCoverSize; s++) {
            record.ranks[s] = in.ranks[base + s];
        }

        record.position = p;
    }

    __device__ __forceinline__ bool lessRecord(const MergeInput& in, const MergeRecord& a, const MergeRecord& b) {
        const uint32_t n = in.n;
        const uint32_t i = a.position, j = b.position;
        const uint32_t k = cTieOffset[i % kCoverPeriod][j % kCoverPeriod];
        const uint32_t common = min(k, min(n - i, n - j));

#pragma unroll
        for (uint32_t w = 0; w < 4; w++) {
            if (8 * w >= common) {
                break;
            }

            const uint64_t mask = prefixMask(common - 8 * w);
            const uint64_t x = (w < kRecordWords ? a.words[w] : load8(in.text, i + 8 * w)) & mask;
            const uint64_t y = (w < kRecordWords ? b.words[w] : load8(in.text, j + 8 * w)) & mask;

            if (x != y) {
                return x < y;
            }
        }

        if (common < k || i + k == n || j + k == n) {
            return i > j;
        }

        return a.ranks[cWindowSlot[i % kCoverPeriod][k]] < b.ranks[cWindowSlot[j % kCoverPeriod][k]];
    }

    // ---- step 1: splitters -------------------------------------------------------------------

    // sorted[t] = index of the t-th splitter in the concatenated lists: element q * g of its list.
    __global__ void pickSplitters(ClassLists layout, ClassLists splitters, uint32_t g, uint32_t count, uint32_t* sorted) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < count) {
            const int l = listOf(splitters, t);

            sorted[t] = layout.start[l] + (t - splitters.start[l]) * g;
        }
    }

    // The splitters as lists of text positions, for the level below.
    __global__ void gatherSplitterPositions(MergeInput in, ClassLists splitters, uint32_t g, uint32_t count, uint32_t* out) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < count) {
            const int l = listOf(splitters, t);

            out[t] = in.lists[in.layout.start[l] + (t - splitters.start[l]) * g];
        }
    }

    // The level below returns indices into its own lists; map them back to this level's.
    __global__ void childToParent(ClassLists layout, ClassLists splitters, uint32_t g, const uint32_t* childSorted, uint32_t count, uint32_t* sorted) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < count) {
            const uint32_t j = childSorted[t];
            const int l = listOf(splitters, j);

            sorted[t] = layout.start[l] + (j - splitters.start[l]) * g;
        }
    }

    struct SplitterLess {
        MergeInput in;

        __device__ bool operator()(uint32_t a, uint32_t b) const {
            return lessGlobal(in, in.lists[a], in.lists[b]);
        }
    };

    // ---- step 2: positions of the splitters in every list ------------------------------------

    // 1 for the splitters (indices into the concatenated lists) in list l.
    struct InList {
        ClassLists layout;
        int l;

        __device__ uint32_t operator()(uint32_t index) const {
            return listOf(layout, index) == l ? 1 : 0;
        }
    };

    // Suffix i < the suffix whose record r holds; r's text and ranks come from registers.
    __device__ __forceinline__ bool lessThanRecord(const MergeInput& in, uint32_t i, const MergeRecord& r) {
        const uint32_t n = in.n;
        const uint32_t j = r.position;
        const uint32_t k = cTieOffset[i % kCoverPeriod][j % kCoverPeriod];
        const uint32_t common = min(k, min(n - i, n - j));

#pragma unroll
        for (uint32_t w = 0; w < 4; w++) {
            if (8 * w >= common) {
                break;
            }

            const uint64_t mask = prefixMask(common - 8 * w);
            const uint64_t x = load8(in.text, i + 8 * w) & mask;
            const uint64_t y = (w < kRecordWords ? r.words[w] : load8(in.text, j + 8 * w)) & mask;

            if (x != y) {
                return x < y;
            }
        }

        if (common < k || i + k == n || j + k == n) {
            return i > j;
        }

        return in.ranks[compactIndex(i + k)] < r.ranks[cWindowSlot[j % kCoverPeriod][k]];
    }

    // The positions of splitter t in every list.
    // Row t holds on entry, for each list l, the number of l's splitters before t: those are smaller,
    // the next one larger, which leaves a window of g - 1 elements to search.
    // The splitter's own record stays in registers.
    __global__ void searchLists(MergeInput in, const uint32_t* sorted, uint32_t* positions, uint32_t count, uint32_t g) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t >= count) {
            return;
        }

        const uint32_t index = sorted[t];
        const int own = listOf(in.layout, index);
        uint32_t* const row = positions + static_cast<size_t>(t) * kPositionStride;
        MergeRecord splitter;

        loadRecord(in, in.lists[index], splitter);

        for (int l = 0; l < kClasses; l++) {
            const uint32_t start = in.layout.start[l];

            if (l == own) {
                row[l] = index - start;
                continue;
            }

            const uint32_t length = in.layout.start[l + 1] - start;
            const uint32_t b = row[l];
            uint32_t lo = b == 0 ? 0 : (b - 1) * g + 1;
            uint32_t hi = min(b * g, length);
            const uint32_t* list = in.lists + start;

            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;

                if (lessThanRecord(in, list[mid], splitter)) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            row[l] = lo;
        }
    }

    __global__ void sumPositions(uint32_t* positions, uint32_t count) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;

        if (t < count) {
            uint32_t sum = 0;

            for (int l = 0; l < kClasses; l++) {
                sum += positions[static_cast<size_t>(t) * kPositionStride + l];
            }

            positions[static_cast<size_t>(t) * kPositionStride + kClasses] = sum;
        }
    }

    // ---- step 3: tiles -----------------------------------------------------------------------

    // tileStart[k] = the last splitter (or the end sentinel) whose output position is <= k * T.
    __global__ void cutTiles(const uint32_t* positions, uint32_t count, uint32_t* tileStart, uint32_t tiles, uint32_t tileTarget) {
        const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

        if (k > tiles) {
            return;
        }

        if (k == tiles) {
            tileStart[k] = count;  // sentinel row: all lists complete
            return;
        }

        const uint64_t target = static_cast<uint64_t>(k) * tileTarget;
        uint32_t lo = 0, hi = count;  // first splitter with position > target

        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;

            if (positions[static_cast<size_t>(mid) * kPositionStride + kClasses] <= target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        tileStart[k] = lo - 1;  // splitter 0 has position 0
    }

    // ---- step 4: merge each tile -------------------------------------------------------------

    struct TileInfo {
        uint32_t segment[kClasses + 1];  // tile-local start of each list's part
        uint32_t source[kClasses];       // start of each list's part in the concatenated lists
        uint32_t size;
        uint32_t outBase;
    };

    // rows[k] = the positions of tile k's first splitter, rows[tiles] the sentinel.
    __global__ void gatherTileRows(const uint32_t* positions, const uint32_t* tileStart, uint32_t tiles, uint32_t* rows) {
        const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

        if (k <= tiles) {
            const uint32_t* row = positions + static_cast<size_t>(tileStart[k]) * kPositionStride;

            for (int l = 0; l < kPositionStride; l++) {
                rows[static_cast<size_t>(k) * kPositionStride + l] = row[l];
            }
        }
    }

    // Fills the tile's layout; false for an empty tile.
    __device__ bool tileInfo(const ClassLists& layout, const uint32_t* rows, uint32_t tile, TileInfo& info) {
        const uint32_t* p0 = rows + static_cast<size_t>(tile) * kPositionStride;
        const uint32_t* p1 = p0 + kPositionStride;

        if (p0[kClasses] == p1[kClasses]) {
            return false;
        }

        uint32_t sum = 0;

        for (int l = 0; l < kClasses; l++) {
            info.segment[l] = sum;
            info.source[l] = layout.start[l] + p0[l];
            sum += p1[l] - p0[l];
        }

        info.segment[kClasses] = sum;
        info.size = sum;
        info.outBase = p0[kClasses];

        return true;
    }

    constexpr int kPairs = kClasses * (kClasses - 1) / 2;
    constexpr uint32_t kMergePerThread = 2;

    // What the merge writes for the element at output row r: its text position (the SA),
    // its index in the concatenated lists or its BWT character T[p - 1] at byte r + 1.
    // The BWT leaves byte 0 for the sentinel's row, T[n - 1], and records suffix 0's row
    // instead of writing it; the caller drops that byte.
    enum class MergeOutput { Positions, Indices, Bwt };

    __device__ __forceinline__ uint8_t characterBefore(const MergeInput& in, uint32_t p) {
        return p == 0 ? 0 : reinterpret_cast<const uint8_t*>(in.text)[p - 1];
    }

    // before = characterBefore(in, p), for the BWT.
    template <MergeOutput kOut>
    __device__ __forceinline__ void writeOutput(uint32_t* out, uint32_t r, uint32_t index, uint32_t p, uint8_t before, uint32_t* suffixZeroRow) {
        if constexpr (kOut == MergeOutput::Indices) {
            out[r] = index;
        } else if constexpr (kOut == MergeOutput::Positions) {
            out[r] = p;
        } else if (p == 0) {
            *suffixZeroRow = r;
        } else {
            reinterpret_cast<uint8_t*>(out)[r + 1] = before;
        }
    }

    // Shared memory of mergeShared: the records, then one 16-bit counter per record, two per word.
    __host__ __device__ constexpr size_t mergeSharedBytes(uint32_t capacity) {
        return static_cast<size_t>(capacity) * sizeof(MergeRecord) + (static_cast<size_t>(capacity) + 1) / 2 * 4;
    }

    // Writes the text positions in sorted order (the suffix array) or, for a level that orders splitters,
    // the indices into the concatenated lists. Every element's output position is its index in its own list
    // plus the number of smaller elements in each of the other 6. Those counts come from merging each of
    // the 21 pairs of sublists along a merge path: the path of a pair is cut into chunks of a few steps,
    // one per thread, each found by one diagonal binary search and then walked with one comparison per step.
    // A step that takes element A[i] counts j elements of B smaller than it, and vice versa.
    // That's about 6 comparisons per element plus the diagonal searches, where binary searches in every other
    // sublist took about 30. mergeTile() merges one tile, info filled and size <= capacity, with the whole block.
    template <MergeOutput kOut>
    __device__ void mergeTile(const MergeInput& in, const TileInfo& info, uint32_t* out, uint32_t* suffixZeroRow, MergeRecord* records, uint32_t* counts) {
        __shared__ uint32_t pairStart[kPairs + 1];  // first chunk of each pair
        __shared__ uint8_t pairList[kPairs][2];
        __shared__ uint32_t chunk;

        if (threadIdx.x == 0) {
            {
                // Steps per chunk so that the chunks of all pairs
                // (6 steps per element, plus at most one partial chunk per pair) take one round of the block's threads.
                const uint32_t steps = (kClasses - 1) * info.size;
                const uint32_t lanes = blockDim.x > kPairs ? blockDim.x - kPairs : 1;
                const uint32_t c = max(1u, (steps + lanes - 1) / lanes);
                uint32_t total = 0;
                int q = 0;

                for (int a = 0; a < kClasses; a++) {
                    for (int b = a + 1; b < kClasses; b++, q++) {
                        const uint32_t length = info.segment[a + 1] - info.segment[a] + info.segment[b + 1] -
                                                info.segment[b];
                        pairStart[q] = total;
                        pairList[q][0] = static_cast<uint8_t>(a);
                        pairList[q][1] = static_cast<uint8_t>(b);
                        // A pair with an empty side adds nothing.
                        const bool empty = info.segment[a] == info.segment[a + 1] || info.segment[b] == info.segment[b + 1];

                        total += empty ? 0 : (length + c - 1) / c;
                    }
                }

                pairStart[kPairs] = total;
                chunk = c;
            }
        }

        __syncthreads();

        const uint32_t size = info.size;
        // The BWT characters of this thread's elements, read with their records;
        // the output loop below visits the same elements.
        // The host keeps capacity <= kMergePerThread * threads.
        uint8_t before[kMergePerThread];

        for (uint32_t e = threadIdx.x, k = 0; e < size; e += blockDim.x, k++) {
            int l = 0;

            while (e >= info.segment[l + 1]) {
                l++;
            }

            const uint32_t p = in.lists[info.source[l] + e - info.segment[l]];

            loadRecord(in, p, records[e]);

            if constexpr (kOut == MergeOutput::Bwt) {
                before[k] = characterBefore(in, p);
            }
        }

        for (uint32_t w = threadIdx.x; w < (size + 1) / 2; w += blockDim.x) {
            counts[w] = 0;
        }

        __syncthreads();

        const uint32_t items = pairStart[kPairs];
        const uint32_t c = chunk;

        for (uint32_t item = threadIdx.x; item < items; item += blockDim.x) {
            int q = 0;

            while (item >= pairStart[q + 1]) {
                q++;
            }

            const uint32_t segA = info.segment[pairList[q][0]], lengthA = info.segment[pairList[q][0] + 1] - segA;
            const uint32_t segB = info.segment[pairList[q][1]], lengthB = info.segment[pairList[q][1] + 1] - segB;
            const MergeRecord* a = records + segA;
            const MergeRecord* b = records + segB;
            uint32_t d = (item - pairStart[q]) * c;
            const uint32_t end = min(d + c, lengthA + lengthB);

            // i = elements of A among the first d of the pair's merge.
            uint32_t lo = d > lengthB ? d - lengthB : 0, hi = min(d, lengthA);

            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;

                if (lessRecord(in, a[mid], b[d - 1 - mid])) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            uint32_t i = lo, j = d - lo;

            for (; d < end; d++) {
                if (j == lengthB || (i < lengthA && lessRecord(in, a[i], b[j]))) {
                    const uint32_t e = segA + i++;

                    if (j != 0) {
                        atomicAdd(&counts[e / 2], j << (16 * (e % 2)));
                    }
                } else {
                    const uint32_t e = segB + j++;

                    if (i != 0) {
                        atomicAdd(&counts[e / 2], i << (16 * (e % 2)));
                    }
                }
            }
        }

        __syncthreads();

        for (uint32_t e = threadIdx.x, k = 0; e < size; e += blockDim.x, k++) {
            int l = 0;

            while (e >= info.segment[l + 1]) {
                l++;
            }

            const uint32_t r = e - info.segment[l] + ((counts[e / 2] >> (16 * (e % 2))) & 0xFFFF);

            writeOutput<kOut>(out, info.outBase + r, info.source[l] + e - info.segment[l], records[e].position,
                kOut == MergeOutput::Bwt ? before[k] : 0, suffixZeroRow);
        }
    }

    template <MergeOutput kOut>
    __global__ void mergeShared(MergeInput in, const uint32_t* rows, uint32_t* out, uint32_t capacity, uint32_t* overflowCount, uint32_t* overflow, uint32_t* suffixZeroRow) {
        extern __shared__ __align__(16) unsigned char sharedBytes[];
        MergeRecord* records = reinterpret_cast<MergeRecord*>(sharedBytes);
        uint32_t* counts = reinterpret_cast<uint32_t*>(records + capacity);
        __shared__ TileInfo info;
        __shared__ bool skip;

        if (threadIdx.x == 0) {
            skip = !tileInfo(in.layout, rows, blockIdx.x, info);

            if (!skip && info.size > capacity) {
                overflow[atomicAdd(overflowCount, 1)] = blockIdx.x;
                skip = true;
            }
        }

        __syncthreads();

        if (!skip) {
            mergeTile<kOut>(in, info, out, suffixZeroRow, records, counts);
        }
    }

    // Tiles over mergeShared's capacity, about 1.6% of the elements in tiles of up to about 700:
    // the same merge with a larger capacity, at fewer blocks per SM.
    // Tiles larger still count by binary searches in global memory, but that should not happen.
    template <MergeOutput kOut>
    __global__ void mergeOverflow(MergeInput in, const uint32_t* rows, uint32_t* out, uint32_t capacity, const uint32_t* overflowCount, const uint32_t* overflow, uint32_t* suffixZeroRow) {
        extern __shared__ __align__(16) unsigned char sharedBytes[];
        MergeRecord* records = reinterpret_cast<MergeRecord*>(sharedBytes);
        uint32_t* counts = reinterpret_cast<uint32_t*>(records + capacity);
        __shared__ TileInfo info;

        for (uint32_t q = blockIdx.x; q < *overflowCount; q += gridDim.x) {
            __syncthreads();

            if (threadIdx.x == 0) {
                tileInfo(in.layout, rows, overflow[q], info);
            }

            __syncthreads();

            if (info.size <= capacity) {
                mergeTile<kOut>(in, info, out, suffixZeroRow, records, counts);
                continue;
            }

            for (uint32_t e = threadIdx.x; e < info.size; e += blockDim.x) {
                int l = 0;

                while (e >= info.segment[l + 1]) {
                    l++;
                }

                const uint32_t index = info.source[l] + e - info.segment[l];
                const uint32_t p = in.lists[index];
                uint32_t r = e - info.segment[l];

                for (int o = 0; o < kClasses; o++) {
                    if (o == l) {
                        continue;
                    }

                    uint32_t lo = 0, hi = info.segment[o + 1] - info.segment[o];

                    while (lo < hi) {
                        const uint32_t mid = (lo + hi) / 2;

                        if (lessGlobal(in, in.lists[info.source[o] + mid], p)) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }

                    r += lo;
                }

                writeOutput<kOut>(out, info.outBase + r, index, p,
                    kOut == MergeOutput::Bwt ? characterBefore(in, p) : 0, suffixZeroRow);
            }
        }
    }

    // ---- host --------------------------------------------------------------------------------

    struct MergeParams {
        uint32_t g = 128;                 // splitter spacing
        uint32_t tileTarget = 256;        // elements per tile
        uint32_t capacity = 480;          // records per tile in shared memory: 23.4 KiB, four blocks per SM
        uint32_t threads = 256;
        uint32_t overflowCapacity = 960;  // the same for tiles over capacity, two blocks per SM
        uint32_t overflowThreads = 512;
        int levels = 1;                   // levels of splitter merging below the top one
    };

    // One level of the merge.
    // With a child, the splitters are ordered by merging the splitter lists; without, by CUB's merge sort.
    // plan() sizes everything without allocating, so the caller can check the memory first.
    class Merger {
     public:
        void plan(const ClassLists& layout, const MergeParams& params, int depth) {
            m_layout = layout;
            m_params = params;
            m_total = layout.start[kClasses];

            for (int l = 0; l < kClasses; l++) {
                m_splitters.start[l + 1] =
                    m_splitters.start[l] + (layout.start[l + 1] - layout.start[l] + params.g - 1) / params.g;
            }

            m_count = m_splitters.start[kClasses];
            m_tiles = (m_total + params.tileTarget - 1) / params.tileTarget;

            if (depth > 0) {
                m_child = std::make_unique<Merger>();
                m_child->plan(m_splitters, params, depth - 1);
            }
        }

        cudaError_t queryTemp() {
            size_t scan = 0, sort = 0;

            GPUSA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, scan,
                cuda::transform_iterator(static_cast<const uint32_t*>(nullptr), InList{m_layout, 0}),
                cuda::strided_iterator(static_cast<uint32_t*>(nullptr), kPositionStride), m_count));

            if (!m_child) {
                GPUSA_TRY(cub::DeviceMergeSort::SortKeys(
                    nullptr, sort, static_cast<uint32_t*>(nullptr), m_count, SplitterLess{}));
            } else {
                GPUSA_TRY(m_child->queryTemp());
            }

            m_tempBytes = std::max(scan, sort);

            return cudaSuccess;
        }

        // Words of the arrays steps 1 to 3 use, which may live in the output buffer.
        size_t scratchWords() const {
            const size_t count = m_count;
            return count                                // sorted
                   + (count + 1) * kPositionStride      // positions
                   + static_cast<size_t>(m_tiles) + 1   // tile starts
                   + (m_child ? count * 2 : 0);         // child lists, child output
        }

        // Device bytes of this level beyond its scratch, plus everything of the levels below; valid after queryTemp().
        int64_t bytes() const {
            int64_t total = (static_cast<int64_t>(m_tiles) + 1) * (kPositionStride + 1) * 4  // rows, overflow
                            + 8 + static_cast<int64_t>(std::max<size_t>(m_tempBytes, 1));

            if (m_child) {
                total += static_cast<int64_t>(m_child->scratchWords()) * 4 + m_child->bytes();
            }

            return total;
        }

        // scratch: scratchWords() words that nothing else uses until run() writes the output, or null to allocate them.
        cudaError_t allocate(DeviceMemory& memory, uint32_t* scratch) {
            if (!scratch) {
                GPUSA_TRY(memory.allocate(m_ownScratch, scratchWords()));
                scratch = m_ownScratch;
            }

            m_sorted = scratch;
            m_positions = m_sorted + m_count;
            m_tileStart = m_positions + (static_cast<size_t>(m_count) + 1) * kPositionStride;
            m_childLists = m_child ? m_tileStart + m_tiles + 1 : nullptr;
            m_childSorted = m_child ? m_childLists + m_count : nullptr;

            GPUSA_TRY(memory.allocate(m_rows, (static_cast<size_t>(m_tiles) + 1) * kPositionStride));
            GPUSA_TRY(memory.allocate(m_overflow, static_cast<size_t>(m_tiles) + 1));
            GPUSA_TRY(memory.allocate(m_overflowCount, 2));  // and suffix 0's row
            GPUSA_TRY(memory.allocate(m_temp, m_tempBytes));

            return m_child ? m_child->allocate(memory, nullptr) : cudaSuccess;
        }

        void release(DeviceMemory& memory) {
            if (m_child) {
                m_child->release(memory);
            }

            memory.release(m_ownScratch);
            memory.release(m_rows);
            memory.release(m_overflow);
            memory.release(m_overflowCount);
            memory.release(m_temp);
        }

        // Merges the lists in.lists (laid out as planned) into out.
        // For the BWT, suffixZeroRow() then holds suffix 0's row.
        cudaError_t run(MergeInput in, uint32_t* out, MergeOutput what) {
            const MergeParams& p = m_params;

            in.layout = m_layout;

            if (p.capacity > kMergePerThread * p.threads || p.overflowCapacity > kMergePerThread * p.overflowThreads) {
                return cudaErrorInvalidValue;
            }

            if (m_child) {
                MergeInput childInput = in;

                childInput.lists = m_childLists;
                gatherSplitterPositions<<<blocksFor(m_count), kThreads>>>(in, m_splitters, p.g, m_count, m_childLists);
                GPUSA_TRY(m_child->run(childInput, m_childSorted, MergeOutput::Indices));
                childToParent<<<blocksFor(m_count), kThreads>>>(m_layout, m_splitters, p.g, m_childSorted, m_count, m_sorted);
            } else {
                size_t bytes = m_tempBytes;

                pickSplitters<<<blocksFor(m_count), kThreads>>>(m_layout, m_splitters, p.g, m_count, m_sorted);
                GPUSA_TRY(cub::DeviceMergeSort::SortKeys(m_temp, bytes, m_sorted, m_count, SplitterLess{in}));
            }

            // Each list's splitters before each splitter, into the rows; then the searches.
            for (int l = 0; l < kClasses; l++) {
                size_t bytes = m_tempBytes;

                GPUSA_TRY(cub::DeviceScan::ExclusiveSum(m_temp, bytes,
                    cuda::transform_iterator(static_cast<const uint32_t*>(m_sorted), InList{m_layout, l}),
                    cuda::strided_iterator(m_positions + l, kPositionStride), m_count));
            }

            searchLists<<<blocksFor(m_count), kThreads>>>(in, m_sorted, m_positions, m_count, p.g);

            // The sentinel row: every list complete.
            uint32_t sentinel[kPositionStride] = {};

            for (int l = 0; l < kClasses; l++) {
                sentinel[l] = m_layout.start[l + 1] - m_layout.start[l];
            }

            sentinel[kClasses] = m_total;
            GPUSA_TRY(cudaMemcpy(m_positions + static_cast<size_t>(m_count) * kPositionStride, sentinel, sizeof sentinel,
                cudaMemcpyHostToDevice));

            sumPositions<<<blocksFor(m_count), kThreads>>>(m_positions, m_count);
            cutTiles<<<blocksFor(
                static_cast<uint64_t>(m_tiles) + 1), kThreads>>>(m_positions, m_count, m_tileStart, m_tiles, p.tileTarget);
            gatherTileRows<<<blocksFor(
                static_cast<uint64_t>(m_tiles) + 1), kThreads>>>(m_positions, m_tileStart, m_tiles, m_rows);
            GPUSA_TRY(cudaMemsetAsync(m_overflowCount, 0, sizeof(uint32_t)));

            // From here on, out may overlap the scratch.
            const size_t shared = mergeSharedBytes(p.capacity);
            int sms = 0;

            GPUSA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

            switch (what) {
                case MergeOutput::Positions:
                    GPUSA_TRY(launchMerge<MergeOutput::Positions>(in, out, shared, sms));
                    break;
                case MergeOutput::Indices:
                    GPUSA_TRY(launchMerge<MergeOutput::Indices>(in, out, shared, sms));
                    break;
                case MergeOutput::Bwt:
                    GPUSA_TRY(launchMerge<MergeOutput::Bwt>(in, out, shared, sms));
                    break;
            }

            return cudaGetLastError();
        }

        const uint32_t* suffixZeroRow() const {
            return m_overflowCount + 1;
        }

     private:
        template <MergeOutput kOut>
        cudaError_t launchMerge(const MergeInput& in, uint32_t* out, size_t shared, int sms) {
            const size_t overflowShared = mergeSharedBytes(m_params.overflowCapacity);

            GPUSA_TRY(cudaFuncSetAttribute(
                mergeShared<kOut>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
            GPUSA_TRY(cudaFuncSetAttribute(
                mergeOverflow<kOut>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(overflowShared)));
            mergeShared<kOut><<<m_tiles, m_params.threads, shared>>>(
                in, m_rows, out, m_params.capacity, m_overflowCount, m_overflow, m_overflowCount + 1);
            mergeOverflow<kOut><<<sms * 2, m_params.overflowThreads, overflowShared>>>(
                in, m_rows, out, m_params.overflowCapacity, m_overflowCount, m_overflow, m_overflowCount + 1);

            return cudaSuccess;
        }

        ClassLists m_layout{};
        ClassLists m_splitters{};
        MergeParams m_params;
        uint32_t m_total = 0;
        uint32_t m_count = 0;
        uint32_t m_tiles = 0;
        size_t m_tempBytes = 0;
        std::unique_ptr<Merger> m_child;
        uint32_t* m_ownScratch = nullptr;
        uint32_t* m_sorted = nullptr;
        uint32_t* m_positions = nullptr;
        uint32_t* m_tileStart = nullptr;
        uint32_t* m_childLists = nullptr;
        uint32_t* m_childSorted = nullptr;
        uint32_t* m_rows = nullptr;
        uint32_t* m_overflow = nullptr;
        uint32_t* m_overflowCount = nullptr;
        uint8_t* m_temp = nullptr;
    };
}
