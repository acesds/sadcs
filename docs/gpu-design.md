# GPU variant: design and measurements

The [README](../README.md#gpu-variant) describes what the GPU variant does and how fast it is. This document records why it is built the way it is: the design decisions, the measurements behind them, the tuning steps and what each gained, and what was tried and dropped. It replaces the development plan (`docs/cuda-plan.md` until 2026-11-13); the plan's day-by-day log wasn't kept.

All measurements are on one machine: GeForce RTX 5070 Ti (Blackwell, sm_120, 70 SMs, 48 MiB L2, 16 GiB GDDR7, about 15 GiB free with the display attached), CUDA 13.5, Windows 11, Ryzen 7 5800X on PCIe 4.0 x16. The test corpora are linked in the [README](../README.md#test-corpora).

## Goal and outcome

The goal was a GPU suffix array construction on the same difference cover idea as the CPU variant, verified by the same checker, for inputs larger than [libcubwt](https://github.com/IlyaGrebnov/libcubwt) can handle, ideally at libcubwt's speed. Both parts are met:

- **Size.** libcubwt 1.6.3 (BWT only) needs 20.5 bytes of device memory per input byte and 1.0.0 (the last version with suffix arrays) about 30, so on this card they stop at 760 MB and 520 MB. Past that, the Windows driver doesn't fail the allocation but pages to host memory, and libcubwt 1.6.3 gets 4 times slower. sadcs needs 10.02 bytes per input byte and builds inputs up to about 1.547 GB.
- **Speed.** On this card, the SA is faster than libcubwt 1.0.0's and the BWT faster than 1.6.3's on every input both can build (table in the README). libcubwt is tuned for other GPUs, so this may not hold elsewhere. After the merge prototype, the verdict on speed was no-go (the merge alone looked like 62 ms on enwik8, against libcubwt's 93 ms for everything); the tuning below turned it around.

Not goals: the CPU variant's 5n footprint, and inputs that don't fit in device memory (there's no out-of-core mode).

## Why not port the CPU code

- **Induction.** The two induction scans have a strict order. The CPU variant runs them block by block with a sync per block, and falls back to one thread on single-character runs. On a GPU that becomes thousands of kernel launches or grid syncs with little work each.
- **Task-parallel multikey quicksort.** Recursive, data-dependent tasks with uneven partitions don't map to warps. Radix sorts over flat key arrays do.
- **Larsson–Sadakane in-round refinement.** It's what makes the CPU doubling fast, but it's inherently sequential. The GPU uses plain prefix doubling with compaction of the unresolved groups instead.
- **A hybrid** that sorts on the GPU and induces on the CPU was not pursued: the CPU induction plus a PCIe round trip alone takes longer than libcubwt's whole run.

What carries over is the difference cover: after a bounded number of characters, any two suffixes compare in O(1) by sample ranks. On the GPU this bounds both the character passes and the merge's comparisons.

## Choice of cover

On the GPU the tradeoff shifts towards small covers. A larger sample is cheap: just more radix-sorted elements. Long character comparisons are expensive, because the threads of a warp diverge on them. v = 32 has a minimum cover of 7 residues, D = {0, 1, 3, 8, 12, 19, 25}, a 21.9% sample. The distances d to the next sample position run from 0 to 6 (mean 2.31), so there are 7 classes: of the 25 non-sample residues, 6 have d = 1, 5 have d = 2, 5 have d = 3, 4 have d = 4, 3 have d = 5 and 2 have d = 6.

The number of classes is the cover's largest gap, and the merge compares each element once per other list, so the gap matters more than v. An exhaustive search (element 0 fixed) shows that D above is the best cover modulo 32: no 7-element cover has a largest gap below 7, and none of those has a smaller mean distance. Modulo 64 the minimum cover has 9 elements, but its largest gap is at least 15 (the CPU's cover has 18); larger covers bring it down to 10 with 10 elements, 8 with 11, 7 with 12 and 6 with 13 (a 20.3% sample, fewer classes than v = 32).

All of these were built and measured (`bench --context --pinned`, best of 10 runs over 2 interleaved rounds; each passes the selftest and `test --gpu`). Total time in ms for the suffix array:

| cover | sample | classes | Silesia | Gauntlet | enwik8 | sources | dna | proteins |
|---|---|---|---|---|---|---|---|---|
| 32, 7 elements | 21.9% | 7 | **212** | **144** | **94** | **233** | **418** | **1558** |
| 64, 13 | 20.3% | 6 | 325 | 210 | 137 | 354 | 617 | 2444 |
| 64, 12 | 18.8% | 7 | 325 | 213 | 136 | 350 | 609 | 2399 |
| 64, 11 | 17.2% | 8 | 304 | 201 | 122 | 316 | 549 | 2143 |
| 64, 9 | 14.1% | 15 | 390 | 253 | 153 | 391 | 676 | 2578 |
| 64, 9 (the CPU's) | 14.1% | 18 | 459 | 292 | 177 | 457 | 775 | 3013 |

Every cover modulo 64 is 30% to 117% slower, because the merge takes 1.8 to 6 times as long: comparisons run up to 63 characters instead of 31, so more of them read past the 16 characters a record caches, and a record carries 9 to 13 sample ranks instead of 7 (64 or 80 bytes instead of 48). Scaling the tiles down to keep four blocks per SM changed the totals by −3% to +5%. The smaller samples do rank faster (proteins: 513 ms with 9 elements against 661 ms), but not by enough. The peak hardly changes (9.83 to 9.94 bytes per input byte against 10.02), since it is set by the merge's n-sized arrays, not by the sample. So v = 64 brings no advantage on the GPU, and the cover stays at 32.

The cover's tables (`src/common/differencecover.h`) are `constexpr`, shared with the CPU, and copied to `__constant__` memory. Ranks of sample suffixes are stored compactly at `(i / 32) * 7 + slot[i mod 32]`.

## How it works

The steps are numbered as in the source comments.

### Step 1: upload

The text goes to the device padded with zero words, so 8-byte loads past the end are safe and read zeros. Sample positions are computed arithmetically, never selected.

### Step 2: rank the sample (`samplerank.cuh`)

A list of the unresolved sample suffixes, each with a group id: the position of its group's first element in the final order. The ids live in the rank array for all sample positions, so they are always valid ranks of the groups found so far. Each pass sorts the list by a subkey (stably within the groups), marks where a new group starts, and gives it its old group's id plus its offset in the old group, with max-scans. Groups of one are final and leave the list (`DeviceSelect::Flagged`), and the host reads the count of what's left: one sync per pass.

- **The first pass** covers 15 characters with two LSD sorts on 64-bit keys: characters 8 to 14 with min(n − p, 15) in 4 bits (60 bits), then stably characters 0 to 7 (64 bits, zero past the end, no tag). Zero padding can tie a shorter suffix with a longer one but never order it after it, and the minor key's tag then puts the shorter first. The tag must be capped at the characters covered: a 12-character variant capped at 15 split a suffix with 13 characters left from an equal longer one in the wrong order (the selftest caught it). The first pass needs no group sort, so it reads the sorted major keys from the sort's output, and `markWideGroups` reads the minor key from the text only where the major key ties with a neighbour.
- **Three character passes** sort by the next 7 characters plus a length tag (64 bits), then stably by group id (32 bits). `markNewGroups` reads each element's subkey once and takes its predecessor's with `__shfl_up_sync`; only lane 0 reads a second one. After 36 characters every group shares at least v.
- **Doubling rounds** with h = 32, 64, 128, …: the subkey is rank[i + h] + 1, at most m, so it's sorted as a 32-bit key over bit_width(m) bits. Each round otherwise has the shape of a character pass.
- **Grouped rounds.** A pass leaves the list in group order, and the compaction keeps it. So a pass before a doubling round on at least 2^20 elements also compacts the group ids and the groups' start offsets, read back with the remaining count. If no group is over 128 elements, the round sorts each group by its subkey with `DeviceSegmentedSort::StableSortPairs`, replacing the subkey sort, the gather of the group ids and the group sort. The ids stay valid because a segment has one id.
- **Partitioned stores.** Writing the ids to the rank array is a scatter to random addresses, one DRAM sector per 4-byte write. For lists of 4M elements and more (`kPartitionThreshold`), one radix pass on the top 8 bits of the position orders the writes first, so each bucket's writes fall into a window of about 2 MB that the L2 combines. The list's order doesn't matter then, because the next pass sorts it completely; the unresolved flag rides along in bit 31 of the id. Such passes don't keep the group order, so the next round can't be a grouped one.

Memory for m = 7n/32 sample suffixes: text n, ranks 4m, the position list double-buffered 8m, the sort keys double-buffered 16m (which also hold the group ids, scan results and flags), plus CUB's scratch: 7.16 bytes per input byte.

### Step 3: sort the classes (`classsort.cuh`)

For i in class d ≥ 1, i + 1 is in class d − 1, so (T[i], suffix i + 1) orders class d, and class d − 1 is already sorted. Each class comes from the one before it: walk class d − 1 in order (with n prepended, which yields position n − 1, a prefix of every other suffix starting with T[n − 1]), keep the positions whose predecessor is in class d (a flag, an exclusive sum and a gather), and sort the predecessors stably by one character: a single 8-bit radix pass. Class 1 comes from the sample's order, scattered from its ranks. The key is 32 bits, T[q] down to T[q − 3], sorted on its top byte only; the next class's key is the previous one shifted left a byte, so classes 2 to 4 and 6 read no text, and classes 1 and 5 read 4 characters at random. The flags are scanned in place.

The 7 sorted lists go into one array of n positions, the merge's input. The sort's scratch (at most 4 · largest class + m + 1 words) lives in the merge's output buffer, which isn't needed yet.

### Step 4: merge the 7 lists (`merge.cuh`)

Suffixes i and j compare by k = `cTieOffset[i mod 32][j mod 32]` < 32, where i + k and j + k are both sampled: T[i .. i + k) against T[j .. j + k), then rank[i + k] against rank[j + k].

1. **Splitters**: every G-th element of every list, G = 128. The splitters of each list are again sorted lists, so they're ordered by this same merge one level down, and only that level's splitters go to CUB's merge sort with the comparator. A third level saved only 0.2 ms on enwik8.
2. **Positions in every list.** For splitter s and list l, the splitters of l before s confine s to a window of G − 1 elements of l. Seven exclusive sums, one per list, count them straight into the splitter's row of the position table (a `cuda::transform_iterator` flags the list, a `cuda::strided_iterator` writes the row). Then one thread per splitter binary-searches all 6 other lists with the splitter's record in registers. The sum of a row is the splitter's output position.
3. **Tiles.** The last splitter at or before every T-th output position (T = 256) starts a tile. Steps 1 to 3 work in the output buffer, which step 4 only writes afterwards; only the rows at tile boundaries are copied out.
4. **Merge each tile in shared memory.** One block of 256 threads loads a 48-byte record per element: the first 16 characters, the ranks of the 7 sample positions in [i, i + 32), and the position. Comparisons that reach characters 16 to 31 read them from global memory; that's 5 to 15% of them (see below). Each of the 21 pairs of sublists is merged along a merge path: each thread takes a chunk of ⌈6 · size / (threads − 21)⌉ steps, finds its start by one diagonal search, and walks it with one comparison per step. A step that takes A[i] adds the j elements of B before it to A[i]'s count, and vice versa, with 16-bit counters, two to a word, added by shared-memory atomics. An element's output position is the tile's base plus its index in its own list plus its counts. At a capacity of 480 records the block takes 23.4 KiB, so four blocks fit per SM.
5. **Oversized tiles** (over 480 elements: 6.4% of the tiles and 13.8% of the elements on the 500 MB prefix, all with 384 to 1023 elements) go to `mergeOverflow`, the same tile code with room for 960 records, 512 threads and two blocks per SM, looping over the overflow list. Only tiles over 960 count by binary searches in global memory.

For the BWT, the merge writes T[p − 1] to byte r + 1 of the output instead of p to word r, and records suffix 0's row instead. The host sets byte 0 to T[n − 1], and downloads the bytes before and after suffix 0's; its row plus one is the primary index.

How far comparisons read, by where they end:

| input | word 0 | word 1 | word 2 | word 3 | ranks, k ≤ 16 | ranks, k > 16 |
|---|---|---|---|---|---|---|
| enwik8 | 34.3% | 30.2% | 2.7% | 0.2% | 30.1% | 2.6% |
| webster | 27.8% | 22.4% | 4.2% | 0.9% | 35.5% | 9.2% |
| 500 MB prefix | 22.2% | 30.6% | 4.2% | 0.4% | 36.3% | 6.4% |

Nothing in the record can go structurally: every residue needs all 7 ranks and 25 to 31 characters for some other residue.

### Step 5: download

The suffix array (4n bytes) or the BWT (n bytes). `sadcs test --gpu` builds on the CPU too, compares, and runs the checker on the GPU's result.

## Memory

Tracked allocation by allocation, in bytes per input byte. It's the same on every input, because nothing depends on the data:

| phase | peak | what's allocated |
|---|---|---|
| sample ranking | 7.16 | text 1, ranks 0.875, positions and keys 5.25 |
| class sorts | 9.91 | text, ranks, lists 4, output 4 (as sort scratch) |
| merge | 10.02 | text, ranks, lists, output, tile rows and overflow list 0.15 |

Small inputs peak a little higher (10.23 summed over the Calgary corpus) because of fixed-size buffers.

`gpusa.cu` plans the peak before allocating anything and refuses an input when the peak plus a 256 MiB reserve exceeds the free device memory. That check matters on Windows: the driver's sysmem fallback lets `cudaMalloc` succeed past the end of VRAM and silently pages to host memory.

**Size limit**, tested with nothing else on the GPU (15 037 MiB free):

- enwik9 followed by its first 545 MB (1.545 GB, with repeats up to 545 MB long, a hard case for the doubling) peaks at 14 763 MiB. The SA (5.9 s) passes `sadcs check`, the BWT (5.7 s) `sadcs check --bwt`.
- enwik9 plus 550 MB (1.55 GB) is refused before anything is allocated: it needs 14 811 MiB plus the reserve.

So the limit on this card is (15 037 − 256) MiB / 10.02 ≈ 1.547 GB.

## Transfers and the context

`GpuContext` (`gpuCreateContext(capacity)`, `gpuDestroyContext()`, and `gpuSuffixArray()`/`gpuBwt()` overloads taking one) keeps the device memory across calls, like libcubwt's `libcubwt_allocate_device_storage`. It's a CUDA memory pool with an unlimited release threshold, reserved once by allocating and freeing the planned peak; allocations then come from it with `cudaMallocFromPoolAsync` and go back with `cudaFreeAsync`, both nearly free. The reservation is the peak rounded up to 32 MiB chunks and doesn't grow over repeated runs. Calls without a context create a temporary one.

Host transfers go through two 8 MiB pinned chunks in the context, double-buffered, unless the caller's buffer is pinned already (`cudaPointerGetAttributes`), which is then copied directly. Pinning the whole host buffers instead costs more: `cudaHostRegister` returns quickly, but the driver maps the pages at the next synchronous call, about 36 ms per GB, and unregistering takes 11 ms per GB. Measured on their own:

| transfer | 400 MiB | 4000 MiB |
|---|---|---|
| register, copy, unregister, download | 33 ms | 324 ms |
| staged through 2 × 8 MiB pinned buffers, 1 copy thread | 27 ms | 270 ms |
| the same with 2, 4 or 8 copy threads | 30 to 35 ms | 300 to 357 ms |
| pageable `cudaMemcpy`, download | 29 ms | 287 to 503 ms |

The CPU copy and the DMA share the memory bandwidth, so staging is bound at about 15 GB/s, and more copy threads make it slower. Pinned transfers of 256 MiB and up reach 28.4 GB/s up and 28.6 GB/s down.

`sadcs bench --context --pinned` measures the way libcubwt's benchmark does: the context and pinned host buffers are allocated before the runs. Without them (a one-shot call from pageable memory), enwik8's SA takes about 35 ms longer.

## Measurements

### Primitives

Measured before the design was settled, to calibrate a cost model:

| primitive | measured |
|---|---|
| device copy, 1 GiB (read + write) | 768 GB/s, 86% of the nominal 896 |
| CUB `SortPairs`, 32-bit keys, 32-bit values | 0.098 ns per pair (4 passes), 690 GB/s |
| CUB `SortPairs`, 64-bit keys, 32-bit values | 0.287 ns per pair (8 passes), 700 GB/s |
| CUB `SortPairs`, 64-bit keys restricted to 40 or 48 bits | 0.036 ns per pair per 8-bit pass, as at full width |
| random 4-byte gather, source of 256 MiB to 4 GiB | 8.3 to 9.3 G loads/s |
| random 4-byte gather, 16 MiB source (fits the L2) | 46 G loads/s |
| random 8-byte gather, source of 1 GiB or more | 7.7 to 8.0 G loads/s |
| 8 text bytes at a random offset, 1 GiB text | 6.9 G keys/s |
| the same, positions in text order | 50 G keys/s |

Sort bandwidth is flat from 16M to 256M elements, and the cost per pass depends only on the number of 8-bit digits, not on the key width. A random load costs about as much as streaming 90 bytes, which is why so much of the tuning below is about turning random reads and writes into ordered ones.

### Where the time goes now

Time by phase, in ms, with `bench --context --pinned`:

| input | upload | rank sample | sort classes | merge | download | total |
|---|---|---|---|---|---|---|
| enwik8, SA | 3.8 | 25.8 | 11.0 | 38.7 | 14.5 | 94.0 |
| enwik8, BWT | 3.8 | 26.1 | 11.0 | 41.2 | 3.9 | 86.1 |
| Gauntlet, SA | 3.3 | 94.3 | 6.5 | 29.1 | 11.1 | 145.2 |
| enwik9, first 500 MB, SA | 18.8 | 167.4 | 58.6 | 217.8 | 70.3 | 533.1 |
| enwik9, first 500 MB, BWT | 18.8 | 168.2 | 58.2 | 228.7 | 18.0 | 491.9 |
| Pizza&Chili sources, SA | 7.9 | 83.7 | 23.5 | 86.8 | 29.9 | 231.9 |
| Pizza&Chili dna, SA | 15.2 | 120.2 | 47.1 | 175.2 | 56.9 | 414.7 |
| Pizza&Chili proteins, SA | 44.4 | 656.5 | 138.5 | 541.0 | 165.6 | 1546.2 |

The BWT is faster than the suffix array mostly because a quarter as much comes back over PCIe. The merge reads about four random 32-byte sectors of text and ranks per element, and that is most of its time. The radix sorts are bound by memory bandwidth. With 16 GB of VRAM, inputs up to about 1.547 GB build and pass the checker (enwik9 followed by its first 545 MB, 5.9 s for the suffix array and 5.7 s for the BWT).

Kernel time per run on the 500 MB enwik9 prefix (SA, 533 ms in all including transfers), measured before the last tuning step, which took about 2 ms off `markNewGroups`:

| kernel | ms | |
|---|---|---|
| `mergeShared` | 154 | mostly loading records: about 4 random 32-byte sectors per element |
| radix sorts, 64-bit keys | 79 | the first ranking pass and the character passes; bandwidth-bound |
| `mergeOverflow` | 35 | the oversized tiles, about 1.4 times `mergeShared`'s time per element |
| radix sorts, 32-bit keys | 26 | group sorts, doubling rounds, class sorts |
| `gatherPredecessors` | 25 | one random text read per class element |
| `searchLists` | 24 | both levels |
| `markNewGroups` | 23 | subkeys read at random, 12 ms of it in the first pass |
| `makeWideKeys` | 15 | the first pass's keys, the second in sorted order |

The ranking (167 ms) splits into the first pass (94 ms, of which 61 ms of radix sorts), the second (34 ms) and the doubling rounds (about 30 ms). On Gauntlet the ranking is 95 of 146 ms, mostly doubling rounds with a fixed cost of about 0.5 to 0.75 ms each. On Pizza&Chili proteins (1.18 GB, too large for either libcubwt version on this card) it is 657 of 1546 ms, a larger share than on any other input except Gauntlet.

### Tuning steps

Each step was compared with the build before it in interleaved runs. Times are as libcubwt measures (except the first two rows, before `--context --pinned` existed), best of 5.

| step | what changed | effect |
|---|---|---|
| first full build | | enwik8 SA 197 ms one-shot, 500 MB prefix 1113 ms |
| ranking passes | subkeys passed on by warp shuffle; the first pass reuses its sorted keys and covers 14 characters with two sorts | enwik8 ranking 51 → 42 ms, enwik9 806 → 656 ms |
| context and staging | memory pool kept across calls, staged transfers, `bench --context --pinned` | enwik8 SA 188 → 179 ms one-shot, 145 ms measured like libcubwt |
| merge path | 21 pairwise merge paths with counters instead of 6 binary searches per element | merge 23 to 27% faster; enwik8 SA 146 → 129 ms |
| classes from the previous class | one 8-bit pass per class instead of up to 10 | class sorts on enwik8 28 → 17 ms |
| BWT from the merge | T[p − 1] written directly, no gather and pack passes | enwik8 BWT 116 → 110 ms |
| partitioned scatters | rank and sample writes ordered by a radix pass first | with the two above: enwik8 SA 129 → 109 ms, 500 MB prefix 756 → 606 ms |
| splitter spacing 96 | | 500 MB prefix merge 266 → 257 ms |
| characters in the class keys | 4 characters carried from class to class | class sorts on the 500 MB prefix 75 → 59 ms |
| 48-byte records | 16 instead of 32 characters, four blocks per SM instead of three | merge on enwik8 50 → 44 ms, on the 500 MB prefix 258 → 244 ms |
| oversized tiles in shared memory | `mergeOverflow` with room for 960 | 500 MB prefix merge 244 → 230 ms |
| splitter spacing 128, tiles of 256 | | 500 MB prefix merge 229 → 225 ms, peak 10.07n → 10.02n |
| 15 characters in the first pass | 4-bit tag instead of a length byte | 500 MB prefix ranking 176 → 171 ms |
| 32-bit doubling subkeys | | Gauntlet ranking 111 → 103 ms |
| segmented doubling rounds | | Gauntlet ranking 102 → 97.5 ms |
| one splitter search kernel | all lists per thread, the splitter's record in registers | 500 MB prefix merge 224.5 → 217.5 ms |
| tail keys only on ties | | 500 MB prefix ranking 169 → 167 ms |
| now | | enwik8 SA 94 ms, 500 MB prefix 533 ms |

### Tried and dropped

- **A pairwise merge** (3 rounds of merging classes in pairs): a cost model put it at 85 ms on enwik8, more than everything else together, because every comparison does random reads. The multiway merge replaced it before it was built.
- **Binary searches instead of merge paths** in the tiles, as the prototype had them: about 30 comparisons per element instead of 6. In the prototype, with 64-byte records, the merge path was no faster, because loading the records dominated; with the records in place and the loads tuned, it gained 23 to 27%.
- **Cover tables in shared memory** instead of `__constant__` (the threads of a warp look up different entries, which constant memory serializes): slower, because the extra 2 KiB per block cost an SM a block. Occupancy, set by the shared memory per block, mattered more than the lookups throughout.
- **Merge parameters.** 64, 128 and 192 threads per block are slower than 256. Capacity 368 or 384 (five blocks per SM) and 656 (three, larger tiles) were no better than 480. Tiles of 160 were a little faster but raised the peak to 10.11n. G = 256 is much slower (283 against 224 ms on the 500 MB prefix). An overflow capacity of 768 with 384 threads saved 2 ms on the 500 MB prefix but cost 1 ms on Silesia and Gauntlet; a capacity of 640 in either kernel was slower.
- **A narrower first pass** (8 characters, then 4 with a tag: 12 characters in 13 radix passes instead of 14 in 16): slower, because two characters fewer leave more suffixes tied for the second pass. 14 characters in 15 passes was as fast as 15 in 16.
- **Larger segments in the grouped rounds.** CUB sorts segments above its medium size with one block each over all 32 bits: 2.8 ms for 13 600 segments of up to 256 elements on fib_s14930352, against 0.75 ms for a radix round. A cap of 1024 or 4096 made fib and fss10 slower overall, and below 2^20 elements the extra kernels cost more than they save.
- **Cached heads for the splitter search.** A table of the first 8 characters of every 16th list element let the search narrow each window by comparing heads. The search went only from 24.3 to 22.7 ms on the 500 MB prefix, and building the table took 4.7 ms. A splitter is compared against its neighbours in sorted order, which usually share more than 8 characters with it, so the heads tie.
- **More copy threads for staging**: slower, see above.

## Testing

- `sadcs selftest` compares GPU and CPU suffix arrays and BWTs, and the GPU's sample ranks with those a CPU-built suffix array implies, on adversarial inputs: all lengths 0 to 80 as runs and random binary text, periods 1 to 1000, Fibonacci, Thue–Morse, random text over 1 to 256 symbols, and 4 MB runs, periods, Fibonacci, Thue–Morse and random text. A context section reuses one context from a 20 MB input down to empty ones (which also covers the partitioned stores) and checks that an over-capacity input is refused while the context stays usable.
- `sadcs test <dir> --gpu` and `--gpu --bwt` on Silesia, Gauntlet, enwik8 and enwik9: the GPU's output matches the CPU's and passes the checker.
- Paths that the corpora rarely reach were checked by forcing them in a test build: the overflow kernel and the global-memory fallback (capacities of 64 and 32), and the grouped doubling rounds for every list size.

## Development notes

- **Measuring.** Run `sadcs bench <in|dir> --context --pinned` (and `--bwt`) with the GPU otherwise idle. GPU timings drift by about 5% over a session, as much as most single changes, so compare two builds in interleaved runs, not against numbers from earlier. Nsight Systems (`nsys profile`, then `nsys stats --report cuda_gpu_kern_sum` or `cuda_gpu_trace`) shows the time per kernel and per pass.
- **Checking.** After every change: `sadcs selftest`, `sadcs test <dir> --gpu` and `--gpu --bwt` on the corpora, and the peak column of `bench` (10.02 bytes per input byte).
- **libcubwt.** `bench/libcubwt/bench_cubwt.cu` times libcubwt end to end, host to host, with the device storage and pinned buffers allocated beforehand. Build it against libcubwt 1.6.3 for the BWT, or against 1.0.0 with `-DCUBWT_SA` for the suffix array, for example `nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler "/O2 /EHsc /Zc:preprocessor" -I libcubwt bench_cubwt.cu libcubwt/libcubwt.cu`. 1.6.3 builds unmodified with CUDA 13.5; `libcubwt-1.0.0-cuda13.patch` gives 1.0.0 the two changes 1.6.1 and 1.6.2 made (`sm_120` in the architecture list, and `cub::Sum/Max/Min` replaced for CCCL 3).
- **Build.** CUDA is optional in CMake (`check_language(CUDA)`, off with `-DSADCS_CUDA=OFF`). On Windows nvcc uses `cl.exe` as host compiler while the rest keeps clang-cl, which needs the Ninja `cuda` preset: CMake 3.30 and 3.31 can't identify the compiler through the CUDA 13.5 Visual Studio integration (MSB4023 in `AddCudaCompileMetadata`). Under Ninja, clang-cl ignores `/GL`, so that build has no link-time code generation. CCCL 3 refuses cl.exe's traditional preprocessor, hence `-Xcompiler /Zc:preprocessor`. The `.cu` files must not depend on OpenMP. The phases (`samplerank.cuh`, `classsort.cuh`, `merge.cuh`, sharing `devicecommon.cuh`) are one translation unit, `gpusa.cu`.

## Open questions

- **Oversized tiles**: 13.8% of the elements on large text go through the slower `mergeOverflow`. Cutting tiles more evenly, between splitters, might move most of them back.
- **Fixed costs per pass**: small inputs and repetitive ones pay about 0.3 to 0.75 ms per ranking pass in launches and the host sync. A CUDA graph, or finishing small lists in a single block, could cut that.
- **Overlap**: could the sample ranking and the class sorts overlap on separate streams, the way the CPU variant overlaps the sample chain with the RMS sort?
