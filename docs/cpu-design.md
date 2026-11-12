# CPU variant: design and measurements

The [README](../README.md#cpu-variant) gives an overview of the CPU variant. This document describes how it works in detail and has the measurements: the phases, the choice of cover, thread scaling against libsais and libdivsufsort, and where libsais stays faster. The GPU variant has its own [design record](gpu-design.md).

All measurements are on a Ryzen 7 5800X (8 cores, 16 threads), Windows 11, clang-cl 22, AVX2. The test corpora (Silesia, enwik8 and enwik9, Gauntlet, Calgary, Pizza&Chili) are linked in the [README](../README.md#test-corpora).

## The approach

sadcs is not a new algorithm, just a small experiment in combining two known techniques and spreading them over many cores. Two-stage induced sorting, as in libdivsufsort, sorts only about a third of the suffixes directly and derives the rest. libdivsufsort sorts those suffixes by their substrings up to the next directly sorted position, then runs prefix doubling over all of them. sadcs string-sorts them to a fixed depth instead and breaks the remaining ties with a difference cover sample, from Burkhardt and Kärkkäinen's *Fast Lightweight Suffix Array Construction and Checking* (CPM 2003), which makes comparing two suffixes O(1) after a bounded number of characters. The sample is only a sorting aid, not a recursion as in DC3/skew or DCX. Only the sample goes through prefix doubling, and that runs alongside the main sort. The code builds on Yuta Mori's libdivsufsort and adds OpenMP and SIMD.

libsais implements SA-IS (Nong, Zhang and Chan), which runs in linear time and is carefully tuned for the cache. It does less total work than sadcs and is the fastest on one thread, but its induction and recursion are largely serial, and its speedup levels off at about 2×. Almost all of sadcs's work splits into independent buckets, tasks and blocks, so it scales to all cores.

The independent checker, based on the checking algorithm from Burkhardt and Kärkkäinen's paper, serves both variants and shares no sorting code with either construction. It verifies any suffix array, including ones from other tools (the file format is divsufsort's). How it checks is under [How it works](#how-it-works).

## How it works

sadcs uses a difference cover D modulo v (v = 128 by default): a set of residues such that for any two positions `i` and `j` there's a `k < v` with both `i + k` and `j + k` in D, mod v. The default is `{0,6,11,26,30,34,42,43,44,75,82,104,107}` mod 128 (about 10% of all positions). `--cover 64` uses `{0,1,2,5,14,16,34,42,59}` (about 14%), and `--cover 256` a cover of 20 residues (about 8%). With 9 and 13 residues, the covers modulo 64 and 128 have the minimum size; for 256 no exhaustive search has settled whether 20 is minimal.

1. **Rank the sample.** The suffixes at positions `i` with `i mod v` in D are bucketed by two characters and string-sorted to depth v + 1. Whatever is still tied goes to Larsson–Sadakane prefix doubling with tandem repeat handling: when the keys of a tied group point back into the group itself, as on periodic text such as `abcabc…`, the order of that part is copied from the finished parts on either side instead of taking about log(run length / period) more rounds. Mid-sized groups are radix sorted on their keys, using the unused part of the suffix array as scratch. Afterwards, two suffixes that agree on their first v − 1 characters compare by `rank[i + k]` vs. `rank[j + k]` in O(1).
2. **Classify.** A suffix is S if it's smaller than the next one and L if it's larger. The RMS (rightmost S) suffixes, S suffixes followed by an L suffix, are about a third of the text. libdivsufsort calls them B\*.
3. **Sort the RMS suffixes.** They are bucketed by two characters and string-sorted to depth v − 1, and the sample ranks break the remaining ties. Within a tie group, suffixes at the same distance `d` from the next sample position compare by `rank[i + d]` alone. So each of these classes (at most 31 with the default cover) is radix sorted on integer keys looked up once, and only the merge of the classes uses the general comparison. Groups of up to 2^15 suffixes are sorted one per thread. Larger ones are sorted with tasks afterwards: the classes in parallel, and each merge split into pieces along merge paths.
4. **Induce the rest.** One right-to-left scan places all other S suffixes and one left-to-right scan all L suffixes, each in front of its already placed successor.

Every step runs in parallel except the prefix doubling, which runs alongside step 3 (see [the pipeline](#the-overlapped-execution-pipeline)).

**Parallelism and SIMD.** Bucketing is a parallel counting sort. String sorting is a multikey quicksort whose partitions become OpenMP tasks, so a single huge bucket (`aaaa…`) still spreads over all threads, and so do large tie groups. Partitioning compares 7 characters at a time: a big-endian word plus a length tag in the low byte, so the end of the text sorts correctly. It's branch-free unless the sampled pivot shows many equal keys, in which case a three-way partition gathers them. Mid-sized groups get a one-character radix pass first, small groups are insertion sorted on keys loaded once per suffix, and the text is prefetched ahead of the partitioning loops.

**Checking.** The default and the `--lowmem` check are Burkhardt and Kärkkäinen's. The default builds the inverse suffix array in parallel and verifies that every entry is in range and unique, and that each neighbour pair is ordered by its first character or by the rank of the suffix one character later. `--lowmem` runs the sequential bucket-walk check (O(σ) extra memory plus an n-bit permutation bitmap). `--naive` compares every neighbour pair in full, with an AVX2 mismatch search.

### The overlapped execution pipeline

The only sequential phase is the sample's prefix doubling. Its Larsson–Sadakane rounds can't usefully be split across threads, because each group profits from the groups refined just before it in the same round. So instead of running it in parallel, sadcs runs it alongside other work.

The RMS string sort, the biggest phase, needs no sample ranks. Only its ties do, and they are sorted afterwards. So once the RMS suffixes are bucketed by their first two characters, one OpenMP parallel region runs two things at once:

```
                           ┌───────────────────── RMS sort ──────────────────────┐
bucket sample ─ classify ─ bucket pass ─┬─ string sort ──────────────────────────┬─ tie groups ─ induce S ─ induce L
                                        └─ sample string sort ─ prefix doubling ─┘
```

- The **sample chain** is one task: the sample's string sort (itself task-parallel, in a `taskgroup`), followed by the prefix doubling on whichever thread finishes it. The chain has OpenMP task priority 1.
- The **RMS string sort** spawns one task per two-character bucket, with priority 0.

Since the chain is the critical path, idle threads take its string sort tasks first. Once it reaches the prefix doubling, one thread runs that while all the others take the RMS buckets. On enwik8 the doubling then takes only milliseconds, and on Silesia its 0.07s disappear entirely into the 0.9s of the main sort. Only on very repetitive input, such as nci or the Gauntlet corpus, can the doubling still outlast the string sort.

The two sorts share the suffix array without extra memory. Its back part holds the RMS marks, then the sample's suffix array and ranks. The doubling uses the free stretch between the RMS entries and the marks as radix scratch.

With MSVC (`/openmp:llvm`, which has neither task priorities nor `taskgroup`) the sample's string sort runs before the RMS sort, and only the doubling overlaps.

On one thread nothing can overlap, so the RMS suffixes are sorted first and the sample is only built if their sort leaves ties. Inputs where no two RMS suffixes share their first v − 1 characters skip the sample entirely; osdb, sao and x-ray from Silesia get 11–17% faster that way.

### Parallel induction blocks

Induction is inherently a sequential scan. Reading `sa[i]` places the suffix one position earlier at the front (or back) of its bucket, and that slot may be read later in the same scan. Splitting the array between threads would break this order.

sadcs keeps the sequential semantics but scans in blocks of up to 2^18 entries. A block is a stretch of one bucket's part whose slots are all final already, so every suffix it induces lands outside it. The threads process a block in four steps:

1. Each thread scans its share of the block. It collects the suffixes it induces, with their first characters, in its part of a block-sized buffer, and counts them per target bucket.
2. A prefix sum over the threads, in scan order, gives each thread its own write position in every bucket.
3. Each thread writes its suffixes from the buffer to those positions, in parallel.
4. The bucket cursors advance by the totals.

The result is exactly what the sequential scan would produce, with no global lock and no atomics. libsais parallelizes its induction scans with the same scheme. Where the scan catches up with slots that are still being filled (long runs of one character, such as `aaaa…` in an S bucket), there is no final stretch ahead, so one thread goes on alone. Short stretches under 2^13 entries also take the single-thread path. The buffer costs 5 bytes per block entry, about 1.3 MB in total. The text is prefetched 32 entries ahead. Beyond that the scans are bound by memory bandwidth.

Only about half the entries a scan reads induce anything: the S scan needs those whose predecessor is S, the L scan those whose predecessor is L. For texts of 64 MiB and more, an entry carries its predecessor's type in the sign bit, as in libsais, so the scans skip the random text read for the others. Placing a suffix reads its first character, and the one before it is nearly always in the same cache line, so the tag costs next to nothing. The RMS suffixes start untagged: the S scan reads their predecessor anyway and tags them in place for the L scan, which clears all tags as it goes. This saves about a quarter of the induction time on enwik9. While the text fits in the cache, the skipped reads are cheap and setting and clearing the tags costs more than it saves, so smaller texts are induced without them.

### Building the BWT directly

With `--bwt`, sadcs builds the Burrows–Wheeler transform instead of the suffix array, the way libdivsufsort's `divbwt` does: the two induction scans write it in place of the suffix array, so no separate pass has to gather `T[SA[i] - 1]`. Once the S scan has induced from an entry, the L scan won't need it, so the S scan replaces it with its BWT character. The L scan does the same for every entry it reads. With tags, an L suffix whose predecessor is S induces nothing, so it is placed as its BWT character right away: the tag check has just read that character. A final parallel pass packs the characters into bytes and drops the sentinel row.

The gather is random reads over the whole text, so leaving it out matters. On enwik8 the BWT takes 0.63s with 16 threads against 0.78s for the suffix array plus the gather (3.68s against 4.40s on one thread), and 0.61s for the suffix array alone. libsais 2.12.1 takes 1.20s for the BWT of enwik8 with 16 threads, and 2.66s on one. sadcs's BWT is identical to libsais's on Silesia, Gauntlet, enwik8 and enwik9.

## Running time and memory

- **Running time is O(n log n + v·n) in practice, O(v·n log n) in the worst case, not linear.** Classifying, bucketing and induction are O(n). The two string sorts are multikey quicksorts to depth v ± 1, O(m log m + v·m) for m suffixes when the pivots split well. Like introsort, they fall back to `std::sort` on a range that has been partitioned about 2 log2(size) times without getting deeper into the text, which bounds them at O(v·m log m). Breaking a tie costs O(1), and the prefix doubling on the sample is O(m log m). libsais (SA-IS) is O(n) in the worst case.
- **Memory is 5n bytes plus tens of MB**: the text (n bytes plus 64 bytes of padding) and the suffix array (4n), the same as libsais. All working arrays live in parts of the suffix array that aren't used yet. The rest, mainly per-thread buffers, takes about 5 MB plus 2.5 MB per thread. The default checker needs another 4n.
- Inputs up to 2 GiB − 1 KiB (int32 indexes). Memory is 5n for the text and the suffix array, plus about 5 MB and 2.5 MB per thread, independent of the cover. The working arrays (the RMS marks, the sample's ranks and suffix array, the doubling's and tie sort's scratch) live in the part of the suffix array the RMS suffixes don't use yet: there are at most n/2 of them. Measured as the process's peak private memory, `sadcs build` takes 5n + 8 MB on enwik8 with one thread and 5n + 47 MB with 16. On enwik9 with 16 threads it takes 5n + 56 MB (5.06 GB), and libsais 5n + 14 MB (5.01 GB). The default checker needs another 4n.

## Performance

Ryzen 7 5800X (8 cores, 16 threads), Windows 11, clang-cl 22, AVX2. 16 threads unless noted:

| file | size | cover | total | sample sort | doubling | main sort | check |
|---|---|---|---|---|---|---|---|
| enwik8 | 100 MB | 128 | 0.59s | 0.20s | 0.00s | 0.58s | 0.93s |
| enwik8 (1 thread) | 100 MB | 128 | 3.33s | 0.51s | 0.00s | 3.28s | 2.07s |
| Silesia (12 files) | 212 MB | 128 | 0.93s | 0.28s | 0.07s | 0.90s | 1.36s |
| Silesia nci | 34 MB | 128 | 0.19s | 0.07s | 0.05s | 0.19s | 0.24s |
| Gauntlet (10 files) | 66 MB | 128 | 0.52s | 0.09s | 0.23s | 0.51s | 0.28s |
| enwik9 | 1 GB | 128 | 10.50s | 3.23s | 0.32s | 10.41s | 10.03s |
| Pizza&Chili sources | 211 MB | 128 | 1.34s | 0.42s | 0.12s | 1.32s | 1.93s |
| Pizza&Chili dna | 404 MB | 128 | 5.05s | 2.16s | 0.03s | 5.02s | 3.94s |
| Pizza&Chili proteins | 1.18 GB | 128 | 14.75s | 3.54s | 3.10s | 14.65s | 11.76s |

Best of 3 runs (enwik9 one run). Only the sample's bucket pass runs before the main sort. Its string sort and the doubling run during it (see [the pipeline](#the-overlapped-execution-pipeline)), so they are part of the main sort's time, and the columns add up to more than the total. Sharing the machine with the main string sort makes both slower than they would be alone.

### Choice of cover

Construction time by cover, best of 3 runs (enwik9 one run):

| cover | sample | Silesia | Silesia, 1 thread | enwik8 | Gauntlet | Gauntlet, 4 threads | Gauntlet, 1 thread | enwik9 |
|---|---|---|---|---|---|---|---|---|
| 64 | 14.1% | 1.00s | 6.04s | 0.60s | 0.62s | 0.70s | 2.21s | 10.83s |
| 128 | 10.2% | 0.92s | 6.01s | 0.58s | 0.52s | 0.76s | 2.59s | 10.50s |
| 256 | 7.8% | 0.96s | 6.05s | 0.56s | 0.57s | 1.00s | 3.45s | 9.65s |

16 threads unless noted. Cover 128 is the default: it's the fastest or close to it almost everywhere. Only highly repetitive input with few threads favours cover 64: by about a seventh on one thread and 8% on 4. Cover 256 gains about 8% on large, less repetitive input such as enwik9, but costs the most with few threads on repetitive input. A cover modulo 512 (6.1% sample) was tried as well. It only gained another 2–3% on enwik8 and enwik9 and was slower than cover 256 on repetitive input, even with 16 threads.

### Thread scaling against libsais

Construction time only, cover 128. Silesia is summed over its 12 files, best of 5 runs. enwik8 is best of 3 and enwik9 a single run. libsais 2.12.1 was built with the same compiler, flags and OpenMP runtime. Both produce identical suffix arrays.

| threads | Silesia sadcs | libsais | ratio | enwik8 sadcs | libsais | ratio | enwik9 sadcs | libsais | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.96s | 3.51s | 1.70 | 3.33s | 2.53s | 1.32 | 47.94s | 29.19s | 1.64 |
| 2 | 3.13s | 2.72s | 1.15 | 1.78s | 1.72s | 1.03 | 26.27s | 21.37s | 1.23 |
| 3 | 2.21s | 2.32s | 0.95 | 1.28s | 1.40s | 0.91 | 19.55s | 17.61s | 1.11 |
| 4 | 1.74s | 2.04s | 0.85 | 1.01s | 1.26s | 0.80 | 15.50s | 15.97s | 0.97 |
| 6 | 1.27s | 1.89s | 0.67 | 0.78s | 1.13s | 0.69 | 12.12s | 14.97s | 0.81 |
| 8 | 1.09s | 1.90s | 0.57 | 0.69s | 1.13s | 0.61 | 11.09s | 15.03s | 0.74 |
| 12 | 0.98s | 1.95s | 0.50 | 0.62s | 1.11s | 0.56 | 10.36s | 14.79s | 0.70 |
| 16 | 0.92s | 2.02s | 0.45 | 0.59s | 1.11s | 0.53 | 10.50s | 14.95s | 0.70 |

Ratio is sadcs ÷ libsais; below 1 sadcs is faster. On Silesia sadcs gets 5.5× from 8 cores, and hyper-threading adds about 18% on top, for 6.5× in all. libsais levels off at about 1.9× on Silesia (2.3× on enwik8, 2.0× on enwik9), and gets slightly slower again past 6 to 8 threads. enwik9 is the hardest case for sadcs: a 1 GB suffix array doesn't fit in any cache, so the induction scans are bound by memory bandwidth, and sadcs gets 4.6×. Its 12- and 16-thread times are single runs and within noise of each other.

On three files of the [Pizza&Chili corpus](https://pizzachili.dcc.uchile.cl/texts.html), with 1 and 16 threads, best of 3 runs (libdivsufsort as in [the next section](#libdivsufsort): without OpenMP on one thread, with OpenMP on 16):

| file | size | sadcs, 1 thread | libsais | libdivsufsort | sadcs, 16 threads | libsais | libdivsufsort |
|---|---|---|---|---|---|---|---|
| sources | 211 MB | 8.07s | 5.18s | 8.58s | 1.34s | 2.80s | 6.45s |
| dna | 404 MB | 20.24s | 10.36s | 23.33s | 5.05s | 5.30s | 20.87s |
| proteins | 1.18 GB | 82.63s | 39.05s | 86.77s | 14.75s | 20.49s | 50.71s |

With 16 threads sadcs is ahead on all three, by 2.1× on sources and 1.4× on proteins, but only by 5% on dna. sadcs gets 6.0× from 16 threads on sources and 5.6× on proteins, but 4.0× on dna, whose sample sort takes 2.16s of the 5.05s and scales only 1.9× (4.07s on one thread). libsais gets 1.85× to 1.95× on all three. sadcs overtakes libsais from 3 threads on sources, 6 on proteins and 8 on dna (5.32s against 5.35s; the [chart in the README](../README.md#cpu-variant) has all thread counts). On one thread libsais is 1.6 to 2.1 times as fast as sadcs, and sadcs is 5% to 13% faster than libdivsufsort. english (2.2 GB) exceeds the 2 GiB limit.

### libdivsufsort

Construction time of libdivsufsort 2.0.2 (current master), built with `USE_OPENMP` and the same compiler, flags and OpenMP runtime, measured the same way as libsais above. Ratio is sadcs ÷ libdivsufsort, with the sadcs times from the table above.

| threads | Silesia | ratio | enwik8 | ratio | enwik9 | ratio |
|---|---|---|---|---|---|---|
| 1, without OpenMP | 6.38s | 0.93 | 4.80s | 0.69 | 54.74s | 0.88 |
| 1 | 6.72s | 0.89 | 5.02s | 0.66 | 57.48s | 0.83 |
| 2 | 5.69s | 0.55 | 4.12s | 0.43 | 48.57s | 0.54 |
| 3 | 5.43s | 0.41 | 3.82s | 0.34 | 44.37s | 0.44 |
| 4 | 5.21s | 0.33 | 3.67s | 0.28 | 42.98s | 0.36 |
| 6 | 5.11s | 0.25 | 3.57s | 0.22 | 41.28s | 0.29 |
| 8 | 5.12s | 0.21 | 3.50s | 0.20 | 40.73s | 0.27 |
| 12 | 5.27s | 0.19 | 3.50s | 0.18 | 40.35s | 0.26 |
| 16 | 5.32s | 0.17 | 3.47s | 0.17 | 40.52s | 0.26 |

Only the substring sort runs in parallel, so libdivsufsort levels off at 1.25× to 1.4× over its build without OpenMP, from about 6 threads. On one thread the OpenMP build is about 5% slower than the plain one. sadcs is faster on one thread on all three inputs, but not on the Gauntlet corpus: 2.59s against libdivsufsort's 2.19s. With 16 threads, sadcs takes 0.52s there and libdivsufsort 2.35s. All three libraries produce identical suffix arrays.

### Where libsais is faster

- **One core is slower.** sadcs does more total work than SA-IS: string sorting isn't linear, and long repeats cost up to v − 1 characters per comparison, where v is the modulus of the difference cover (128 by default, see [How it works](#how-it-works)). On one thread libsais is 1.3 to 2.1 times as fast. sadcs wins only when the extra work spreads over enough cores.
- **Highly repetitive input favours libsais.** On the [Gauntlet corpus](https://github.com/michaelmaniscalco/gauntlet_corpus) with 16 threads libsais takes 0.47s against sadcs's 0.52s. On one thread even libdivsufsort is faster there: 2.19s against 2.59s. Long repeats leave much of the sample to the prefix doubling, which is sequential.
- **Small inputs favour libsais.** On the [Calgary corpus](https://corpus.canterbury.ac.nz/descriptions/#calgary) (18 files, 3.2 MB) libsais is 1.6 to 1.8 times as fast at any thread count: 36, 27 and 34 ms with 1, 4 and 16 threads, against sadcs's 66, 44 and 54 ms. Each thread adds fixed costs, mainly its own 65 792-entry counting tables, while the work per thread shrinks, so sadcs runs an input of n bytes on at most about √(n / 8 KB) threads (all 16 from 2 MB on). Fitted on Calgary, this comes within 6% of the best thread count per file. What remains is the character-pair tables every build fills, whatever the input size.
