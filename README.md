# sadcs

A CPU and CUDA implementation of suffix array and Burrows–Wheeler transform construction, in one executable with one [command line](#command-line):

- **CPU** ([CPU variant](#cpu-variant)): two-stage induced sorting with a difference cover sample and prefix doubling, parallelized with OpenMP tasks. Slower than libsais with few threads, faster with many threads on large inputs.
- **GPU** ([GPU variant](#gpu-variant)): a difference cover sample, radix-sorted classes and a tiled multiway merge in CUDA, with the whole construction on the device. On a GeForce RTX 5070 Ti it is faster than the tested libcubwt versions on every input in the benchmark, and needs 10.02 bytes of device memory per input byte, half or less of what libcubwt needs, so the same card handles larger inputs.

The goal is a lightweight suffix array construction algorithm (SACA): the CPU variant needs about 5n bytes, the text and the suffix array, plus a few tens of MB, the same as libsais. The GPU variant needs about 10n on the device and 5n on the host, which is little for a GPU-based SACA: libcubwt needs about 30n for a suffix array.

Both produce the same suffix arrays and BWTs, in divsufsort's file formats. An independent checker verifies them, and those of other tools too.

The algorithms build on prior work; the contribution is their combination, parallelization and implementation (see [References and credits](#references-and-credits)). For production use, I recommend [libsais](https://github.com/IlyaGrebnov/libsais) on the CPU and [libcubwt](https://github.com/IlyaGrebnov/libcubwt) on the GPU. Both are established implementations; sadcs has been benchmarked on one CPU and one GPU so far.

## CPU variant

The CPU variant combines two known techniques and spreads them over many cores. Two-stage induced sorting, as in [libdivsufsort](https://github.com/y-256/libdivsufsort), sorts only about a third of the suffixes directly and derives the rest. A difference cover sample, from Burkhardt and Kärkkäinen's *Fast Lightweight Suffix Array Construction and Checking* (CPM 2003), breaks the ties after a bounded number of characters. The cover is modulo 128 by default, about 10% of all positions:

1. **Rank the sample.** String-sort the sample suffixes to depth v + 1, then finish the ties with Larsson–Sadakane prefix doubling. Afterwards, two suffixes that share their first v − 1 characters compare in O(1) by sample ranks.
2. **Classify** every suffix as S (smaller than the next) or L (larger). The RMS suffixes, S followed by L, are about a third of the text.
3. **Sort the RMS suffixes** by a string sort to depth v − 1, with the sample ranks breaking the remaining ties.
4. **Induce the rest** with one right-to-left and one left-to-right scan.

The string sorts are task-parallel multikey quicksorts that compare 7 characters at a time. The sequential prefix doubling runs alongside the RMS sort, and the induction scans run in parallel blocks. With `--bwt`, the induction scans write the BWT directly. [docs/cpu-design.md](docs/cpu-design.md) has the details and all measurements.

| construction time | sadcs, 1 thread | libsais, 1 thread | sadcs, 16 threads | libsais, 16 threads | sadcs speedup, 16 threads |
|---|---|---|---|---|---|
| Silesia (12 files, 212 MB) | 5.96s | 3.51s | 0.92s | 2.02s | 2.2× |
| enwik8 (100 MB) | 3.33s | 2.53s | 0.59s | 1.11s | 1.9× |
| enwik9 (1 GB) | 47.94s | 29.19s | 10.50s | 14.95s | 1.4× |
| Pizza&Chili sources (211 MB) | 8.07s | 5.18s | 1.34s | 2.80s | 2.1× |
| Pizza&Chili dna (404 MB) | 20.24s | 10.36s | 5.05s | 5.30s | 1.05× |
| Pizza&Chili proteins (1.18 GB) | 82.63s | 39.05s | 14.75s | 20.49s | 1.4× |

The four-letter dna is the closest case: sadcs gets only 4.0× from 16 threads there, the least of these inputs (4.6× to 6.5× on the others). I haven't tested other hardware. libsais is memory-bound and, as its README says, sensitive to fast memory, so on other machines the crossover may well lie elsewhere. sadcs depends on memory too, at least in its induction scans, which are bound by memory bandwidth on enwik9. Which way other hardware would shift the comparison I can't say.

[libdivsufsort 2.0.2](https://github.com/y-256/libdivsufsort), which sadcs builds on, is slower than sadcs on these inputs at every thread count tested (see [libdivsufsort](docs/cpu-design.md#libdivsufsort)).

![Construction time vs. threads for sadcs, libsais and libdivsufsort on Silesia, enwik8, enwik9 and the Pizza&Chili files sources, dna and proteins, measured at 1, 2, 3, 4, 6, 8, 12 and 16 threads. sadcs is slower on one thread and faster than libsais from 3 threads on Silesia, enwik8 and sources, from 4 on enwik9, from 6 on proteins and from 8 on dna; with 16 threads it takes 0.92s, 0.59s, 10.50s, 1.34s, 5.05s and 14.75s against libsais's 2.02s, 1.11s, 14.95s, 2.80s, 5.30s and 20.49s. libdivsufsort, built with OpenMP, is the slowest at every thread count and levels off at about 5.1s, 3.5s, 40.4s, 6.5s, 20.9s and 50.7s.](docs/scaling.svg)

### CPU tradeoffs

- **One core is slower.** sadcs does more total work than SA-IS: on one thread libsais is 1.3 to 2.1 times as fast.
- **Highly repetitive input favours libsais.** On the [Gauntlet corpus](https://github.com/michaelmaniscalco/gauntlet_corpus) with 16 threads libsais takes 0.47s against sadcs's 0.52s: long repeats leave much of the sample to the sequential prefix doubling.
- **Small inputs favour libsais.** On the [Calgary corpus](https://corpus.canterbury.ac.nz/descriptions/#calgary) (18 files, 3.2 MB) libsais is 1.6 to 1.8 times as fast at any thread count: 27 ms against 44 ms with 4 threads.
- **Not linear-time.** The string sorts and tie resolution can do substantially more work than SA-IS, particularly on repetitive input.
- **Memory is 5n bytes plus tens of MB**, the text and the suffix array, the same as libsais.

See [Where libsais is faster](docs/cpu-design.md#where-libsais-is-faster) and [Running time and memory](docs/cpu-design.md#running-time-and-memory).

## GPU variant

The GPU variant builds the same suffix arrays and BWTs on an NVIDIA GPU with CUDA. Of the two variants it is the more interesting one: on the CPU, libsais stays faster on one thread, on highly repetitive input and on small inputs, while on the GPU sadcs is faster than libcubwt on every input tested here and builds inputs two to three times as large on the same card.

It keeps the difference cover, here modulo 32 (7 residues, a 21.9% sample), but none of the CPU variant's sorting, which is sequential at heart or doesn't map to warps. Instead it uses radix sorts, scans and one multiway merge, and the entire construction stays on the device between upload and download:

1. **Rank the sample** by radix sorts on the first 36 characters, then prefix doubling until every sample suffix has a unique rank.
2. **Sort the non-sample positions** in 6 classes, by their distance to the next sample position. Each class is derived from the previous one with a single 8-bit radix pass.
3. **Merge the 7 sorted lists** (the sample and the 6 classes) in one pass: splitters divide the output into tiles of about 256 elements, and each tile is merged in shared memory along merge paths. The sample ranks break ties after at most 32 characters.
4. **For the BWT**, the merge writes the character before each suffix directly.

[docs/gpu-design.md](docs/gpu-design.md) explains each step, the memory budget, the tuning and what didn't work.

Measured on one card only, the way libcubwt's own benchmark measures (`sadcs bench --context --pinned`: GPU context and page-locked host buffers allocated beforehand, best of 5 runs, enwik9 best of 3), on a GeForce RTX 5070 Ti (16 GB) with CUDA 13.5, in ms, transfers included. libcubwt 1.0.0 is the last version that builds suffix arrays; 1.6.3 builds only the BWT.

| input | size | sadcs SA | libcubwt 1.0.0 SA | sadcs BWT | libcubwt 1.6.3 BWT |
|---|---|---|---|---|---|
| Silesia (12 files) | 212 MB | 213 | 428 | 195 | 309 |
| Gauntlet (10 files) | 66 MB | 146 | 421 | 142 | 285 |
| enwik8 | 100 MB | 95 | 122 | 86 | 93 |
| enwik9, first 500 MB | 500 MB | 535 | 679 | 494 | 528 |
| enwik9 | 1 GB | 1142 | doesn't fit | 1061 | doesn't fit |
| Pizza&Chili sources | 211 MB | 232 | 343 | 215 | 251 |
| Pizza&Chili dna | 404 MB | 415 | 559 | 381 | 437 |
| Pizza&Chili proteins | 1.18 GB | 1546 | doesn't fit | 1450 | doesn't fit |

libcubwt is tuned for other GPUs, and on other cards the comparison may well come out differently.

![Throughput in MB/s of sadcs and libcubwt on the GeForce RTX 5070 Ti, from the table above. Suffix array: Silesia 995 against 495, Gauntlet 453 against 157, enwik8 1053 against 820, the first 500 MB of enwik9 935 against 736, enwik9 876, where libcubwt 1.0.0 doesn't fit. BWT: Silesia 1087 against 686, Gauntlet 465 against 232, enwik8 1163 against 1075, the first 500 MB of enwik9 1012 against 947, enwik9 943, where libcubwt 1.6.3 doesn't fit. Pizza&Chili, suffix array: sources 909 against 615, dna 974 against 722, proteins 766, where libcubwt doesn't fit; BWT: sources 982 against 842, dna 1059 against 923, proteins 816.](docs/gpu-throughput.svg)

### GPU tradeoffs

- **Memory is 10 bytes per input byte on the device** (10.02, the same on every input), so a 16 GB card builds inputs up to about 1.547 GB. libcubwt needs 20.5 (1.6.3) or about 30 (1.0.0). Inputs that don't fit are refused rather than left to the driver's slow fallback to host memory. The host needs 5n for a suffix array, n for a BWT.
- **Small inputs pay a fixed cost** of about 2 to 3 ms each in kernel launches and syncs: Calgary takes 53 ms, against 29 ms for libsais.
- **Repetitive input needs more doubling rounds**: long repeated prefixes leave more sample suffixes unresolved after the initial radix sorts. Gauntlet (66 MB) takes 146 ms for the suffix array, against 95 ms for enwik8 (100 MB).
- **The cover is fixed at modulo 32**, so `--cover` and `--threads` don't apply. Covers modulo 64 were tried and bring no advantage: they are 30% to 117% slower, mostly in the merge (see [Choice of cover](docs/gpu-design.md#choice-of-cover)).
- **Tested on one card**, a GeForce RTX 5070 Ti (sm_120) with CUDA 13.5 on Windows.

## Test corpora

The benchmarks use these public corpora:

- [Silesia corpus](https://sun.aei.polsl.pl//~sdeor/index.php?page=silesia): 12 files, 212 MB of mixed text, binaries and data.
- [enwik8 and enwik9](https://mattmahoney.net/dc/textdata.html): the first 100 MB and 1 GB of an English Wikipedia XML dump, from the Large Text Compression Benchmark.
- [Gauntlet corpus](https://github.com/michaelmaniscalco/gauntlet_corpus): 10 files, 66 MB, mostly highly repetitive.
- [Calgary corpus](https://corpus.canterbury.ac.nz/descriptions/#calgary): 18 files, 3.2 MB.
- [Pizza&Chili corpus](https://pizzachili.dcc.uchile.cl/texts.html): sources (211 MB), dna (404 MB) and proteins (1.18 GB) from its text collection. Its english (2.2 GB) exceeds the 2 GiB limit.

## Command line

One executable covers both variants. `build` and `test` run on the CPU, or with `--gpu` on the GPU. Both write the same files, so `check` verifies either, and `test --gpu` builds on both, compares the results and checks the GPU's. `bench` times the GPU only. `selftest` covers both when a GPU is present. A build without CUDA has the same commands, and there `--gpu` reports that sadcs was built without CUDA.

```
sadcs build|-b <in> <out.sa> [--cover 128|64|256] [--threads N] [--gpu]
sadcs build|-b --bwt <in> <out.bwt> [--cover 128|64|256] [--threads N] [--gpu]
sadcs check|-c <in> <in.sa> [--lowmem | --naive] [--threads N]
sadcs check|-c --bwt <in> <in.bwt>
sadcs test|-t <in|dir> [--bwt] [--gpu] [--cover 128|64|256] [--threads N] [--recursive]
sadcs bench <in|dir> [--bwt] [--runs N] [--context] [--pinned] [--recursive]
sadcs selftest [--verbose] [--threads N]
```

- `build`: write the suffix array as n little-endian int32 values (divsufsort's `mksary` format).
- `check`: verify a suffix array file against its text. If it fails, you get the first bad index and a preview of the suffixes involved.
- `test`: build and check in memory, with a per-phase timing breakdown. Give it a directory and it runs every file in it (add `--recursive` to include subdirectories), printing one row per file plus a total, which is handy for whole corpora:

  ```
  > sadcs test Testfiles\Silesia
  Testfiles\Silesia: 12 files, cover 128, 16 threads (times in seconds)
  file           bytes   sample doubling     main    total     MB/s    check
  dickens     10192446    0.010    0.000    0.032    0.035    276.2    0.039  OK
  mozilla     51220480    0.061    0.007    0.182    0.188    260.5    0.389  OK
  ...
  total      211938580    0.271    0.071    0.896    0.925    218.5    1.279  12/12 OK
  ```

  The exit code is nonzero if any file fails the check or can't be read.
- `bench`: times the GPU build (see [GPU variant](#gpu-variant)): a warmup, then the best of `--runs` (default 5) runs from host memory to host memory, with the time of each phase. It doesn't check the output; `test --gpu` does. `--context` reuses one GPU context per file, created before the runs and not timed, as a caller building many inputs would. `--pinned` uses page-locked host buffers allocated before the runs. `--context --pinned` measures the way libcubwt's benchmark does.
- `selftest`: compares against a naive suffix array on small adversarial inputs (runs, periods around the cover sizes, Fibonacci, Thue–Morse, random), runs 4 MB inputs through the parallel paths, checks that the checkers catch planted errors, and compares the GPU's output with the CPU's when a device is present.
- `--cover`: 128 (default), 64 or 256 (CPU only). Cover 64 is faster on highly repetitive input with one thread, cover 256 on large, less repetitive input with many threads. See [Choice of cover](docs/cpu-design.md#choice-of-cover).
- `--bwt`: build the BWT instead (see [Building the BWT directly](docs/cpu-design.md#building-the-bwt-directly)). The file is in divsufsort's `bwt` format as a single block: n and the primary index as little-endian int32 values, then the n bytes, so its `unbwt` decodes it. `build` writes the BWT in place of the text, still 5n bytes in all. `check` and `test` verify it by inverting it, which is sequential and slow on large inputs: 103s on enwik9.
- `--threads`: defaults to all hardware threads (CPU only).
- `--gpu`: build the suffix array or the BWT on the GPU (see [GPU variant](#gpu-variant)). It always uses cover 32, so `--cover` and `--threads` don't apply. Inputs that don't fit in free device memory are refused. It needs a build with CUDA, see [Build](#build).

## Limits

- Inputs up to 2 GiB − 1 KiB (int32 indexes). The CPU variant needs 5n plus about 5 MB and 2.5 MB per thread (measured in [Running time and memory](docs/cpu-design.md#running-time-and-memory)); the default checker needs another 4n.
- The GPU variant needs about 10n bytes of free device memory, so about 1.5 GB of input on a 16 GB card. It's tested on one card only (GeForce RTX 5070 Ti, sm_120).
- Windows x64 only for now, built with clang-cl or MSVC. The code uses a few Windows-specific C runtime functions (`_fseeki64`, `_putenv_s`), so it doesn't build on Linux or macOS yet.

## Build

sadcs currently builds on Windows x64 only (see [Limits](#limits)). You'll need:

- Visual Studio 2022 with the "Desktop development with C++" workload.
- CMake 3.23 or newer.
- For the default `clang` preset, the standalone LLVM package installed to `C:\Program Files\LLVM`. It also provides the OpenMP runtime (`libomp`).

Then:

```
build.bat
```

If the CUDA toolkit is installed (`CUDA_PATH` is set), this builds the CPU and GPU variants into `build-cuda\bin\Release\` (see below), otherwise the CPU variant only into `build\bin\Release\`. `build.bat cpu` always builds the CPU variant only.

The CPU variant by hand:

```
cmake --preset clang -S .
cmake --build build --config Release
```

That produces `sadcs.exe` plus the OpenMP runtime DLL in `build\bin\Release\`. Keep them together.

MSVC fallback (uses `/openmp:llvm`, since classic `/openmp` is OpenMP 2.0 without tasks):

```
cmake --preset msvc -S .
cmake --build build-msvc --config Release
```

`-DSADCS_AVX2=OFF` builds the SSE2 fallback.

The GPU variant (see [GPU variant](#gpu-variant)) needs the CUDA toolkit 13 or newer (for CCCL 3) and its own preset, because the CUDA 13.5 Visual Studio integration fails CMake's compiler check. It is tested with CUDA 13.5. The `cuda` preset uses Ninja with clang-cl for the C++ sources and nvcc, with `cl.exe` as its host compiler, for the CUDA ones:

```
cmake --preset cuda -S .
cmake --build build-cuda
```

By hand, run these from a Developer Command Prompt, since Ninja needs the MSVC environment; `build.bat` sets it up itself. The other presets build the CPU variant only, and there `--gpu` reports that sadcs was built without CUDA.

## Where things live

- `src/cpu/`: the CPU variant; its interface is `dcsort.h`.
- `src/cuda/`: the GPU variant; its interface is `gpusa.h`. Builds without CUDA use `gpusa_stub.cpp` instead.
- `src/common/`: the difference covers, the checker and file I/O, shared by both.
- `src/cli/`: the command line and the selftest.
- `bench/libcubwt/`: the benchmark that times libcubwt, and a patch that builds libcubwt 1.0.0 with CUDA 13.
- `docs/`: the design records, [cpu-design.md](docs/cpu-design.md) and [gpu-design.md](docs/gpu-design.md), and the charts.

## License

MIT, see [LICENSE](LICENSE). `doubling.cpp` is derived from libdivsufsort (MIT), and builds with the GPU variant include NVIDIA's CUB (BSD-3-Clause); [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) has their notices. `cmake --install` puts both files next to `sadcs.exe`.

## References and credits

sadcs builds on the work below. Its contribution is the parallel implementation and the way the techniques are combined: on the CPU the overlapped pipeline, task-parallel string and tie-group sorting, and the 7-character key comparisons; on the GPU the class sorts derived from each other, and the multiway merge of 7 lists in shared memory, with the BWT written by the merge.

### Prior work

- Hideo Itoh and Hozumi Tanaka. "An Efficient Method for in Memory
  Construction of Suffix Arrays." Proceedings of the Sixth Symposium
  on String Processing and Information Retrieval (SPIRE), 1999,
  pp. 81–88. The induced-sorting approach that forms part of sadcs's design.

- Stefan Burkhardt and Juha Kärkkäinen. "Fast Lightweight Suffix
  Array Construction and Checking." Proceedings of the 14th
  Symposium on Combinatorial Pattern Matching (CPM), 2003,
  LNCS 2676, pp. 55–69.
  [DOI](https://doi.org/10.1007/3-540-44888-8_5).
  Difference-cover sampling and the basis for the independent checker.

- N. Jesper Larsson and Kunihiko Sadakane. "Faster Suffix Sorting."
  Theoretical Computer Science 387(3), 2007, pp. 258–272.
  [DOI](https://doi.org/10.1016/j.tcs.2007.07.017).
  The prefix-doubling algorithm used to finish sorting the sample.

- Michael A. Maniscalco and Simon J. Puglisi. "An Efficient, Versatile
  Approach to Suffix Sorting." ACM Journal of Experimental Algorithmics
  12, 2008, pp. 1–23.
  [DOI](https://doi.org/10.1145/1227161.1278374).
  Describes MSufSort, where I first saw the tandem repeat handling
  that sadcs's prefix doubling uses.

- Ge Nong, Sen Zhang and Wai Hong Chan. "Linear Suffix Array
  Construction by Almost Pure Induced-Sorting." Proceedings of the
  Data Compression Conference (DCC), 2009, pp. 193–202.
  [DOI](https://doi.org/10.1109/DCC.2009.42).
  SA-IS, the algorithm libsais implements.

- Udi Manber and Gene Myers. "Suffix Arrays: A New Method for On-Line
  String Searches." SIAM Journal on Computing 22(5), 1993, pp. 935–948.
  Prefix doubling, which the GPU variant uses in its plain form to rank
  the sample.

- Saher Odeh, Oded Green, Zahi Mwassi, Oz Shmueli and Yitzhak Birk.
  "Merge Path – Parallel Merging Made Simple." IEEE International
  Parallel and Distributed Processing Symposium Workshops (IPDPSW), 2012.
  The merge path partitioning the GPU variant's merge uses within a tile.

### Code and implementation

- [Yuta Mori's libdivsufsort](https://github.com/y-256/libdivsufsort)
  (MIT). The starting point for sadcs's two-stage induced sorting.
  `doubling.cpp` is a port of its `trsort.c`; the original license notice is retained.
  I added a radix sort for tied groups of 256 to 2^18 entries: they are sorted on
  cached keys, where `trsort.c` partitions them with a quicksort that needs about
  log2(size) passes of random key lookups.

- [Ilya Grebnov's libsais](https://github.com/IlyaGrebnov/libsais)
  (Apache 2.0). The reference for the benchmarks. sadcs's block-parallel
  induction and the predecessor-type tag follow the same scheme; no code
  is taken from it.

- [Ilya Grebnov's libcubwt](https://github.com/IlyaGrebnov/libcubwt)
  (Apache 2.0). The reference for the GPU benchmarks. No code is taken
  from it; `bench/libcubwt/` has the benchmark that times it and a patch that
  builds version 1.0.0 with CUDA 13 for sm_120.

- [NVIDIA CUB](https://github.com/NVIDIA/cccl), part of the CUDA toolkit
  (BSD 3-clause). The GPU variant's radix sorts, segmented sorts, scans,
  compactions and the merge sort of the lowest splitter level.
