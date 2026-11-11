# speech-swift aligner check

Reproduction for the speech-swift forced aligner bug: `CoreMLForcedAligner` on `test_audio.wav` against
`Qwen/Qwen3-ForcedAligner-0.6B` in PyTorch, compared slot by slot on raw timestamp indices.

Clone next to your speech-swift checkout (the package uses `../speech-swift`) and check out the commit to test.

```bash
swift build -c release
uv run check.py
```

`check.py`:

1. Writes a 16 kHz copy of the clip, so Swift and PyTorch read the same samples.
2. Downloads the INT8 bundle, compiled files only.
3. Loads the PyTorch model. The checkpoint needs a hand-built config and renamed weights for transformers.
4. Computes the reference mel features and audio embeddings.
5. Builds speech-swift's prompt: chat prefix, audio pads, suffix, then `word<timestamp><timestamp>` per word.
6. Gets PyTorch's timestamp classes at the slots.
7. Runs `Sources/aligncheck/main.swift` per compute-unit mode, and compares its raw indices with PyTorch.
8. Runs the bundle alone in coremltools with the reference inputs, per compute unit.

## v0.0.29

```
Raw timestamp indices, two per word (start, end), 80 ms each:
  PyTorch  [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]
  default  [225, 225, 202, 202, 223, 223, 225, 225, 225, 225, 225, 225, 225, 225, 225, 223, 225, 202, 202, 202, 202, 223]  0/22 equal to PyTorch
  all      [225, 225, 202, 202, 223, 223, 225, 225, 225, 225, 225, 225, 225, 225, 225, 223, 225, 202, 202, 202, 202, 223]  0/22 equal to PyTorch
  cpu      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  0/22 equal to PyTorch
  gpu      [0, 66, 69, 69, 69, 69, 77, 77, 79, 85, 86, 86, 94, 88, 92, 92, 93, 93, 100, 100, 105, 105]  6/22 equal to PyTorch

Word starts in seconds, after putting the times in order:
  PyTorch  5.28 5.52 5.52 6.16 6.80 6.88 7.04 7.36 7.44 8.00 8.40
  default  16.16 16.16 16.24 16.56 16.80 17.12 17.36 17.68 18.00 18.00 18.00
  all      16.16 16.16 16.24 16.56 16.80 17.12 17.36 17.68 18.00 18.00 18.00
  cpu      0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  gpu      0.00 5.28 5.44 5.52 6.32 6.80 6.88 7.04 7.36 7.44 8.00

The bundle alone in coremltools, with the reference inputs:
  units        encoder error decoder error  slots decoder ms
  CPU_ONLY              1.4%        100.0%   0/22        567
  CPU_AND_GPU           0.5%          0.4%  22/22        245
  CPU_AND_NE            1.8%         79.0%   0/22        242
  ALL                   1.8%         79.0%   0/22        235
```

`gpu` alone gets the times into range, but its raw list reads like PyTorch's shifted one slot right: the
timestamp slot layout bug.

## With the three fixes

```
Raw timestamp indices, two per word (start, end), 80 ms each:
  PyTorch  [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]
  default  [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]  22/22 equal to PyTorch
  all      [202, 202, 202, 202, 202, 223, 225, 225, 225, 225, 225, 202, 202, 202, 223, 202, 202, 202, 199, 199, 202, 207]  0/22 equal to PyTorch
  cpu      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  0/22 equal to PyTorch
  gpu      [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]  22/22 equal to PyTorch

Word starts in seconds, after putting the times in order:
  PyTorch  5.28 5.52 5.52 6.16 6.80 6.88 7.04 7.36 7.44 8.00 8.40
  default  5.28 5.44 5.52 6.16 6.80 6.88 7.04 7.36 7.44 8.00 8.40
  all      15.92 15.92 15.92 15.92 15.92 15.92 15.92 15.92 15.92 15.92 16.16
  cpu      0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  gpu      5.28 5.44 5.52 6.16 6.80 6.88 7.04 7.36 7.44 8.00 8.40
```

`default` is what `CoreMLForcedAligner.fromDirectory` picks; `all`, `cpu` and `gpu` pass `.all`, `.cpuOnly` and
`.cpuAndGPU` to both models. With equal raw indices, "you" still differs (5.44 vs 5.52): `TimestampCorrection` and
transformers' `_fix_timestamps` fill that out-of-order slot differently. The coremltools table doesn't depend on the
speech-swift commit.

## On main after #547–#552

The sections above are as they were when #546 was opened. Since then the three fixes and the maintainer's #550–#552
are on main (`468db5f`):

```
Raw timestamp indices, two per word (start, end), 80 ms each:
  PyTorch  [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]
  default  [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]  22/22 equal to PyTorch
  all      [202, 202, 202, 202, 202, 223, 225, 225, 225, 225, 225, 202, 202, 202, 223, 202, 202, 202, 199, 199, 202, 207]  0/22 equal to PyTorch
  cpu      [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]  22/22 equal to PyTorch
  gpu      [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]  22/22 equal to PyTorch

Word starts in seconds, after putting the times in order:
  PyTorch  5.28 5.52 5.52 6.16 6.80 6.88 7.04 7.36 7.44 8.00 8.40
  default  5.28 5.52 5.52 6.16 6.80 6.88 7.04 7.36 7.44 8.00 8.40
  all      16.16 16.16 16.16 16.16 16.16 16.16 16.16 16.16 16.16 16.16 16.16
  cpu      5.28 5.52 5.52 6.16 6.80 6.88 7.04 7.36 7.44 8.00 8.40
  gpu      5.28 5.52 5.52 6.16 6.80 6.88 7.04 7.36 7.44 8.00 8.40
```

What changed:

- `cpu` is correct: `.cpuOnly` now loads `text_decoder_cpu_v1`.
- "you" starts at 5.52, as in PyTorch: `TimestampCorrection` fills that slot the same way now.
- `all` is still wrong: the decoder lands on the Neural Engine.
- `aligncheck` also takes `ne`, for `.cpuAndNeuralEngine`.

## `pow` on the Neural Engine

```bash
uv run ane_pow.py
```

`ane_pow.py` builds a small Core ML model from scratch, with `pow(x, 2)` and `mul(x, x)` side by side, and runs it
on each compute unit. It needs no download. On an M1 Pro, macOS 15.8.2:

```
Placement with CPU_AND_NE: {'cast on CPU': 3, 'conv on NeuralEngine': 3, 'pow on NeuralEngine': 1, 'mul on NeuralEngine': 1}

x                              -3       -1     -0.5     -0.1        0      0.1      0.5        1        3   negative x equal to x^2
pow(x) on CPU_ONLY              9        1     0.25  0.00999        0  0.00999     0.25        1        9   100.0%
mul(x) on CPU_ONLY              9        1     0.25  0.00999        0  0.00999     0.25        1        9   100.0%
pow(x) on CPU_AND_GPU           9        1     0.25  0.00999        0  0.00999     0.25        1        9   100.0%
mul(x) on CPU_AND_GPU           9        1     0.25  0.00999        0  0.00999     0.25        1        9   100.0%
pow(x) on CPU_AND_NE            1        1        1        1        0     0.01     0.25        1        9     0.8%
mul(x) on CPU_AND_NE            9        1     0.25  0.00999        0  0.00999     0.25        1        9   100.0%
```

On the Neural Engine `pow(x, 2)` returns 1 for every negative `x`. The 0.8% are the inputs near −1, whose square
is 1. The aligner's text decoder squares with `pow` in its RMSNorm chains, which is why `all` is still wrong.
