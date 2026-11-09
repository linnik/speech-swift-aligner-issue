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
