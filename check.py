"""Compare speech-swift's CoreMLForcedAligner with Qwen3-ForcedAligner in PyTorch on test_audio.wav.

Run from this folder with `uv run check.py`, after `swift build -c release`, with speech-swift at ../speech-swift.
"""
import json
import os
import re
import statistics
import subprocess
import time

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3ASRConfig, Qwen3ASRForTokenClassification
from transformers.models.qwen3_asr.processing_qwen3_asr import _fix_timestamps

text = "Can you guarantee that the replacement part will be shipped tomorrow?"

# 1. A 16 kHz copy of speech-swift's test clip, so Swift and PyTorch read the same samples.
audio, rate = sf.read("../speech-swift/Tests/Qwen3ASRTests/Resources/test_audio.wav", dtype="float32")
count = int(len(audio) * 16000 / rate)
audio = np.interp(np.linspace(0, len(audio) - 1, count), np.arange(len(audio)), audio).astype(np.float32)
sf.write("test_audio_16k.wav", audio, 16000, subtype="PCM_16")
audio, _ = sf.read("test_audio_16k.wav", dtype="float32")

# 2. The CoreML INT8 bundle, compiled files only.
bundle = "models/coreml-int8"
if not os.path.isdir(f"{bundle}/text_decoder.mlmodelc"):
    snapshot_download(
        "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8",
        local_dir=bundle,
        allow_patterns=["audio_encoder.mlmodelc/*", "text_decoder.mlmodelc/*", "embed_tokens.fp16.bin",
                        "config.json", "vocab.json", "merges.txt", "tokenizer_config.json"],
    )

# 3. The PyTorch model. `from_pretrained` reads default sub-configs for this checkpoint (they sit under
#    `thinker_config`, and the encoder type is `qwen3_asr_audio_encoder`), so build the config by hand and
#    rename the weights.
snapshot = snapshot_download("Qwen/Qwen3-ForcedAligner-0.6B")
checkpoint_config = json.load(open(f"{snapshot}/config.json"))
thinker = checkpoint_config["thinker_config"]
model = Qwen3ASRForTokenClassification(Qwen3ASRConfig(
    audio_config=dict(thinker["audio_config"], model_type="qwen3_asr_encoder"),
    text_config=thinker["text_config"],
    audio_token_id=thinker["audio_token_id"],
    timestamp_token_id=checkpoint_config["timestamp_token_id"],
    num_labels=thinker["classify_num"],
)).float().eval()

weights = {}
for name, tensor in load_file(f"{snapshot}/model.safetensors").items():
    name = re.sub(r"^thinker\.audio_tower\.proj1\.", "model.multi_modal_projector.linear_1.", name)
    name = re.sub(r"^thinker\.audio_tower\.proj2\.", "model.multi_modal_projector.linear_2.", name)
    name = re.sub(r"^thinker\.audio_tower\.", "model.audio_tower.", name)
    name = re.sub(r"^thinker\.model\.", "model.language_model.", name)
    name = re.sub(r"^thinker\.lm_head\.", "score.", name)
    weights[name] = tensor.float()
missing, unused = model.load_state_dict(weights, strict=False)
assert not missing and not unused, (missing, unused)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B")

# 4. Reference mel features (transformers' WhisperFeatureExtractor) and audio embeddings.
features = processor.feature_extractor(audio, sampling_rate=16000, return_attention_mask=True, return_tensors="np")
frames = int(features["attention_mask"][0].sum())
mel = features["input_features"][0][:, :frames]

padded = int(np.ceil(frames / 100) * 100)  # the PyTorch encoder takes a multiple of 100 frames, with a mask
padded_mel = np.zeros((1, 128, padded), np.float32)
padded_mel[0, :, :frames] = mel
mask = np.zeros((1, padded), np.int64)
mask[0, :frames] = 1
with torch.no_grad():
    audio_features = model.model.get_audio_features(torch.tensor(padded_mel), torch.tensor(mask))
audio_embeddings = audio_features.pooler_output.reshape(-1, 1024).numpy()

# 5. speech-swift's prompt: chat prefix, one audio pad per audio embedding, suffix, then word<timestamp><timestamp>
#    per word. The audio embeddings replace the pads.
prefix = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
suffix = [151670, 151645, 198, 151644, 77091, 198]
audio_pad, timestamp = 151676, 151705
words = processor.split_words_for_alignment(text, "English")
slotted_words = processor.tokenizer.encode("".join(word + "<timestamp><timestamp>" for word in words),
                                           add_special_tokens=False)
ids = prefix + [audio_pad] * len(audio_embeddings) + suffix + slotted_words
slots = [i for i, token in enumerate(ids) if token == timestamp]

embeddings = model.model.language_model.embed_tokens.weight.detach().numpy()[ids].copy()
embeddings[len(prefix):len(prefix) + len(audio_embeddings)] = audio_embeddings

# 6. PyTorch timestamp classes at the slots: two per word (start, end), one class is 80 ms.
#    transformers' _fix_timestamps makes them run in order, as speech-swift's TimestampCorrection does.
with torch.no_grad():
    hidden = model.model.language_model(inputs_embeds=torch.tensor(embeddings)[None]).last_hidden_state
    reference_logits = model.score(hidden)[0].numpy()[slots]
reference = reference_logits.argmax(-1)
reference_starts = [f"{index * 0.08:.2f}" for index in _fix_timestamps(reference)[0::2]]

# 7. Swift, per compute-unit mode. With ALIGN_DEBUG=1, CoreMLForcedAligner prints its raw indices;
#    its normal output is one "start word" line per word, after TimestampCorrection.
swift_raw, swift_starts = {}, {}
for mode in ["default", "all", "cpu", "gpu"]:
    result = subprocess.run(
        [".build/release/aligncheck", bundle, "test_audio_16k.wav", text, mode],
        capture_output=True, text=True, check=True, env=dict(os.environ, ALIGN_DEBUG="1"),
    )
    raw_line = re.search(r"raw:\s*\[([^\]]*)\]", result.stdout + result.stderr).group(1)
    swift_raw[mode] = [int(index) for index in raw_line.split(",")]
    swift_starts[mode] = [line.split()[0] for line in result.stdout.splitlines() if not line.startswith("[")]

print("Raw timestamp indices, two per word (start, end), 80 ms each:")
print(f"  {'PyTorch':8} {reference.tolist()}")
for mode, raw in swift_raw.items():
    same = sum(swift == torch_index for swift, torch_index in zip(raw, reference))
    print(f"  {mode:8} {raw}  {same}/{len(reference)} equal to PyTorch")

print("\nWord starts in seconds, after putting the times in order:")
print(f"  {'PyTorch':8} {' '.join(reference_starts)}")
for mode, starts in swift_starts.items():
    print(f"  {mode:8} {' '.join(starts)}")

# 8. The bundle alone in coremltools, fed the reference mel and prompt, so only the compute unit varies.
#    Inputs are padded as speech-swift pads them: 3000 mel frames, 768 tokens.
mel_input = np.zeros((1, 128, 3000), np.float32)
mel_input[0, :, :frames] = mel
encoder_input = {"mel": mel_input, "mel_length": np.array([frames], np.int32)}
embeddings_input = np.zeros((1, 768, 1024), np.float32)
embeddings_input[0, :len(ids)] = embeddings
decoder_input = {"inputs_embeds": embeddings_input}

print(f"\nThe bundle alone in coremltools, with the reference inputs:\n  {'units':12} {'encoder error':>13} {'decoder error':>13} {'slots':>6} {'decoder ms':>10}")
for units in ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]:
    compute_units = getattr(ct.ComputeUnit, units)
    encoder = ct.models.CompiledMLModel(f"{bundle}/audio_encoder.mlmodelc", compute_units=compute_units)
    decoder = ct.models.CompiledMLModel(f"{bundle}/text_decoder.mlmodelc", compute_units=compute_units)

    encoded = encoder.predict(encoder_input)["audio_embeddings"][0][:len(audio_embeddings)].astype(np.float64)
    logits = decoder.predict(decoder_input)["logits"][0][slots].astype(np.float64)
    encoder_error = np.linalg.norm(encoded - audio_embeddings) / np.linalg.norm(audio_embeddings)
    decoder_error = np.linalg.norm(logits - reference_logits) / np.linalg.norm(reference_logits)
    same = int((logits.argmax(-1) == reference).sum())

    times = []
    for _ in range(5):
        start = time.perf_counter()
        decoder.predict(decoder_input)
        times.append(time.perf_counter() - start)
    milliseconds = statistics.median(times) * 1000

    print(f"  {units:12} {encoder_error:13.1%} {decoder_error:13.1%} {same:3}/{len(reference)} {milliseconds:10.0f}")
