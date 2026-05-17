"""
Standalone test for Moira.AI GreekTTS-1.5.

Verifies the full Orpheus + LoRA + SNAC pipeline works on your machine
before we commit to a full GUI rewrite. Bypasses Qt and threading entirely.

Run from your project folder:

    python3 -u test_moira.py

Each phase prints a timestamped line. The LAST line printed tells us
exactly where things stopped if anything fails. On success, saves
test_moira_output.wav next to the script - listen to it to judge whether
the quality is actually better than MMS-TTS for your use case.

First run downloads ~6.5 GB of model weights from Hugging Face. Make sure
you have disk space and reasonable internet before starting.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


T0 = time.monotonic()


def log(msg: str, indent: int = 0) -> None:
    """Print with elapsed-time prefix and immediate flush."""
    elapsed = time.monotonic() - T0
    prefix = "  " * indent
    print(f"[+{elapsed:6.1f}s] {prefix}{msg}", flush=True)


# Models
BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"
LORA_ADAPTERS = "moiralabs/GreekTTS-1.5"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"

# Special token IDs (from the GreekTTS-1.5 model card)
TOKENIZER_LENGTH = 128256
START_OF_HUMAN = 128259
END_OF_TEXT = 128009
END_OF_HUMAN = 128260
END_OF_AI = 128258
START_OF_SPEECH_MARKER = 128257
PAD_TOKEN = 128263
AUDIO_TOKEN_OFFSET = 128266

# Test sentence
TEST_TEXT = "Καλημέρα σας. Σας ενημερώνουμε ότι το ραντεβού σας έχει επιβεβαιωθεί."


def main() -> int:
    log("Test script starting.")

    # ─── Phase 1: imports ───
    log("Phase 1: importing torch...")
    try:
        import torch
        log(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", indent=1)
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            log(f"VRAM: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total", indent=1)
            if free < 5e9:
                log("WARNING: under 5 GB VRAM free. Orpheus 3B may not fit.", indent=1)
    except ImportError as e:
        log(f"FAIL: torch not installed: {e}")
        return 2

    log("Phase 1b: importing unsloth (heavy import - can take 10–30s)...")
    try:
        from unsloth import FastLanguageModel
        log("unsloth.FastLanguageModel imported", indent=1)
    except ImportError as e:
        log(f"FAIL: unsloth not installed.")
        log(f"Install: pip install unsloth", indent=1)
        log(f"Error: {e}", indent=1)
        return 3

    log("Phase 1c: importing transformers, peft, snac...")
    try:
        from transformers import AutoTokenizer
        from peft import PeftModel  # noqa: F401
        from snac import SNAC
        log("Other imports OK", indent=1)
    except ImportError as e:
        log(f"FAIL: missing dependency: {e}")
        log("Install: pip install transformers peft snac", indent=1)
        return 4

    # ─── Phase 2: load base Orpheus model ───
    log(f"Phase 2: loading base Orpheus model ({BASE_MODEL})...")
    log("Loading in 4-bit quantization to fit consumer GPUs (~2 GB VRAM).", indent=1)
    log("First run downloads ~2 GB of quantized weights. Subsequent runs use cache.", indent=1)
    try:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,   # was False; 8GB GPU can't hold fp16 + adapter
        )
        log("Base Orpheus model loaded in 4-bit", indent=1)
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            log(f"VRAM after base load: {(total-free)/1e9:.2f} GB used / {total/1e9:.2f} GB total", indent=1)
    except Exception as e:
        log(f"FAIL loading base model: {e}")
        traceback.print_exc()
        return 5

    # ─── Phase 3: apply Greek LoRA adapters ───
    # The adapter files live inside a `checkpoint-264000/` subfolder of the
    # repo, not at the repo root. We download just the two files we need
    # (config + weights, ~389 MB) and point load_adapter at the local path.
    log(f"Phase 3: downloading + loading Greek LoRA adapters ({LORA_ADAPTERS})...")
    log("First run downloads ~389 MB.", indent=1)
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            repo_id=LORA_ADAPTERS,
            allow_patterns=[
                "checkpoint-264000/adapter_config.json",
                "checkpoint-264000/adapter_model.safetensors",
            ],
        )
        adapter_path = Path(snapshot_path) / "checkpoint-264000"
        log(f"Adapter path: {adapter_path}", indent=1)

        model.load_adapter(str(adapter_path))
        log("Greek LoRA adapters applied", indent=1)
    except Exception as e:
        log(f"FAIL loading LoRA: {e}")
        traceback.print_exc()
        return 6

    log("Phase 3b: loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        log("Tokenizer loaded", indent=1)
    except Exception as e:
        log(f"FAIL loading tokenizer: {e}")
        return 7

    # ─── Phase 4: load SNAC neural codec ───
    log(f"Phase 4: loading SNAC vocoder ({SNAC_MODEL})...")
    log("First run downloads ~100 MB.", indent=1)
    try:
        snac = SNAC.from_pretrained(SNAC_MODEL).to("cpu")
        log("SNAC loaded on CPU (recommended per model card)", indent=1)
    except Exception as e:
        log(f"FAIL loading SNAC: {e}")
        return 8

    # ─── Phase 5: synthesis ───
    log("Phase 5: synthesizing test sentence...")
    log(f'Text: "{TEST_TEXT}"', indent=1)

    try:
        FastLanguageModel.for_inference(model)

        # Tokenize the text input
        input_ids = tokenizer(TEST_TEXT, return_tensors="pt").input_ids

        # Wrap with the GreekTTS-1.5 special-token framing:
        #   [Start-of-Human] <text tokens> [End-of-Text] [End-of-Human]
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        attention_mask = torch.ones_like(modified_input_ids)

        log("Generating audio tokens (autoregressive, may take 5–15s)...", indent=1)
        gen_start = time.monotonic()
        generated_ids = model.generate(
            input_ids=modified_input_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=END_OF_AI,
            use_cache=True,
        )
        gen_time = time.monotonic() - gen_start
        n_new = generated_ids.shape[1] - modified_input_ids.shape[1]
        log(f"Generated {n_new} new tokens in {gen_time:.1f}s", indent=1)

        # Crop to what comes after the start-of-speech marker
        token_indices = (generated_ids == START_OF_SPEECH_MARKER).nonzero(as_tuple=True)
        if len(token_indices[1]) > 0:
            last_idx = token_indices[1][-1].item()
            cropped = generated_ids[:, last_idx + 1 :]
        else:
            cropped = generated_ids

        # Strip the end-of-AI markers
        row = cropped[0]
        masked = row[row != END_OF_AI]
        row_length = masked.size(0)

        # Audio tokens come in groups of 7 (one per SNAC time-step)
        new_length = (row_length // 7) * 7
        if new_length == 0:
            log("FAIL: no audio tokens generated. Model may be misconfigured.")
            return 9

        # Subtract offset to get raw codec values
        trimmed = [int(t.item() - AUDIO_TOKEN_OFFSET) for t in masked[:new_length]]
        log(f"Trimmed to {new_length} audio tokens", indent=1)

        # Redistribute the flat token stream into SNAC's three layers
        # (the de-interleaving pattern is from the model card)
        layer_1, layer_2, layer_3 = [], [], []
        for i in range(new_length // 7):
            layer_1.append(trimmed[7 * i])
            layer_2.append(trimmed[7 * i + 1] - 4096)
            layer_3.append(trimmed[7 * i + 2] - 8192)
            layer_3.append(trimmed[7 * i + 3] - 12288)
            layer_2.append(trimmed[7 * i + 4] - 16384)
            layer_3.append(trimmed[7 * i + 5] - 20480)
            layer_3.append(trimmed[7 * i + 6] - 24576)

        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0),
        ]

        log("Decoding audio with SNAC...", indent=1)
        audio_hat = snac.decode(codes)
        audio = audio_hat.detach().squeeze().cpu().numpy()
        duration = audio.shape[0] / 24000.0
        log(f"Audio: {audio.shape[0]} samples at 24 kHz = {duration:.2f}s", indent=1)

    except Exception as e:
        log(f"FAIL during synthesis: {e}")
        traceback.print_exc()
        return 10

    # ─── Phase 6: save WAV ───
    out_path = Path(__file__).parent / "test_moira_output.wav"
    log(f"Phase 6: saving to {out_path}...")
    try:
        import wave
        import numpy as np

        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)

        with wave.open(str(out_path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(pcm.tobytes())
        log(f"Saved {out_path.stat().st_size / 1024:.1f} KB WAV", indent=1)
    except Exception as e:
        log(f"FAIL saving WAV: {e}")
        return 11

    log("=" * 50)
    log("ALL PHASES COMPLETED.")
    log(f"Listen to: {out_path}")
    log("If quality is good, tell me and I'll do the full GUI rewrite.")
    log("If quality is no better than MMS-TTS, we stay on MMS-TTS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
