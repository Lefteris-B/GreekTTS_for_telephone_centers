"""
TTS engine: Moira.AI GreekTTS-1.5 pipeline.

Loads three components:
  1. unsloth/orpheus-3b-0.1-ft   — base Orpheus LLM (loaded in 4-bit)
  2. moiralabs/GreekTTS-1.5      — Greek LoRA adapter on top of Orpheus
  3. hubertsiuzdak/snac_24khz    — SNAC neural codec (audio decoder)

Synthesis pipeline:
  text → tokenize → wrap with special tokens → autoregressive generation
       → strip framing → group into 7-token frames → SNAC decode
       → 24 kHz waveform → ffmpeg → telephony format

Telephony output formats:
  - alaw_8k:   G.711 A-law,  8 kHz mono   (European telephony default)
  - ulaw_8k:   G.711 μ-law,  8 kHz mono   (North American default)
  - pcm16_8k:  Linear PCM,  16-bit, 8 kHz mono
  - pcm16_16k: Linear PCM,  16-bit, 16 kHz mono (HD voice / WebRTC)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


# Model repos
BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"
LORA_REPO = "moiralabs/GreekTTS-1.5"
LORA_SUBFOLDER = "checkpoint-264000"
SNAC_REPO = "hubertsiuzdak/snac_24khz"

# Files inside LORA_SUBFOLDER that are needed for inference (skip optimizer state)
LORA_FILES = [
    f"{LORA_SUBFOLDER}/adapter_config.json",
    f"{LORA_SUBFOLDER}/adapter_model.safetensors",
]

# Special token IDs (from the GreekTTS-1.5 model card)
START_OF_HUMAN = 128259
END_OF_TEXT = 128009
END_OF_HUMAN = 128260
END_OF_AI = 128258
START_OF_SPEECH_MARKER = 128257
AUDIO_TOKEN_OFFSET = 128266
SNAC_OUTPUT_RATE = 24000

# Telephony format presets — passed straight to ffmpeg.
FORMATS = {
    "alaw_8k":   {"label": "A-law 8 kHz mono (European telephony)",       "ar": "8000",  "codec": "pcm_alaw"},
    "ulaw_8k":   {"label": "μ-law 8 kHz mono (North American telephony)", "ar": "8000",  "codec": "pcm_mulaw"},
    "pcm16_8k":  {"label": "Linear PCM 16-bit 8 kHz mono",                "ar": "8000",  "codec": "pcm_s16le"},
    "pcm16_16k": {"label": "Linear PCM 16-bit 16 kHz mono (HD voice)",    "ar": "16000", "codec": "pcm_s16le"},
}

DEFAULT_FORMAT = "alaw_8k"


class TTSEngineError(RuntimeError):
    """Raised for any error in the synthesis pipeline."""


class GreekTTSEngine:
    """
    Loads Orpheus + Greek LoRA + SNAC once and reuses across calls.

    Thread safety: PyTorch model inference is generally safe for read-only
    inference sharing, but the underlying KV cache and SNAC decoder hold
    mutable state. Callers should serialize calls to .synthesize().
    """

    def __init__(self, use_cuda: bool = True) -> None:
        if not shutil.which("ffmpeg"):
            raise TTSEngineError(
                "Δεν βρέθηκε το ffmpeg στο PATH. Εγκαταστήστε το και προσπαθήστε ξανά.\n"
                "ffmpeg not found on PATH. Install it and try again.\n"
                "  Windows: download from https://ffmpeg.org and add to PATH\n"
                "  macOS:   brew install ffmpeg\n"
                "  Ubuntu:  sudo apt install ffmpeg"
            )

        # Defer heavy imports so the GUI can launch even if these libs are
        # broken — surface the error from the worker thread, not at app start.
        try:
            import torch
            from unsloth import FastLanguageModel
            from transformers import AutoTokenizer
            from snac import SNAC
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise TTSEngineError(
                "Λείπει εξάρτηση Python. Ελέγξτε τις οδηγίες στο README.\n"
                f"Missing Python dependency: {e}\n"
                "Required: torch, unsloth, transformers, peft, snac, huggingface_hub."
            ) from e

        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
        if not use_cuda:
            raise TTSEngineError(
                "Το Moira GreekTTS-1.5 απαιτεί GPU. CPU-only λειτουργία δεν υποστηρίζεται.\n"
                "Moira GreekTTS-1.5 requires a CUDA GPU. CPU-only mode not supported."
            )

        # ─── Load base Orpheus in 4-bit ───
        # 4-bit fits in ~2 GB VRAM, vs 6 GB for fp16. Required on 8 GB GPUs.
        try:
            model, _ = FastLanguageModel.from_pretrained(
                model_name=BASE_MODEL,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
        except Exception as e:
            raise TTSEngineError(
                f"Αποτυχία φόρτωσης Orpheus base: {e}\n"
                f"Failed to load Orpheus base model: {e}"
            ) from e

        # ─── Download + apply Greek LoRA from the checkpoint subfolder ───
        try:
            snapshot_path = snapshot_download(
                repo_id=LORA_REPO,
                allow_patterns=LORA_FILES,
            )
            adapter_path = Path(snapshot_path) / LORA_SUBFOLDER
            model.load_adapter(str(adapter_path))
        except Exception as e:
            raise TTSEngineError(
                f"Αποτυχία φόρτωσης Greek LoRA adapter: {e}\n"
                f"Failed to load Greek LoRA adapter: {e}"
            ) from e

        # Switch model to inference mode (Unsloth's optimized path)
        FastLanguageModel.for_inference(model)

        # ─── Tokenizer (from base model — has the speech special tokens) ───
        try:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        except Exception as e:
            raise TTSEngineError(f"Failed to load tokenizer: {e}") from e

        # ─── SNAC vocoder (model card recommends CPU placement) ───
        try:
            snac = SNAC.from_pretrained(SNAC_REPO).to("cpu")
        except Exception as e:
            raise TTSEngineError(f"Failed to load SNAC vocoder: {e}") from e

        # Cache references for use in synthesize()
        self._torch = torch
        self._unsloth_fast = FastLanguageModel
        self._model = model
        self._tokenizer = tokenizer
        self._snac = snac
        self.device = "cuda"
        self.sample_rate = SNAC_OUTPUT_RATE

    # ----- public API -----

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        fmt: str = DEFAULT_FORMAT,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1200,
        seed: int | None = None,
    ) -> Path:
        """
        Synthesize Greek text and write a telephony-ready WAV to output_path.
        Returns output_path on success, raises TTSEngineError on failure.

        Generation is sampled (non-deterministic). Pass `seed` to reproduce
        a specific run. `max_new_tokens=1200` corresponds to roughly 8–10
        seconds of audio — long enough for typical IVR prompts.
        """
        if not text.strip():
            raise TTSEngineError("Δεν υπάρχει κείμενο προς μετατροπή.")
        if fmt not in FORMATS:
            raise TTSEngineError(f"Άγνωστο φορμά '{fmt}'. Έγκυρα: {list(FORMATS)}")

        output_path = Path(output_path)

        tmp_raw = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            try:
                self._synthesize_to_wav(
                    text=text,
                    out_path=tmp_raw,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
            except TTSEngineError:
                raise
            except Exception as e:
                raise TTSEngineError(
                    f"Αποτυχία μετατροπής κειμένου σε ομιλία: {e}\n"
                    f"Synthesis failed: {e}"
                ) from e

            self._convert(tmp_raw, output_path, fmt)
        finally:
            tmp_raw.unlink(missing_ok=True)

        return output_path

    # ----- internals -----

    def _synthesize_to_wav(
        self,
        text: str,
        out_path: Path,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
        seed: int | None,
    ) -> None:
        torch = self._torch

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Tokenize input text and wrap with the GreekTTS framing tokens:
        #   [Start-of-Human] <text> [End-of-Text] [End-of-Human]
        input_ids = self._tokenizer(text, return_tensors="pt").input_ids
        start_t = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_t = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        wrapped = torch.cat([start_t, input_ids, end_t], dim=1)
        attn_mask = torch.ones_like(wrapped)

        # Generate audio tokens autoregressively
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=wrapped.to(self.device),
                attention_mask=attn_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=END_OF_AI,
                use_cache=True,
            )

        # Crop everything before (and including) the start-of-speech marker
        marker_indices = (generated_ids == START_OF_SPEECH_MARKER).nonzero(as_tuple=True)
        if len(marker_indices[1]) > 0:
            last_idx = marker_indices[1][-1].item()
            cropped = generated_ids[:, last_idx + 1:]
        else:
            cropped = generated_ids

        # Strip end-of-AI tokens
        row = cropped[0]
        masked = row[row != END_OF_AI]
        n_tokens = (masked.size(0) // 7) * 7
        if n_tokens == 0:
            raise TTSEngineError(
                "Το μοντέλο δεν παρήγαγε ηχητικά tokens. Δοκιμάστε διαφορετικό κείμενο "
                "ή ξανατρέξτε (η γεννήτρια χρησιμοποιεί δειγματοληψία)."
            )

        # Subtract offset and de-interleave into SNAC's three layers.
        # The 7-tokens-per-frame pattern is fixed by SNAC's codec structure
        # and the way GreekTTS was trained.
        trimmed = [int(t.item() - AUDIO_TOKEN_OFFSET) for t in masked[:n_tokens]]
        layer_1, layer_2, layer_3 = [], [], []
        for i in range(n_tokens // 7):
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

        # Decode codes → waveform via SNAC (on CPU)
        with torch.no_grad():
            audio_hat = self._snac.decode(codes)

        audio = audio_hat.detach().squeeze().cpu().float().numpy().clip(-1.0, 1.0)
        pcm = (audio * 32767.0).astype("int16")

        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())

    @staticmethod
    def _convert(src: Path, dst: Path, fmt: str) -> None:
        spec = FORMATS[fmt]
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ar", spec["ar"],
            "-ac", "1",
            "-acodec", spec["codec"],
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise TTSEngineError(
                f"ffmpeg conversion failed:\n{result.stderr.strip()}"
            )
