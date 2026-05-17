# Greek TTS for Telephony

A desktop application that converts Greek text into telephony-ready audio
files (WAV) using **Moira.AI GreekTTS-1.5** - a Greek LoRA adapter on top of
the Orpheus 3B foundation model, with SNAC neural codec for audio decoding.

Designed for non-developer users in Greek call-center environments.
Type Greek text → click Generate → save a WAV that drops straight into
Asterisk, FreeSWITCH, 3CX, or any PBX.

---

## Table of contents

- [Features](#features)
- [Hardware requirements](#hardware-requirements)
- [Installation](#installation)
- [First launch](#first-launch)
- [Daily use](#daily-use)
- [Customizing the voice](#customizing-the-voice)
- [Output formats](#output-formats)
- [Using the generated files in Asterisk](#using-the-generated-files-in-asterisk)
- [Project layout](#project-layout)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Licensing](#licensing)
- [GDPR and Greek law for outbound calls](#gdpr-and-greek-law-for-outbound-calls)
- [Architecture & history](#architecture--history)

---

## Features

- **High-quality Greek output.** Trained specifically on a Greek single-speaker corpus, with natural prosody and accurate word stress.
- **Runs on-premise** after first launch. Three model components auto-download once, then run fully offline. No data leaves your machine after setup.
- **Simple interface.** Type Greek text → click Δημιουργία → save WAV. No reference voice or transcript management.
- **Greek-language UI** throughout, built for Greek operators.
- **Telephony-ready output.** A-law 8 kHz mono by default; μ-law and PCM also available.
- **Drop-in for any PBX** that plays standard WAV files.
- **Voice shaping controls.** Pitch and speed adjustments let you turn the single base voice into a small palette of variants.

---

## Hardware requirements

- **GPU:** NVIDIA with **4 GB+ VRAM** (the app uses 4-bit quantization to fit on consumer GPUs). Verified on RTX 2070 Super (8 GB).
- **CPU-only mode is NOT supported.** Synthesis would be impractically slow for the Orpheus 3B base model.
- **RAM:** 8 GB system RAM.
- **Disk:** ~3 GB for Python dependencies + ~2.5 GB for cached models = **~5.5 GB total**.

---

## Installation

### 1. System dependencies

Install **ffmpeg** (one-time, system-level):

- **Windows:** download from [ffmpeg.org](https://ffmpeg.org), unzip, add the `bin/` folder to your `PATH`.
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`

Verify it works in a terminal:
```bash
ffmpeg -version
```

### 2. Python environment

Python 3.10 or newer is required.

```bash
cd greek-tts-app
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# or:  .venv\Scripts\activate       # Windows
```

### 3. Install PyTorch (CUDA-matching version)

This is the one step that depends on your hardware. Pick the right command for your CUDA driver from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). Examples:

```bash
# NVIDIA GPU with CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# NVIDIA GPU with CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA is detected:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
This must print `True` before continuing.

### 4. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

If `bitsandbytes` is missing when the model loads, install it explicitly:
```bash
pip install bitsandbytes
```

---

## First launch

```bash
python app.py
```

What happens on the very first launch:

1. **Warm-up** in the terminal (~10–15 seconds). You'll see progress lines for torch, unsloth, transformers, and SNAC. This is normal and only this slow once.
2. **Model download dialog.** The app warns you that ~2.5 GB of model files need to download. Click OK.
3. **Background download** of three components from Hugging Face:
   - `unsloth/orpheus-3b-0.1-ft` (~2 GB - the slow one, expect 5–10 minutes on typical broadband)
   - `moiralabs/GreekTTS-1.5` (~389 MB)
   - `hubertsiuzdak/snac_24khz` (~80 MB)
4. **Ready.** The status bar shows "Έτοιμο. Μοντέλο φορτωμένο σε GPU (4-bit, 24000 Hz)."

The UI stays responsive throughout - synthesis runs on a background thread.

**Subsequent launches** use the cached models and reach the ready state in 30–60 seconds with no internet required.

If the first-launch download fails or stalls, see [Troubleshooting](#troubleshooting).

---

## Daily use

1. Open the app - wait for the status bar to say "Έτοιμο. Μοντέλο φορτωμένο…".
2. Type or paste Greek text into the main field.
3. Choose your output format (A-law 8 kHz mono is correct for most European telephony).
4. Click **Δημιουργία**. Synthesis takes about 5–15 seconds per sentence.
5. Click **▶** to preview the result. Click **■** to stop.
6. If you like it, click **Αποθήκευση…** and save to your desired location (typically your PBX's sounds folder).
7. If you don't like it (sampled generation occasionally has glitches), just click **Δημιουργία** again - the next attempt will be different.

Settings (default format, generation parameters, voice shaping) persist between sessions in `config.json`.

---

## Customizing the voice

GreekTTS-1.5 is a **single-speaker model** - there is no built-in male/female switch. The voice's identity is locked in by the LoRA training data.

However, you can shape the acoustic character of that one voice through three knobs in `config.json`. Combined, they give you a small palette of distinguishable variants. To use these, open `config.json` in any text editor while the app is closed, change the values, save, and relaunch.

### Pitch (`pitch_semitones`)

Range: `-12.0` to `+12.0`. Default: `0.0`.

Shifts the entire voice up or down in pitch. Most useful range is `±5` semitones. Beyond `±7` it starts to sound artificial because no formant correction is applied.

| Value | Effect |
|------:|--------|
| `+5.0` | Clearly higher voice, often reads as feminine |
| `+3.0` | Noticeably higher, softer character |
| `0.0`  | Original voice |
| `-3.0` | Slightly deeper, more authoritative |
| `-5.0` | Distinctly deeper voice |

### Speed (`speed`)

Range: `0.5` to `2.0`. Default: `1.0`. Pitch is preserved regardless of speed - only the speaking rate changes.

| Value | Effect |
|------:|--------|
| `0.85` | Slower, calmer, suitable for elderly listeners or formal announcements |
| `1.0`  | Original speed |
| `1.15` | Slightly faster, more urgent |
| `1.25` | Brisk, energetic |

### Generation parameters (`temperature`, `top_p`, `repetition_penalty`)

These affect the autoregressive generation process, not the voice itself. Defaults from the Moira.AI model card work well for most cases:

- `temperature: 0.6` - sampling randomness. Lower = more consistent but sometimes flatter. Try `0.4` if you're getting unstable output.
- `top_p: 0.95` - nucleus sampling. Rarely needs changing.
- `repetition_penalty: 1.1` - discourages the model from getting stuck. Don't lower below `1.0`.

### Voice prefix (experimental - usually does nothing)

`voice_prefix` lets you try injecting one of the Orpheus base voices' names (e.g., `"tara"`, `"leah"`, `"leo"`). In practice this has no audible effect on most prompts because the Greek LoRA was trained without these prefixes. Left in for completeness; set to `null` (the default) for normal operation.

### Reproducibility (`seed`)

By default each generation is different. Set `seed` to an integer (e.g., `42`) to make output reproducible - useful if you want to recreate an exact prompt later.

---

## Output formats

| Preset                | Use this for                                                    |
|-----------------------|-----------------------------------------------------------------|
| **A-law 8 kHz mono**  | European telephony - Asterisk, 3CX, OTE / Nova SIP trunks. **Default.** |
| μ-law 8 kHz mono      | North American telephony                                        |
| PCM 16-bit 8 kHz      | Generic 8 kHz playback, debugging                               |
| PCM 16-bit 16 kHz     | HD-voice systems / WebRTC                                       |

The native synthesis rate is 24 kHz. Downsampling and codec conversion happen in a single ffmpeg pass at the end of the pipeline.

---

## Using the generated files in Asterisk

Save the WAV files into Asterisk's sounds directory, for example:

```
/var/lib/asterisk/sounds/el/custom/reminder_intro.wav
```

Then play them from your dialplan:

```
exten => s,1,Answer()
 same => n,Playback(custom/reminder_intro)
 same => n,WaitExten(5)
```

A-law WAV files are recognized natively by Asterisk - no extra conversion required.

For FreeSWITCH, 3CX, or other PBXes, consult their documentation for the equivalent prompt-file path and playback verb. The output format is standard enough to drop in unchanged.

---

## Project layout

```
greek-tts-app/
  app.py             # entry point - `python app.py`
  ui.py              # PySide6 GUI (Greek-language)
  tts_engine.py      # Orpheus + LoRA + SNAC pipeline + ffmpeg post-processing
  config.py          # JSON-backed settings persistence
  config.json        # auto-created on first save (your settings live here)
  network.py         # pre-flight model-cache check (warns before silent hangs)
  requirements.txt   # Python dependencies (torch installed separately)
  output/            # (optional) suggested folder for saved WAVs
  README.md          # this file
```

---

## Configuration reference

`config.json` is created automatically the first time you change any setting. You can also create it by hand. All fields are optional - missing fields fall back to defaults.

```json
{
  "default_format": "alaw_8k",
  "temperature": 0.6,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "max_new_tokens": 1200,
  "seed": null,
  "voice_prefix": null,
  "pitch_semitones": 0.0,
  "speed": 1.0
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `default_format` | string | `"alaw_8k"` | One of `alaw_8k`, `ulaw_8k`, `pcm16_8k`, `pcm16_16k` |
| `temperature` | float | `0.6` | Generation sampling temperature |
| `top_p` | float | `0.95` | Nucleus sampling cutoff |
| `repetition_penalty` | float | `1.1` | Discourages repetition; keep ≥ 1.0 |
| `max_new_tokens` | int | `1200` | ~8–10 seconds of audio max per generation |
| `seed` | int or null | `null` | Set to an integer for reproducible output |
| `voice_prefix` | string or null | `null` | Experimental; usually no effect |
| `pitch_semitones` | float | `0.0` | Useful range ±5 |
| `speed` | float | `1.0` | 0.85 = slower, 1.15 = faster |

---

## Troubleshooting

### "Δεν βρέθηκε το ffmpeg" / "ffmpeg not found on PATH"

ffmpeg isn't installed or isn't reachable. Install it system-wide (see [Installation](#installation) step 1) and confirm `ffmpeg -version` works in a terminal before relaunching.

### "Το Moira GreekTTS-1.5 απαιτεί GPU"

Your PyTorch install doesn't see CUDA. Run:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If it prints `False`, reinstall PyTorch with the correct CUDA wheel (see Installation step 3).

### "CUDA out of memory" during model load

Another process is using your GPU. Close other GPU-using applications and try again. If you have less than 4 GB free VRAM available, this model won't fit on your hardware.

### Model download is slow or fails

Pre-download the three components manually with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf download unsloth/orpheus-3b-0.1-ft
hf download moiralabs/GreekTTS-1.5
hf download hubertsiuzdak/snac_24khz
```

These cache to `~/.cache/huggingface/hub/` on Linux/macOS or `%USERPROFILE%\.cache\huggingface\hub\` on Windows. Once cached, the app will skip the download step.

For air-gapped machines: run those commands on a connected machine and copy the entire `~/.cache/huggingface/` folder to the target.

### App hangs on "Φόρτωση μοντέλου…" forever

Run from a terminal - never double-click:
```bash
python -u app.py
```
The terminal will show what's actually happening (download progress, errors, stack traces). The progressive timeout messages in the status bar will also update at 30s, 2min, and 10min.

### Output sounds noisy, has glitches, or is partially garbled

Sampled generation is non-deterministic - sometimes a particular run goes wrong. Click **Δημιουργία** again; the next attempt will be different.

If glitches happen on most attempts, try lowering `temperature` in `config.json` from `0.6` to `0.4` for more conservative sampling.

### Output sounds muffled or thin on the phone but fine on the computer

This is a fundamental property of telephony, not the model. The 8 kHz / a-law encoding strips frequencies above ~3.4 kHz. Audio that sounds clean on headphones can sound thin through a phone, and vice versa. Always evaluate on a real phone before judging quality.

### "Λείπει εξάρτηση Python"

A Python dependency is missing. The error message includes the specific missing package - install it with pip and try again. The most common one is `bitsandbytes`, which sometimes needs explicit installation:
```bash
pip install bitsandbytes
```

---

## Licensing

- **Greek LoRA adapter** (moiralabs/GreekTTS-1.5): Apache 2.0
- **Orpheus base model** (unsloth/orpheus-3b-0.1-ft): inherits Llama 3 Community License from Meta's Llama-3.2-3B
- **SNAC vocoder** (hubertsiuzdak/snac_24khz): MIT

The Llama 3 Community License has its own conditions (most notably an acceptable use policy and a 700M monthly-active-user threshold that triggers commercial-use restrictions). For government and most commercial use this is generally workable, but verify the current license text on the Meta and Hugging Face model cards before deploying. Consult counsel if your deployment context is sensitive.

---

## GDPR and Greek law for outbound calls

This tool only generates audio. The legal obligations around using that audio in outbound calls - particularly automated calls to consumers - are entirely the responsibility of the calling system and your operational policies. **Make sure the following are in place before you start dialing:**

- **Documented consent** from each recipient (Law 3471/2006 + GDPR). Automated calls without prior consent are a hard violation.
- **Article 11 opt-out registry suppression** (Μητρώο του άρθρου 11). Cross-reference every number before dialing.
- **Time-of-day enforcement.** Typically no calls before 09:00 or after 20:00 local time, and not on Sundays or public holidays. Specific sector rules may further restrict.
- **In-call opt-out option.** For example: "πατήστε 9 για να μην ξαναλάβετε τέτοιες κλήσεις."
- **Audit logs** for consent, attempts, opt-outs, and timestamps. The HDPA (Ελληνική Αρχή Προστασίας Δεδομένων Προσωπικού Χαρακτήρα) fines for violations are not symbolic.

If you're calling on behalf of a regulated sector (banking, health, debt collection, telecommunications) additional sector-specific rules apply on top of the above.

---

## Architecture & history

### Current architecture

```
  ┌──────────────┐    ┌────────────────────────────────┐   ┌────────┐
  │ Greek text   │───▶│ Orpheus 3B (4-bit) + Greek LoRA │──▶│ SNAC   │──▶ 24 kHz WAV
  │ (UI input)   │    │ (autoregressive token gen)     │   │ decoder│       │
  └──────────────┘    └────────────────────────────────┘   └────────┘       │
                                                                            ▼
                                                                  ┌──────────────┐
                                                                  │ ffmpeg       │
                                                                  │ pitch+speed  │──▶ Final WAV
                                                                  │ + telephony  │   (alaw/ulaw/pcm)
                                                                  │   format     │
                                                                  └──────────────┘
```

Synthesis runs on a Qt worker thread. The UI never freezes. Model loading also runs on a worker thread, with pre-flight cache checking and progressive timeout messages to surface common failures (network blocks, slow downloads, missing dependencies) before they become silent hangs.

### How we got here

The model selection went through several iterations, each ruled out for a concrete reason. Recording these is useful so future-you doesn't redo the work:

- **Piper (rapunzelina-low).** Free, fast, CPU-friendly, but only one Greek voice exists at the "low" quality tier. Autoregressive decoder regularly got stuck on phonemes and produced repetition glitches. Hit the quality ceiling immediately.
- **F5-TTS-Greek (PetrosStav).** Better architecture (flow-matching, no get-stuck failures) but required a reference voice + matching transcript. After fighting silent hangs (vocoder download, Qt threading, worker-thread CUDA init, GC of worker objects), got it working - but final output was gibberish, almost certainly due to subtle reference-clip / transcript mismatch issues.
- **MMS-TTS (facebook/mms-tts-ell).** Robust VITS architecture, no reference clip needed, very simple stack (just transformers). Single fixed voice but stable and intelligible. Works perfectly. **The simpler app you can fall back to if Moira ever breaks.**
- **Moira GreekTTS-1.5.** Current selection. The most natural-sounding Greek TTS we found that's self-hosted. Heavier stack (Orpheus 3B + LoRA + SNAC) and requires a 4 GB+ GPU, but the quality justifies it for production prompts. 4-bit quantization makes it fit on consumer GPUs.

### Trade-offs that remain

The honest list of what this stack does *not* do:

- **Only one voice.** Pitch/speed shaping gives you variants but not a fundamentally different person. For multiple voices, consider hiring multiple voice actors for fixed prompt sets, or layering MMS-TTS alongside for a second voice.
- **No real-time / streaming.** Each synthesis is a one-shot ~5–15 second wait. Fine for pre-generating IVR prompts, not suitable for interactive low-latency voice agents.
- **Non-commercial-clean isn't guaranteed.** The Llama 3 license is workable for most cases but you should verify for yours.
- **Long prompts must be chunked.** The 1200-token output cap is ~10 seconds. Longer prompts need to be split and concatenated.

### When to consider alternatives

- **You need broadcast-grade quality.** Hire a Greek voice actor - €200–500 for an afternoon session, full commercial rights, broadcast quality, zero technical dependencies. For fixed prompt sets this is hard to beat.
- **You need an officially commercially-licensed on-prem solution.** Microsoft Azure Speech containers (paid, government-friendly).
- **You need multiple voices.** Same as above - multiple voice actors, or wait for a future multi-speaker Greek open model.

---

## Credits

- **Voice model:** [Moira.AI GreekTTS-1.5](https://huggingface.co/moiralabs/GreekTTS-1.5) by Moira.AI
- **Base model:** [Orpheus 3B](https://huggingface.co/unsloth/orpheus-3b-0.1-ft) (unsloth quantization)
- **Vocoder:** [SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz) by Hubert Siuzdak
- **GUI framework:** PySide6 (Qt for Python)
- **Audio post-processing:** ffmpeg