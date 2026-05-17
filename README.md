# Greek TTS for Telephony - Moira GreekTTS-1.5 edition


[![Latest release](https://img.shields.io/github/v/release/Lefteris-B/GreekTTS_for_telephone_centers?label=latest%20release)](https://github.com/Lefteris-B/GreekTTS_for_telephone_centers/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/Lefteris-B/GreekTTS_for_telephone_centers/total?label=downloads)](https://github.com/Lefteris-B/GreekTTS_for_telephone_centers/releases)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue?logo=windows&logoColor=white)](INSTALL-WINDOWS.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://www.apache.org/licenses/LICENSE-2.0)



A desktop application that converts Greek text into telephony-ready audio
files (WAV) using **Moira.AI GreekTTS-1.5** - a Greek LoRA adapter on top of
the Orpheus 3B foundation model, with SNAC neural codec for audio decoding.

- **High-quality Greek output.** Trained specifically on Greek speech, with natural prosody and accent on word stress.
- **Runs on-premise** after first launch. Three model components auto-download once, then run offline.
- **Simple interface.** Type Greek text → click Generate → save WAV. No reference voice or transcripts to manage.
- **Greek-language UI.**
- **Telephony-ready output.** A-law 8 kHz mono by default; μ-law and PCM also available.
- **Drop-in for Asterisk / FreeSWITCH / 3CX / any PBX.**

---

## Hardware requirements

- **GPU:** NVIDIA with **4 GB+ VRAM** (uses 4-bit quantization). Tested on RTX 2070 Super (8 GB).
- **CPU-only mode is NOT supported.** Synthesis would be impractically slow.
- **RAM:** 8 GB system RAM.
- **Disk:** ~3 GB for Python deps + ~2.5 GB for models = **~5.5 GB total**.

---

## Setup

### 1. System dependencies

Install **ffmpeg** (one-time, system-level):

- **Windows:** download from [ffmpeg.org](https://ffmpeg.org), unzip, add the `bin/` folder to your `PATH`.
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`

### 2. Python environment

```bash
cd greek-tts-app
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# or:  .venv\Scripts\activate       # Windows
```

### 3. Install PyTorch (CUDA-matching version)

Pick the right command for your CUDA driver from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). Example:

```bash
# NVIDIA GPU with CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

If `bitsandbytes` complains during model load:
```bash
pip install bitsandbytes
```

### 5. Run

```bash
python app.py
```

**First launch will:**
1. Warm up PyTorch + unsloth (~10–15 seconds, you'll see progress in the terminal).
2. Show a dialog noting that ~2.5 GB of model components need to download. Click OK.
3. Download the three model files automatically:
   - `unsloth/orpheus-3b-0.1-ft` (~2 GB, the slow one - 5–10 min on typical broadband)
   - `moiralabs/GreekTTS-1.5` (~389 MB)
   - `hubertsiuzdak/snac_24khz` (~80 MB)
4. Show "Έτοιμο. Μοντέλο φορτωμένο σε GPU…" in the status bar when ready.

**Subsequent launches** use the cached models and are ready in 30–60 seconds with no internet required.

**Synthesis** takes about 5–15 seconds per sentence on an 8 GB GPU. First synthesis after launch is slower (PyTorch JIT warm-up); subsequent generations of similar length are faster.

---

## Output formats

| Preset                | Use this for                                    |
|-----------------------|-------------------------------------------------|
| **A-law 8 kHz mono**  | European telephony (default - Asterisk, 3CX, OTE/Nova SIP trunks) |
| μ-law 8 kHz mono      | North American telephony                        |
| PCM 16-bit 8 kHz      | Generic 8 kHz playback / debugging              |
| PCM 16-bit 16 kHz     | HD-voice systems / WebRTC                       |

---

## Using the generated files in Asterisk

```
/var/lib/asterisk/sounds/el/custom/reminder_intro.wav
```

Play from your dialplan:

```
exten => s,1,Answer()
 same => n,Playback(custom/reminder_intro)
 same => n,WaitExten(5)
```

---

## Project layout

```
greek-tts-app/
  app.py             # entry point - `python app.py`
  ui.py              # PySide6 GUI
  tts_engine.py      # Orpheus + LoRA + SNAC pipeline + ffmpeg post-processing
  config.py          # JSON-backed settings
  config.json        # auto-created on first save
  network.py         # pre-flight model-cache check
  requirements.txt   # Python dependencies (torch installed separately)
  output/            # (optional) suggested folder for saved WAVs
  README.md
```

---

## Tips

- **Sampling-based generation.** Output varies slightly between runs even for the same input. If a particular generation has artifacts, try clicking Generate again. To get reproducible output, set a `seed` value in `config.json`.
- **Keep prompts under ~10 seconds.** The model's `max_new_tokens=1200` cap roughly translates to 8–10 seconds of audio. For longer prompts, split into multiple sentences and concatenate WAVs.
- **For telephony, evaluate on a real phone.** 8 kHz a-law strips a lot of high-frequency content. Audio that sounds great on headphones may sound thin on a phone.

---

## Troubleshooting

**"Δεν βρέθηκε το ffmpeg"**
Install ffmpeg system-wide and confirm `ffmpeg -version` works in a terminal.

**"Το Moira GreekTTS-1.5 απαιτεί GPU"**
Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`. If `False`, your PyTorch was installed without CUDA support - reinstall using the CUDA wheel command above.

**"CUDA out of memory" during load**
You may have other GPU processes running. Close them and try again. If you have <4 GB free VRAM, this model won't fit - fall back to the MMS-TTS edition.

**Model download is slow / fails**
Pre-download the three components manually:
```bash
pip install -U "huggingface_hub[cli]"
hf download unsloth/orpheus-3b-0.1-ft
hf download moiralabs/GreekTTS-1.5
hf download hubertsiuzdak/snac_24khz
```

**Output is noisy or has glitches**
Try clicking Generate again - sampled generation is non-deterministic. If consistent across multiple attempts, try lowering `temperature` in `config.json` from 0.6 to 0.4.

**App hangs at "Φόρτωση μοντέλου…" forever**
Run from terminal: `python -u app.py`. The console will show what's happening (download progress, errors, etc.). Send the terminal output for diagnosis.

---
# Licensing & attribution

This application bundles or relies on several third-party components, each under its own license. The summary below reflects what was published on the source repositories at the time of writing - verify the current license text on each linked page before deployment.

## Models

- Moira.AI GreekTTS-1.5 (the Greek LoRA adapter) - licensed under Apache License 2.0. See the model card. Cite as @misc{moira2025greektts15} per the citation block on the model page.
- Orpheus 3B (canopylabs/orpheus-3b-0.1-ft) - the base TTS LLM, licensed under Apache License 2.0. See the model card. The card carries an explicit Model Misuse clause prohibiting impersonation without consent, misinformation or deception (including fake news or fraudulent calls), and any illegal or harmful activity. This is directly relevant for telephony deployments - read it before using this app in production.
- unsloth/orpheus-3b-0.1-ft - the 4-bit quantized variant we actually load. Same Apache 2.0 license inherited from canopylabs. See the unsloth fork.
- Meta Llama 3.2 3B - Orpheus is a fine-tune of Llama 3.2 3B Instruct, so the Llama 3.2 Community License flows through to anything derived from it, including this app. Key points: free commercial use is granted royalty-free, BUT (1) you must comply with Meta's Acceptable Use Policy which forbids deceptive/fraudulent practices, (2) you must include the attribution string "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." in distributions, and (3) organizations whose products exceed 700 million monthly active users must obtain a separate license from Meta. The MAU threshold is irrelevant for typical Greek call-center deployments but worth knowing exists.
- SNAC vocoder (hubertsiuzdak/snac_24khz) - licensed under MIT License. See the model card and the SNAC paper.

### Runtime libraries
The major Python packages this app depends on:

- PyTorch - BSD-3-Clause
- Transformers (Hugging Face) - Apache License 2.0
- Unsloth - Apache License 2.0
- PEFT (Hugging Face) - Apache License 2.0
- bitsandbytes - MIT License
- snac (the Python package, distinct from the model weights) - MIT License
- huggingface_hub - Apache License 2.0
- PySide6 (Qt for Python) - LGPL v3, with the standard LGPL re-linking obligations
- ffmpeg - separately installed by the user; ffmpeg itself is dual-licensed LGPL v2.1+ / GPL v2+ depending on which features are compiled in. This app calls ffmpeg as an external process and does not redistribute it.
---

