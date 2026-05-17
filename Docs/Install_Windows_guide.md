# Greek TTS — Windows installation

This guide is for end users who downloaded the `GreekTTS-windows-*.zip`
release. If you're a developer building from source, see the main README.

---

## Requirements

- **Windows 10 or 11, 64-bit.**
- **NVIDIA GPU with 4 GB+ VRAM** and a recent driver (release 525+, October 2022 or later).
- **Internet connection on first launch** to download the ~2.5 GB of model files. After that, the app runs fully offline.
- **~10 GB free disk space** (5 GB for the app, 2.5 GB for models, plus working space).
- **ffmpeg** installed and on your system PATH.

### Installing ffmpeg on Windows

1. Download the latest Windows build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html). The "essentials" build from gyan.dev or BtbN is fine.
2. Extract the zip to a permanent location, e.g. `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter.
   - Tab "Advanced" → button "Environment Variables".
   - Under "System variables", find `Path`, click "Edit", click "New", paste `C:\ffmpeg\bin`, click OK.
4. Open a fresh Command Prompt and run `ffmpeg -version` to confirm.

---

## Installation

### The easy way (recommended)

1. **Download** `GreekTTS-windows-*.zip` from the [Releases page](../../releases).
2. **Extract** the zip to a permanent folder — for example `C:\GreekTTS`. Avoid Downloads or Desktop.
3. **Double-click `setup.cmd`** at the top of the extracted folder.
   - The script installs ffmpeg automatically, installs Python if needed, and downloads the ~2.5 GB of model files.
   - Total time: 10–20 minutes depending on your network.
   - If something fails partway, just run `setup.cmd` again — it resumes from where it stopped.
4. When setup is done, **double-click `GreekTTS\GreekTTS.exe`** to launch the app.

That's it. The first launch will be ready immediately because everything is already installed.

### The manual way

If `setup.cmd` doesn't work for your environment (locked-down corporate machine, no winget, no internet on the target machine):

1. **Install ffmpeg manually:**
   - Download the latest Windows build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html). The "essentials" build from gyan.dev or BtbN is fine.
   - Extract to a permanent location, e.g. `C:\ffmpeg`.
   - Add `C:\ffmpeg\bin` to your system PATH:
     - `Win + R` → type `sysdm.cpl` → Enter.
     - "Advanced" tab → "Environment Variables".
     - Under "System variables", find `Path`, click "Edit", "New", paste `C:\ffmpeg\bin`, OK.
   - Open a fresh Command Prompt and run `ffmpeg -version` to confirm.

2. **Launch `GreekTTS.exe`.** The first time, it will show a dialog asking to download ~2.5 GB of model files. Click OK and wait 5–15 minutes.

3. For air-gapped machines, the models live in `%USERPROFILE%\.cache\huggingface\hub\`. Copy that folder from a connected machine to the same path on the target.

---

## First launch

The first time you run `GreekTTS.exe`:

1. The window appears within a few seconds.
2. A dialog warns that ~2.5 GB of model files need to download from Hugging Face. Click **OK**.
3. Download takes 5–15 minutes depending on your internet speed. The status bar shows progress.
4. When complete, the status bar shows **"Έτοιμο. Μοντέλο φορτωμένο σε GPU…"** — you're ready to use the app.

Subsequent launches skip the download and start within 30–60 seconds.

---

## Daily use

1. Open the app — wait for "Έτοιμο. Μοντέλο φορτωμένο…".
2. Type or paste Greek text in the main field.
3. Choose your output format (A-law 8 kHz mono is correct for most Greek telephony setups).
4. Click **Δημιουργία**. Synthesis takes 5–15 seconds per sentence.
5. Click **▶** to listen, then **Αποθήκευση…** to save the WAV file.

---

## Troubleshooting

**"This app can't run on your PC"**
Confirm you downloaded the 64-bit Windows zip and your system is 64-bit Windows 10/11.

**Antivirus quarantines `GreekTTS.exe`**
PyInstaller bundles trigger heuristic detection on some antivirus tools. The exe is unsigned (a code-signing certificate costs €200+/year). Either whitelist the folder or build from source. The full source is published in the repository.

**"CUDA is not available" or "Το Moira GreekTTS-1.5 απαιτεί GPU"**
- Update your NVIDIA driver to the latest version from [nvidia.com/drivers](https://www.nvidia.com/drivers).
- Verify your GPU is NVIDIA. AMD and Intel GPUs are not supported by this build.
- Open Command Prompt and run `nvidia-smi`. It must report your GPU and a CUDA version of 12.0 or higher.

**App hangs at "Φόρτωση μοντέλου…"**
The model download stalled. Most common cause: corporate firewall blocking `huggingface.co`. Either move to an unrestricted network for first launch, or pre-download the model files manually (see main README).

**Synthesis output sounds wrong**
- Sometimes the model has a bad generation. Click **Δημιουργία** again — output is non-deterministic.
- Audio sounding thin or muffled on a phone is normal — telephony's 8 kHz / A-law encoding strips high frequencies. Evaluate on a real phone, not headphones.

**Where does the app store data?**

- **App folder:** wherever you extracted the zip. Contains `GreekTTS.exe` and bundled libraries.
- **`config.json`:** in the same folder as the exe — your settings (default format, pitch, speed, etc.).
- **Model cache:** `C:\Users\<you>\.cache\huggingface\hub\`. About 2.5 GB after first launch.

To completely uninstall: delete the app folder and (optionally) the Hugging Face cache folder.

---

## Updating

Download the new release zip, extract it over your existing installation (or to a new folder), and run. The model cache and your `config.json` are preserved if you extract to the same folder.

---

## Licensing

See LICENSE and the main README for details on the open-source licenses
covering the Greek LoRA adapter, the Orpheus base model, and the SNAC vocoder.