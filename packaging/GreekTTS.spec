# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Greek TTS — Windows build.

Notes:
  - One-folder build (not one-file): startup is much faster, antivirus
    is much less likely to flag the result, and unsloth's runtime
    code generation needs a writable working directory anyway.
  - We avoid bundling Hugging Face model weights — they auto-download
    to the user's HF cache on first launch and are reused thereafter.
  - Bundling CUDA libraries: torch's CUDA .dll files are huge but
    required. They live under site-packages\\torch\\lib and are picked
    up automatically by PyInstaller's torch hook.
"""

import sys
from pathlib import Path

block_cipher = None

# Project root = where this spec file lives
PROJECT_ROOT = Path(SPECPATH).resolve()

# Hidden imports: modules PyInstaller can't find by static analysis.
# These are needed because the heavy ML libs use dynamic imports.
hidden_imports = [
    # PySide6 multimedia plugins
    "PySide6.QtMultimedia",
    # transformers + tokenizers — lots of dynamic loading
    "transformers",
    "transformers.models.llama",
    "transformers.models.llama.modeling_llama",
    "transformers.models.auto",
    "tokenizers",
    # unsloth and unsloth_zoo (the latter is imported indirectly)
    "unsloth",
    "unsloth_zoo",
    # peft — adapter loading uses runtime imports
    "peft",
    "peft.tuners.lora",
    # bitsandbytes — 4-bit quantization, has native .so/.dll
    "bitsandbytes",
    # snac — vocoder, small but uses dynamic config
    "snac",
    # huggingface_hub — cache + download
    "huggingface_hub",
    # safetensors — used by both transformers and peft
    "safetensors",
    "safetensors.torch",
]

# Data files to bundle alongside the executable.
# Empty list — config.json is created at runtime, models download to
# the user's HF cache, ffmpeg is expected on PATH.
datas = []

# Binaries — same approach: trust PyInstaller's auto-detection for torch/cuda.
binaries = []

a = Analysis(
    ["app.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Trim the build by excluding things we definitely don't use.
        # These are common PyInstaller false positives.
        "tkinter",
        "matplotlib",
        "scipy",  # not used; transformers may reference it conditionally
        "PIL",    # ditto
        "pytest",
        "IPython",
        "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GreekTTS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                # UPX often corrupts torch's CUDA DLLs — keep off
    console=False,            # GUI app — no console window on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                # add an .ico path here later if you make one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GreekTTS",
)
