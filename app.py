"""
Greek TTS for Telephony — desktop application (Moira GreekTTS-1.5 edition).

Run:    python app.py

No setup required beyond the install steps in README.md. Three model
components (~2.5 GB total) are auto-downloaded from Hugging Face on first
launch and cached locally for offline use afterwards.

Hardware: requires NVIDIA GPU with 4 GB+ VRAM (uses 4-bit quantization
to fit on consumer GPUs like the RTX 2060/2070/3060/4060).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui import MainWindow


APP_DIR = Path(__file__).resolve().parent


def _warm_up_torch() -> None:
    """
    Pre-initialize PyTorch, unsloth, and transformers on the main thread.

    Why this matters:
        - torch.cuda.is_available() does lazy one-time CUDA driver context
          init. Calling it from a Qt worker thread for the first time can
          hang on Linux + NVIDIA setups (we hit this in earlier iterations).
        - unsloth's import patches PyTorch internals — if this happens on a
          worker thread mid-app, behavior can be unpredictable.
        - transformers transitively imports rust-backed tokenizers with
          rayon thread pools that don't always behave outside the main thread.

        Doing all three on the main thread before the worker thread is
        spawned avoids these classes of hang. Cost: ~10–15 seconds of
        main-thread time at startup before the GUI window appears.

    If anything isn't installed, this is a silent no-op — the engine will
    surface a clear error message when the user tries to load the model.
    """
    print("Initializing PyTorch + unsloth + transformers...", flush=True)
    try:
        import torch
        torch.cuda.is_available()  # forces lazy CUDA driver context init
        print(f"  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    except ImportError:
        print("  torch not installed (ok — will surface in UI)", flush=True)
        return
    except Exception as e:
        print(f"  torch warm-up skipped: {e}", flush=True)

    try:
        from unsloth import FastLanguageModel  # noqa: F401
        print("  unsloth imported", flush=True)
    except ImportError:
        print("  unsloth not installed (ok — will surface in UI)", flush=True)
    except Exception as e:
        print(f"  unsloth warm-up skipped: {e}", flush=True)

    try:
        from transformers import AutoTokenizer  # noqa: F401
        from snac import SNAC  # noqa: F401
        from huggingface_hub import snapshot_download  # noqa: F401
        print("  transformers + snac + huggingface_hub imported", flush=True)
    except ImportError as e:
        print(f"  some imports missing (ok — will surface in UI): {e}", flush=True)
    except Exception as e:
        print(f"  warm-up skipped: {e}", flush=True)

    print("Warm-up done. Opening main window.", flush=True)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Greek TTS")
    app.setOrganizationName("Greek TTS")

    # Must run AFTER QApplication is created (so we're on Qt's main thread)
    # and BEFORE MainWindow is shown.
    _warm_up_torch()

    window = MainWindow(APP_DIR)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
