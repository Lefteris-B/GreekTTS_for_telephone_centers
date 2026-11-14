"""
Greek TTS for Telephony - desktop application (Moira GreekTTS-1.5 edition).

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
from typing import Callable

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen


APP_DIR = Path(__file__).resolve().parent


SPLASH_TITLE = "Μετατροπέας Κειμένου σε Ομιλία"
SPLASH_SUBTITLE = "Greek TTS (Moira) - starting up, please wait…"


class _Splash(QSplashScreen):
    """QSplashScreen whose status message gets the same margins as the title."""

    def drawContents(self, painter: QPainter) -> None:
        painter.setPen(QColor("#4a5260"))
        painter.drawText(
            self.rect().adjusted(28, 0, -28, -24),
            Qt.AlignLeft | Qt.AlignBottom,
            self.message(),
        )


def _make_splash() -> QSplashScreen:
    """
    Build a splash screen shown while the heavy imports run.

    Without it, nothing appears for ~10-15 seconds after launch and users
    double-click again, starting extra copies. Drawn in code (colors match
    MainWindow's stylesheet) so no image asset needs bundling.
    """
    width, height = 520, 200
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#faf7f2"))

    painter = QPainter(pixmap)
    painter.setPen(QColor("#e3ddd2"))
    painter.drawRect(0, 0, width - 1, height - 1)

    title_font = QFont("Georgia", 18)
    title_font.setWeight(QFont.DemiBold)
    painter.setFont(title_font)
    painter.setPen(QColor("#14233a"))
    painter.drawText(QRect(28, 36, width - 56, 40), Qt.AlignLeft | Qt.AlignVCenter, SPLASH_TITLE)

    painter.setFont(QFont("Segoe UI", 10))
    painter.setPen(QColor("#6a7280"))
    painter.drawText(QRect(28, 80, width - 56, 24), Qt.AlignLeft | Qt.AlignVCenter, SPLASH_SUBTITLE)
    painter.end()

    splash = _Splash(pixmap, Qt.WindowStaysOnTopHint)
    splash.setFont(QFont("Segoe UI", 10))
    return splash


def _warm_up_torch(status: Callable[[str], None]) -> None:
    """
    Pre-initialize PyTorch, unsloth, and transformers on the main thread.

    Why this matters:
        - torch.cuda.is_available() does lazy one-time CUDA driver context
          init. Calling it from a Qt worker thread for the first time can
          hang on Linux + NVIDIA setups (we hit this in earlier iterations).
        - unsloth's import patches PyTorch internals - if this happens on a
          worker thread mid-app, behavior can be unpredictable.
        - transformers transitively imports rust-backed tokenizers with
          rayon thread pools that don't always behave outside the main thread.

        Doing all three on the main thread before the worker thread is
        spawned avoids these classes of hang. Cost: ~10–15 seconds of
        main-thread time at startup before the GUI window appears.

    If anything isn't installed, this is a silent no-op - the engine will
    surface a clear error message when the user tries to load the model.

    `status` is called before each stage so the splash screen can say
    what's happening.
    """
    print("Initializing PyTorch + unsloth + transformers...", flush=True)
    status("Φόρτωση PyTorch / CUDA…  (Loading PyTorch / CUDA)")
    try:
        import torch
        torch.cuda.is_available()  # forces lazy CUDA driver context init
        print(f"  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    except ImportError:
        print("  torch not installed (ok - will surface in UI)", flush=True)
        return
    except Exception as e:
        print(f"  torch warm-up skipped: {e}", flush=True)

    status("Φόρτωση unsloth…  (Loading unsloth)")
    try:
        from unsloth import FastLanguageModel  # noqa: F401
        print("  unsloth imported", flush=True)
    except ImportError:
        print("  unsloth not installed (ok - will surface in UI)", flush=True)
    except Exception as e:
        print(f"  unsloth warm-up skipped: {e}", flush=True)

    status("Φόρτωση transformers / SNAC…  (Loading transformers / SNAC)")
    try:
        from transformers import AutoTokenizer  # noqa: F401
        from snac import SNAC  # noqa: F401
        from huggingface_hub import snapshot_download  # noqa: F401
        print("  transformers + snac + huggingface_hub imported", flush=True)
    except ImportError as e:
        print(f"  some imports missing (ok - will surface in UI): {e}", flush=True)
    except Exception as e:
        print(f"  warm-up skipped: {e}", flush=True)

    print("Warm-up done. Opening main window.", flush=True)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Greek TTS")
    app.setOrganizationName("Greek TTS")

    splash = _make_splash()
    splash.show()

    def status(message: str) -> None:
        splash.showMessage(message)
        # Warm-up blocks the event loop; repaint the splash between stages
        app.processEvents()

    # Must run AFTER QApplication is created (so we're on Qt's main thread)
    # and BEFORE MainWindow is shown.
    _warm_up_torch(status)

    status("Άνοιγμα παραθύρου…  (Opening main window)")
    # Imported here, not at module top: ui pulls in tts_engine, and we want
    # the splash on screen before any of that import work happens.
    from ui import MainWindow

    window = MainWindow(APP_DIR)
    window.show()
    splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
