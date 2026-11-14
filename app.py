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

import os
import sys
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen


APP_DIR = Path(__file__).resolve().parent


def _ensure_std_streams() -> None:
    """
    Give the app a stdout/stderr when it has none.

    The Windows build is windowed (console=False), so sys.stdout and
    sys.stderr are None. Libraries that write to them directly - tqdm
    progress bars, transformers warnings - then crash with
    "'NoneType' object has no attribute 'write'". Point both at a log file
    instead, which also leaves users something to send when a load fails.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    log_path = Path(base) / "GreekTTS" / "greektts.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(log_path, "w", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


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
    Pre-initialize PyTorch, bitsandbytes, and transformers on the main thread.

    Why this matters:
        - torch.cuda.is_available() does lazy one-time CUDA driver context
          init. Calling it from a Qt worker thread for the first time can
          hang on Linux + NVIDIA setups (we hit this in earlier iterations).
        - bitsandbytes loads its native CUDA library on import - do that
          once, up front, rather than mid-load on the worker thread.
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
    print("Initializing PyTorch + bitsandbytes + transformers...", flush=True)
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

    status("Φόρτωση bitsandbytes…  (Loading bitsandbytes)")
    try:
        import bitsandbytes  # noqa: F401
        print("  bitsandbytes imported", flush=True)
    except ImportError:
        print("  bitsandbytes not installed (ok - will surface in UI)", flush=True)
    except Exception as e:
        print(f"  bitsandbytes warm-up skipped: {e}", flush=True)

    status("Φόρτωση transformers / SNAC…  (Loading transformers / SNAC)")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        import peft  # noqa: F401
        from snac import SNAC  # noqa: F401
        from huggingface_hub import snapshot_download  # noqa: F401
        print("  transformers + peft + snac + huggingface_hub imported", flush=True)
    except ImportError as e:
        print(f"  some imports missing (ok - will surface in UI): {e}", flush=True)
    except Exception as e:
        print(f"  warm-up skipped: {e}", flush=True)

    print("Warm-up done. Opening main window.", flush=True)


def _self_test(report_path: str) -> int:
    """
    Import the whole model stack the way the engine does, then exit.

    Run by CI against the packaged exe (`GreekTTS.exe --selftest out.txt`)
    so modules PyInstaller missed fail the build instead of a user's first
    model load. Needs no GPU and no model downloads.
    """
    lines = []
    status = 0
    try:
        import torch
        lines.append(f"torch {torch.__version__} cuda {torch.version.cuda}")
        if not torch.version.cuda:
            lines.append("FAIL: torch is a CPU-only build")
            status = 1
        import bitsandbytes
        import peft
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: F401
        from transformers.models.llama import modeling_llama  # noqa: F401
        from transformers.quantizers import auto  # noqa: F401
        from snac import SNAC  # noqa: F401
        from huggingface_hub import snapshot_download  # noqa: F401
        import ui  # noqa: F401
        import tts_engine  # noqa: F401
        lines.append(
            f"transformers {transformers.__version__} peft {peft.__version__} "
            f"bitsandbytes {bitsandbytes.__version__}"
        )
    except Exception:
        import traceback
        lines.append(traceback.format_exc())
        status = 1
    lines.append("SELFTEST OK" if status == 0 else "SELFTEST FAILED")
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def main() -> int:
    # Before anything else runs, so every import below can print safely
    _ensure_std_streams()

    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        return _self_test(sys.argv[2])

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
