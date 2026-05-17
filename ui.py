"""
GUI for the Greek TTS desktop app (PySide6).
Backed by Moira.AI GreekTTS-1.5 (Orpheus + Greek LoRA + SNAC).

Layout:
  ┌──────────────────────────────────────────┐
  │ Title                                     │
  │ ── Text to synthesize ──                 │
  │   [textarea]                             │
  │ [format] [Generate] [Play] [Stop] [Save] │
  │ Status bar                                │
  └──────────────────────────────────────────┘
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig
from network import check_models_cached, model_warning_message
from tts_engine import DEFAULT_FORMAT, FORMATS, GreekTTSEngine, TTSEngineError


# =====================================================================
# Worker threads
# =====================================================================

class ModelLoaderWorker(QObject):
    """Loads the Greek TTS pipeline in a background thread."""
    finished = Signal(object)
    failed = Signal(str)

    def run(self) -> None:
        print(">>> ModelLoader: thread started, building GreekTTSEngine...", flush=True)
        try:
            engine = GreekTTSEngine(use_cuda=True)
            print(">>> ModelLoader: engine ready, emitting finished signal", flush=True)
            self.finished.emit(engine)
        except TTSEngineError as e:
            print(f">>> ModelLoader: TTSEngineError: {e}", flush=True)
            self.failed.emit(str(e))
        except Exception as e:
            print(f">>> ModelLoader: {type(e).__name__}: {e}", flush=True)
            self.failed.emit(f"Unexpected error: {e}")


class SynthesisWorker(QObject):
    """Runs one synthesis on a background thread."""
    finished = Signal(Path)
    failed = Signal(str)

    def __init__(
        self,
        engine: GreekTTSEngine,
        text: str,
        out_path: Path,
        fmt: str,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
        seed: int | None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.text = text
        self.out_path = out_path
        self.fmt = fmt
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    def run(self) -> None:
        try:
            result = self.engine.synthesize(
                text=self.text,
                output_path=self.out_path,
                fmt=self.fmt,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                seed=self.seed,
            )
            self.finished.emit(result)
        except TTSEngineError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"Unexpected error: {e}")


# =====================================================================
# Main window
# =====================================================================

class MainWindow(QMainWindow):
    SAMPLE_TEXT = (
        "Καλημέρα σας. Σας ενημερώνουμε ότι το ραντεβού σας έχει επιβεβαιωθεί. "
        "Πατήστε ένα για επιβεβαίωση, δύο για αλλαγή ημερομηνίας."
    )

    def __init__(self, app_dir: Path) -> None:
        super().__init__()
        self.app_dir = app_dir
        self.config = AppConfig.load(app_dir)
        self.engine: GreekTTSEngine | None = None
        self.last_output: Path | None = None
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="greek-tts-"))
        self._loader_thread: QThread | None = None
        self._synth_thread: QThread | None = None
        # Workers stored as members (not locals) so Python's GC doesn't
        # destroy them while they're running on the worker thread.
        self._loader_worker: ModelLoaderWorker | None = None
        self._synth_worker: SynthesisWorker | None = None
        self._load_timeouts: list[QTimer] = []

        self.setWindowTitle("Μετατροπέας Κειμένου σε Ομιλία — Greek TTS (Moira)")
        self.resize(780, 580)
        self.setMinimumSize(640, 480)

        self._build_ui()
        self._setup_audio()
        self._apply_styles()

        # Trigger model load immediately
        self._start_model_load()

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 18)
        root.setSpacing(14)

        title = QLabel("Μετατροπέας Κειμένου σε Ομιλία")
        title.setObjectName("title")
        subtitle = QLabel("Greek text → telephony-ready audio · Moira GreekTTS-1.5")
        subtitle.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("separator")
        root.addWidget(sep)

        text_label = QLabel("Κείμενο προς μετατροπή")
        text_label.setObjectName("fieldLabel")
        root.addWidget(text_label)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "Πληκτρολογήστε ή επικολλήστε ελληνικό κείμενο εδώ…"
        )
        self.text_edit.setPlainText(self.SAMPLE_TEXT)
        self.text_edit.setMinimumHeight(160)
        root.addWidget(self.text_edit, 1)

        # Format + actions row
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        format_label = QLabel("Φορμά:")
        format_label.setObjectName("inlineLabel")
        action_row.addWidget(format_label)

        self.format_combo = QComboBox()
        for key, spec in FORMATS.items():
            self.format_combo.addItem(spec["label"], userData=key)
        idx = self.format_combo.findData(self.config.default_format or DEFAULT_FORMAT)
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.format_combo.setMinimumWidth(280)
        action_row.addWidget(self.format_combo)

        action_row.addStretch(1)

        self.generate_btn = QPushButton("Δημιουργία")
        self.generate_btn.setObjectName("primary")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._on_generate)
        action_row.addWidget(self.generate_btn)

        self.play_btn = QPushButton("▶")
        self.play_btn.setEnabled(False)
        self.play_btn.setMaximumWidth(40)
        self.play_btn.clicked.connect(self._on_play)
        action_row.addWidget(self.play_btn)

        self.stop_btn = QPushButton("■")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMaximumWidth(40)
        self.stop_btn.clicked.connect(self._on_stop)
        action_row.addWidget(self.stop_btn)

        self.save_btn = QPushButton("Αποθήκευση…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_as)
        action_row.addWidget(self.save_btn)

        root.addLayout(action_row)

        # Indeterminate progress bar (for load + synthesis)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(4)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._set_status("Έτοιμο.", info=True)

    def _setup_audio(self) -> None:
        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.player.setAudioOutput(self.audio_out)
        self.player.playbackStateChanged.connect(self._on_playback_state)

    # ---------- Model loading ----------

    def _start_model_load(self) -> None:
        if self._loader_thread is not None:
            return

        # Pre-flight: check whether all three model components are cached.
        # Spares us a multi-minute mystery hang on first launch.
        is_cached, missing = check_models_cached()
        if not is_cached:
            ret = QMessageBox.question(
                self,
                "Λήψη μοντέλων",
                model_warning_message(missing),
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Ok,
            )
            if ret != QMessageBox.StandardButton.Ok:
                self._set_status(
                    "Ακυρώθηκε η φόρτωση. Επανεκκινήστε για νέα προσπάθεια.",
                    info=True,
                )
                return

        self._set_busy(True, msg="Φόρτωση Moira GreekTTS-1.5… (μπορεί να διαρκέσει αρκετά λεπτά)")

        # Progressive timeout messages — model load can take 30s (cached)
        # to 10+ minutes (first-time download on slow networks).
        self._stop_load_timeouts()
        timeout_messages = [
            (30_000,  "Φόρτωση… ελέγξτε το τερματικό για πρόοδο λήψης."),
            (120_000, "Η φόρτωση καθυστερεί. Πιθανή λήψη μοντέλου από αργό δίκτυο."),
            (600_000, "Δεν ολοκληρώθηκε εδώ και 10 λεπτά. Δείτε το τερματικό."),
        ]
        for delay_ms, msg in timeout_messages:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda m=msg: self._set_status(m, info=True))
            timer.start(delay_ms)
            self._load_timeouts.append(timer)

        self._loader_thread = QThread(self)
        self._loader_worker = ModelLoaderWorker()
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.finished.connect(self._on_model_loaded)
        self._loader_worker.failed.connect(self._on_model_load_failed)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.failed.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._loader_worker.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.finished.connect(self._on_loader_thread_finished)
        self._loader_thread.start()

    def _stop_load_timeouts(self) -> None:
        for timer in self._load_timeouts:
            timer.stop()
            timer.deleteLater()
        self._load_timeouts = []

    def _on_loader_thread_finished(self) -> None:
        self._loader_thread = None
        self._loader_worker = None

    def _on_model_loaded(self, engine: GreekTTSEngine) -> None:
        self._stop_load_timeouts()
        self.engine = engine
        self._set_busy(False)
        self._set_status(
            f"Έτοιμο. Μοντέλο φορτωμένο σε GPU (4-bit, {engine.sample_rate} Hz).",
            info=True,
        )
        self._update_generate_button()

    def _on_model_load_failed(self, msg: str) -> None:
        self._stop_load_timeouts()
        self.engine = None
        self._set_busy(False)
        self._set_status("Αποτυχία φόρτωσης μοντέλου.", error=True)
        QMessageBox.critical(self, "Σφάλμα φόρτωσης μοντέλου", msg)
        self._update_generate_button()

    # ---------- Synthesis ----------

    def _on_generate(self) -> None:
        if self.engine is None:
            QMessageBox.information(self, "Μοντέλο", "Το μοντέλο δεν είναι ακόμα φορτωμένο.")
            return
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Κενό κείμενο", "Παρακαλώ εισάγετε κείμενο.")
            return

        fmt = self.format_combo.currentData()
        out_path = self._tmp_dir / f"preview_{fmt}.wav"

        if fmt != self.config.default_format:
            self.config.default_format = fmt
            self.config.save(self.app_dir)

        # Stop any current playback (Windows holds the file open during playback)
        self.player.stop()
        self.player.setSource(QUrl())

        self._set_busy(
            True,
            msg="Γίνεται μετατροπή… (Orpheus παράγει tokens autoregressive ~5–15s)",
        )

        self._synth_thread = QThread(self)
        self._synth_worker = SynthesisWorker(
            engine=self.engine,
            text=text,
            out_path=out_path,
            fmt=fmt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            max_new_tokens=self.config.max_new_tokens,
            seed=self.config.seed,
        )
        self._synth_worker.moveToThread(self._synth_thread)
        self._synth_thread.started.connect(self._synth_worker.run)
        self._synth_worker.finished.connect(self._on_synth_done)
        self._synth_worker.failed.connect(self._on_synth_failed)
        self._synth_worker.finished.connect(self._synth_thread.quit)
        self._synth_worker.failed.connect(self._synth_thread.quit)
        self._synth_thread.finished.connect(self._synth_worker.deleteLater)
        self._synth_thread.finished.connect(self._synth_thread.deleteLater)
        self._synth_thread.finished.connect(self._on_synth_thread_finished)
        self._synth_thread.start()

    def _on_synth_thread_finished(self) -> None:
        self._synth_thread = None
        self._synth_worker = None

    def _on_synth_done(self, path: Path) -> None:
        self.last_output = path
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._set_busy(False)
        self._set_status(f"Έτοιμο. Διάρκεια: {self._duration_hint(path)}.", info=True)

    def _on_synth_failed(self, msg: str) -> None:
        self._set_busy(False)
        self._set_status("Αποτυχία μετατροπής.", error=True)
        QMessageBox.critical(self, "Σφάλμα μετατροπής", msg)

    # ---------- Playback ----------

    def _on_play(self) -> None:
        if self.last_output:
            self.player.play()

    def _on_stop(self) -> None:
        self.player.stop()

    def _on_playback_state(self, state) -> None:
        playing = state == QMediaPlayer.PlayingState
        self.play_btn.setEnabled(self.last_output is not None and not playing)
        self.stop_btn.setEnabled(playing)

    def _on_save_as(self) -> None:
        if not self.last_output:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Αποθήκευση WAV", "prompt.wav", "WAV (*.wav);;All files (*)"
        )
        if not path:
            return
        try:
            shutil.copyfile(self.last_output, path)
            self._set_status(f"Αποθηκεύτηκε: {path}", info=True)
        except Exception as e:
            QMessageBox.critical(self, "Σφάλμα αποθήκευσης", str(e))

    # ---------- Helpers ----------

    def _set_busy(self, busy: bool, msg: str | None = None) -> None:
        self.progress.setVisible(busy)
        self.text_edit.setReadOnly(busy)
        self.format_combo.setEnabled(not busy)
        if busy:
            self.generate_btn.setEnabled(False)
            if msg:
                self._set_status(msg, info=True)
        else:
            self._update_generate_button()

    def _update_generate_button(self) -> None:
        self.generate_btn.setEnabled(self.engine is not None)

    def _set_status(self, msg: str, info: bool = False, error: bool = False) -> None:
        if error:
            self.status.setStyleSheet("color: #b22; padding: 4px;")
        else:
            self.status.setStyleSheet("color: #555; padding: 4px;")
        self.status.showMessage(msg)

    @staticmethod
    def _duration_hint(path: Path) -> str:
        import wave
        try:
            with wave.open(str(path), "rb") as f:
                seconds = f.getnframes() / float(f.getframerate())
            return f"{seconds:.1f} s"
        except Exception:
            return "?"

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #faf7f2;
                color: #1a1d24;
            }
            QLabel#title {
                font-family: "Georgia", "Times New Roman", serif;
                font-size: 24px;
                font-weight: 600;
                color: #14233a;
                padding-top: 4px;
            }
            QLabel#subtitle {
                font-size: 12px;
                color: #6a7280;
                letter-spacing: 0.4px;
                padding-bottom: 2px;
            }
            QLabel#fieldLabel {
                font-size: 11px;
                font-weight: 600;
                color: #4a5260;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                padding-top: 4px;
            }
            QLabel#inlineLabel {
                font-size: 12px;
                color: #4a5260;
            }
            QFrame#separator {
                color: #e3ddd2;
                background: #e3ddd2;
                max-height: 1px;
            }
            QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #d8d2c5;
                border-radius: 4px;
                padding: 10px 12px;
                font-size: 14px;
                selection-background-color: #14233a;
                selection-color: #ffffff;
            }
            QPlainTextEdit:focus { border: 1px solid #14233a; }
            QComboBox {
                background: #ffffff;
                border: 1px solid #d8d2c5;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 13px;
            }
            QComboBox:focus { border: 1px solid #14233a; }
            QComboBox::drop-down { border: none; width: 22px; }
            QPushButton {
                background: #ffffff;
                border: 1px solid #c8c1b2;
                border-radius: 4px;
                padding: 7px 14px;
                font-size: 13px;
                color: #1a1d24;
            }
            QPushButton:hover { background: #f3eee5; }
            QPushButton:pressed { background: #e8e2d4; }
            QPushButton:disabled { color: #aaa; background: #f5f2ec; }
            QPushButton#primary {
                background: #14233a;
                color: #faf7f2;
                border: 1px solid #14233a;
                font-weight: 600;
                padding: 7px 18px;
            }
            QPushButton#primary:hover { background: #1d3354; }
            QPushButton#primary:pressed { background: #0f1c30; }
            QPushButton#primary:disabled { background: #6a7280; border-color: #6a7280; }
            QProgressBar {
                background: #e3ddd2;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: #14233a;
                border-radius: 2px;
            }
            QStatusBar {
                background: transparent;
                border-top: 1px solid #e3ddd2;
            }
        """)

    def closeEvent(self, event) -> None:
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass
        super().closeEvent(event)
