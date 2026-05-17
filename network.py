"""
Pre-flight cache checks for the GreekTTS-1.5 model stack.

Three components must be downloaded on first launch:
  - unsloth/orpheus-3b-0.1-ft           (~2 GB in 4-bit form, the slow one)
  - moiralabs/GreekTTS-1.5              (~389 MB LoRA adapter)
  - hubertsiuzdak/snac_24khz            (~80 MB SNAC vocoder)

The Orpheus base download is by far the longest. Warning the user up-front
turns a silent 8-minute hang into an explicit choice.
"""

from __future__ import annotations

from pathlib import Path


BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"
LORA_REPO = "moiralabs/GreekTTS-1.5"
SNAC_REPO = "hubertsiuzdak/snac_24khz"


def _is_cached(repo_id: str, filename: str) -> bool:
    """Return True if the named file is already in the local HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    except Exception:
        return False
    return isinstance(cached, str) and Path(cached).is_file()


def check_models_cached() -> tuple[bool, list[str]]:
    """
    Returns (all_cached, list_of_missing_components).

    Each component is checked by probing one signature file. If a component
    is partially downloaded (interrupted), the signature check usually still
    flags it as missing, which is the safer behavior — let HuggingFace
    resume the download rather than assume it's complete.
    """
    missing: list[str] = []

    # Orpheus base — check for the model weights file (different name in 4-bit form)
    if not (
        _is_cached(BASE_MODEL, "model.safetensors")
        or _is_cached(BASE_MODEL, "model.safetensors.index.json")
    ):
        missing.append("Orpheus base (~2 GB)")

    # LoRA adapter — check inside the checkpoint subfolder
    if not _is_cached(LORA_REPO, "checkpoint-264000/adapter_model.safetensors"):
        missing.append("Greek LoRA adapter (~389 MB)")

    # SNAC vocoder
    if not (
        _is_cached(SNAC_REPO, "pytorch_model.bin")
        or _is_cached(SNAC_REPO, "model.safetensors")
    ):
        missing.append("SNAC vocoder (~80 MB)")

    return len(missing) == 0, missing


def model_warning_message(missing: list[str]) -> str:
    """User-facing Greek text shown when one or more components aren't cached."""
    items = "\n".join(f"  • {m}" for m in missing)
    return (
        "Λείπουν τα παρακάτω αρχεία μοντέλου:\n\n"
        f"{items}\n\n"
        "Στη φόρτωση θα γίνει αυτόματη λήψη από το Hugging Face — "
        "απαιτείται σύνδεση internet στην πρώτη εκτέλεση. "
        "Συνολικά έως ~2.5 GB. Η λήψη μπορεί να πάρει 5–15 λεπτά "
        "ανάλογα με την ταχύτητα του δικτύου.\n\n"
        "Μετά την πρώτη λήψη, η εφαρμογή λειτουργεί πλήρως offline."
    )
