"""
Text Preprocessing Module
=========================
Conservative text preprocessing for security-sensitive prompt classification.
Preserves attack-relevant features like punctuation, URLs, code, and special tokens.
"""

import logging
import re
import unicodedata
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form for consistency."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Normalize excessive internal whitespace while preserving structure.
    
    - Strips leading/trailing whitespace.
    - Collapses runs of spaces/tabs on the same line to a single space.
    - Preserves single newlines (attack indicators can rely on line structure).
    """
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines to at most 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace (spaces and tabs) to single space per line
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(lines)
    return text.strip()


def preprocess_text(
    text: str,
    do_normalize_unicode: bool = True,
    do_normalize_whitespace: bool = True,
    do_lowercase: bool = False,
) -> str:
    """Apply conservative text preprocessing.
    
    Deliberately does NOT:
    - Remove stop words
    - Stem or lemmatize
    - Strip punctuation or special chars
    - Remove URLs or code fragments
    
    These features are informative for detecting prompt injection / jailbreak attacks.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    if do_normalize_unicode:
        text = normalize_unicode(text)

    if do_normalize_whitespace:
        text = normalize_whitespace(text)

    if do_lowercase:
        text = text.lower()

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    processed_col: str = "prompt_processed",
    do_normalize_unicode: bool = True,
    do_normalize_whitespace: bool = True,
    do_lowercase: bool = False,
) -> pd.DataFrame:
    """Preprocess prompts in a DataFrame, keeping originals for audit.
    
    Args:
        df: Input DataFrame.
        prompt_col: Column containing raw prompts.
        processed_col: Column to store preprocessed prompts.
        do_normalize_unicode: Apply Unicode NFC normalization.
        do_normalize_whitespace: Normalize whitespace.
        do_lowercase: Lowercase text (not recommended for security).
    
    Returns:
        DataFrame with additional processed column.
    """
    df = df.copy()

    # Handle missing prompts
    missing_mask = df[prompt_col].isnull()
    if missing_mask.any():
        logger.warning(f"Found {missing_mask.sum()} null prompts, filling with empty string")
        df[prompt_col] = df[prompt_col].fillna("")

    # Ensure string type
    df[prompt_col] = df[prompt_col].astype(str)

    # Apply preprocessing
    df[processed_col] = df[prompt_col].apply(
        lambda x: preprocess_text(
            x,
            do_normalize_unicode=do_normalize_unicode,
            do_normalize_whitespace=do_normalize_whitespace,
            do_lowercase=do_lowercase,
        )
    )

    # Log preprocessing changes
    changed_mask = df[prompt_col] != df[processed_col]
    n_changed = changed_mask.sum()
    logger.info(
        f"Preprocessing modified {n_changed}/{len(df)} prompts "
        f"({n_changed / len(df) * 100:.1f}%)"
    )

    return df
