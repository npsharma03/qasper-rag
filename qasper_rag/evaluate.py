"""Small evaluation helpers shared by the regression script and the app."""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

DEFAULT_KS = (1, 5, 10, 20)


def hit_flags(ranked_uids: Sequence[str], gold_uids: Iterable[str],
              ks: Sequence[int] = DEFAULT_KS) -> dict[str, int]:
    """Hit@k: 1 if any gold chunk is among the first k ranked chunks (as in the notebook)."""
    gold = set(gold_uids)
    return {f"hit@{k}": int(bool(set(ranked_uids[:k]) & gold)) for k in ks}


def first_gold_rank(table: pd.DataFrame, gold_uids: Iterable[str], rank_column: str) -> int | None:
    """Best (smallest) rank of any gold chunk in a Retrieval table, or None if none is present."""
    mask = table["chunk_uid"].isin(set(gold_uids))
    if not mask.any():
        return None
    return int(table.loc[mask, rank_column].min())
