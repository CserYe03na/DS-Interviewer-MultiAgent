from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

MERGED_PATH = Path("data/merged1.jsonl")

def load_merged(merged_path: Path) -> List[dict]:
    records: List[dict] = []
    with merged_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def build_embeddings(
    merged: List[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[SentenceTransformer, np.ndarray]:

    embedder = SentenceTransformer(model_name)

    texts = [r["vector_text"] for r in merged]
    emb = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    return embedder, emb

def mmr_rerank(
    qvec: np.ndarray,
    cand_idx: np.ndarray,
    emb: np.ndarray,
    lambda_: float = 0.7,
    topk: int = 10,
) -> np.ndarray:

    cand_idx = np.asarray(cand_idx, dtype=np.int64)
    C = emb[cand_idx]
    qsim = C @ qvec

    selected: List[int] = []
    selected_mask = np.zeros(len(cand_idx), dtype=bool)

    cc_sim = C @ C.T

    for _ in range(min(topk, len(cand_idx))):
        if not selected:
            j = int(np.argmax(qsim))
        else:
            sel = np.array(selected, dtype=np.int64)
            max_to_sel = cc_sim[:, sel].max(axis=1)
            mmr_score = lambda_ * qsim - (1.0 - lambda_) * max_to_sel
            mmr_score[selected_mask] = -1e9
            j = int(np.argmax(mmr_score))

        selected.append(j)
        selected_mask[j] = True

    return cand_idx[np.array(selected, dtype=np.int64)]

class Retriever:

    def __init__(
        self,
        merged: List[dict],
        emb: np.ndarray,
        embedder: SentenceTransformer,
    ) -> None:

        self.merged = merged
        self.emb = emb
        self.embedder = embedder

        self.ids = [r["id"] for r in merged]
        self.titles = [r["metadata"].get("title", "") for r in merged]
        self.types = [r["metadata"].get("type", "") for r in merged]
        self.skills = [r["metadata"].get("taxonomy_skills", []) for r in merged]
        self.difficulties = [r["metadata"].get("difficulty") for r in merged]
        self.urls = [r["metadata"].get("url") for r in merged]
        self.backup_urls = [r["metadata"].get("backup_url") for r in merged]

    def retrieve(
        self,
        query: str,
        topk: int = 10,
        fetch: int = 200,
        lambda_: float = 0.7,
        type_filter: Optional[str] = None,
        skill_filter: Optional[str] = None,
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:

        q = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)[0]

        sims = self.emb @ q 

        if difficulty_distribution is not None:
            weights = np.ones_like(sims)
            for i, diff in enumerate(self.difficulties):
                if diff in difficulty_distribution:
                    weights[i] = difficulty_distribution[diff]
            sims = sims * weights

        n = len(self.merged)
        mask = np.ones(n, dtype=bool)

        if type_filter is not None:
            types_arr = np.array(self.types, dtype=object)
            mask &= (types_arr == type_filter)

        if skill_filter is not None:
            sf = skill_filter
            has_skill = np.array([sf in sk for sk in self.skills], dtype=bool)
            mask &= has_skill

        idx = np.where(mask)[0]
        if len(idx) == 0:
            idx = np.arange(n)

        cand_sims = sims[idx]
        M = min(fetch, len(idx))
        top_cand_local = np.argpartition(-cand_sims, M - 1)[:M]
        cand_idx = idx[top_cand_local]

        picked_idx = mmr_rerank(
            q, cand_idx, self.emb, lambda_=lambda_, topk=topk
        )

        results: List[Dict[str, Any]] = []
        for i in picked_idx:
            rec = self.merged[i]
            results.append(
                {
                    "score": float(sims[i]),
                    "id": self.ids[i],
                    "type": self.types[i],
                    "title": self.titles[i],
                    "difficulty": self.difficulties[i],
                    "url": self.urls[i], 
                    "backup_url": self.backup_urls[i],
                    "taxonomy_skills": self.skills[i][:8],
                    "preview": rec["vector_text"][:180].replace("\n", " "),
                    "metadata": rec["metadata"],
                    "data": rec["data"],
                }
            )

        return results

def init_retriever(
    merged_path: Path = MERGED_PATH,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Retriever:

    merged = load_merged(merged_path)
    embedder, emb = build_embeddings(merged, model_name=model_name)
    return Retriever(merged=merged, emb=emb, embedder=embedder)