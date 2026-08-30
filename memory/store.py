"""
AURA — long-term memory (Phase 1).

ChromaDB vector store with local sentence-transformers embeddings. Every exchange
is saved; relevant ones are retrieved to give AURA continuity across restarts.

Two decisions worth knowing:

* Embeddings run on CPU. The RTX 4050 has 6GB and qwen3:8b already occupies
  ~5.2GB of it. A 22M-parameter embedding model is fast enough on CPU and never
  competes for VRAM, which matters more than the few milliseconds saved.

* Corrections outrank everything. When Keerthana says "no, I meant X", that is
  far higher signal than a dozen casual mentions. Corrections are stored with a
  distinct kind and given a relevance boost at retrieval, so an explicit fix
  wins over older, more numerous, more loosely related memories.

Nothing reaches the store without passing the redaction filter first.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from aura import config
from aura.safety import redaction

KIND_EXCHANGE = "exchange"
KIND_CORRECTION = "correction"
KIND_FACT = "fact"
KIND_EVENT = "event"


@dataclass
class Memory:
    id: str
    text: str
    kind: str
    speaker: str
    timestamp: str
    score: float = 0.0
    metadata: dict[str, Any] | None = None

    def age_days(self) -> float:
        try:
            then = datetime.fromisoformat(self.timestamp)
        except ValueError:
            return 0.0
        return (datetime.now().astimezone() - then).total_seconds() / 86_400


class MemoryStore:
    """Persistent vector memory."""

    def __init__(self, path: Any = None, collection: str | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from chromadb.utils import embedding_functions

        cfg = config.SETTINGS.memory
        self.path = str(path or config.MEMORY_DIR)
        self.collection_name = collection or cfg.collection

        config.ensure_dirs()

        started = time.perf_counter()
        self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.embedding_model,
            device=cfg.embedding_device,
        )
        self._client = chromadb.PersistentClient(
            path=self.path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "memory ready at {} ({} items, {:.1f}s)",
            self.path,
            self._collection.count(),
            time.perf_counter() - started,
        )

    # ----------------------------------------------------------------- write
    def add(
        self,
        text: str,
        kind: str = KIND_FACT,
        speaker: str = "system",
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Store one memory. Returns its id, or None if nothing was stored."""
        result = redaction.redact(text)
        if not result.is_clean:
            logger.warning("redacted before storing memory: {}", result.summary())
        clean = result.text.strip()
        if not clean:
            return None

        memory_id = uuid.uuid4().hex
        metadata: dict[str, Any] = {
            "kind": kind,
            "speaker": speaker,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "redacted": not result.is_clean,
        }
        if extra:
            # Chroma metadata values must be scalars.
            metadata.update(
                {k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool))}
            )

        self._collection.add(ids=[memory_id], documents=[clean], metadatas=[metadata])
        return memory_id

    def add_exchange(self, user_text: str, aura_text: str, speaker: str = "") -> list[str]:
        """Store a conversational turn as a single retrievable unit.

        Stored together rather than separately because a reply is meaningless
        without the prompt that produced it.
        """
        who = speaker or config.SETTINGS.primary_user
        combined = f"{who}: {user_text.strip()}\nAURA: {aura_text.strip()}"
        memory_id = self.add(combined, kind=KIND_EXCHANGE, speaker=who)
        return [memory_id] if memory_id else []

    def add_correction(self, text: str, speaker: str = "") -> str | None:
        """Store an explicit correction. These outrank ordinary memories."""
        who = speaker or config.SETTINGS.primary_user
        logger.info("correction recorded")
        return self.add(text, kind=KIND_CORRECTION, speaker=who)

    # ------------------------------------------------------------------ read
    def search(
        self,
        query: str,
        k: int | None = None,
        kinds: list[str] | None = None,
    ) -> list[Memory]:
        """Retrieve by relevance, with corrections boosted."""
        cfg = config.SETTINGS.memory
        limit = k or cfg.retrieve_k
        if self._collection.count() == 0:
            return []

        where = {"kind": {"$in": kinds}} if kinds else None
        # Over-fetch so the correction boost has candidates to promote.
        fetch = min(limit * 3, max(self._collection.count(), 1))

        response = self._collection.query(
            query_texts=[redaction.redact(query).text],
            n_results=fetch,
            where=where,
        )

        docs = (response.get("documents") or [[]])[0]
        metas = (response.get("metadatas") or [[]])[0]
        dists = (response.get("distances") or [[]])[0]
        ids = (response.get("ids") or [[]])[0]

        memories: list[Memory] = []
        for doc, meta, dist, mid in zip(docs, metas, dists, ids, strict=False):
            meta = meta or {}
            # Cosine distance -> similarity.
            score = 1.0 - float(dist)
            if meta.get("kind") == KIND_CORRECTION:
                score += cfg.correction_boost
            memories.append(
                Memory(
                    id=mid,
                    text=doc,
                    kind=str(meta.get("kind", KIND_FACT)),
                    speaker=str(meta.get("speaker", "")),
                    timestamp=str(meta.get("timestamp", "")),
                    score=score,
                    metadata=meta,
                )
            )

        memories.sort(key=lambda m: m.score, reverse=True)
        return memories[:limit]

    def recall_block(self, query: str, k: int | None = None) -> str:
        """Format retrieved memories for injection into the system prompt."""
        memories = self.search(query, k=k)
        if not memories:
            return ""
        lines = ["Relevant things you remember (most relevant first):"]
        for m in memories:
            marker = " [correction - this overrides older memories]" if m.kind == KIND_CORRECTION else ""
            lines.append(f"- {m.text}{marker}")
        return "\n".join(lines)

    # ------------------------------------------------------------ management
    def count(self) -> int:
        return self._collection.count()

    def stats(self) -> dict[str, int]:
        """Counts by kind, for the control panel."""
        total = self._collection.count()
        out = {"total": total}
        if total == 0:
            return out
        got = self._collection.get(include=["metadatas"])
        for meta in got.get("metadatas") or []:
            kind = str((meta or {}).get("kind", "unknown"))
            out[kind] = out.get(kind, 0) + 1
        return out

    def recent(self, limit: int = 10) -> list[Memory]:
        got = self._collection.get(include=["documents", "metadatas"])
        rows: list[Memory] = []
        for mid, doc, meta in zip(
            got.get("ids") or [],
            got.get("documents") or [],
            got.get("metadatas") or [],
            strict=False,
        ):
            meta = meta or {}
            rows.append(
                Memory(
                    id=mid,
                    text=doc,
                    kind=str(meta.get("kind", "")),
                    speaker=str(meta.get("speaker", "")),
                    timestamp=str(meta.get("timestamp", "")),
                    metadata=meta,
                )
            )
        rows.sort(key=lambda m: m.timestamp, reverse=True)
        return rows[:limit]

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def reset(self) -> None:
        """Wipe all memories. Used by tests, never at runtime."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )
