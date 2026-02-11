from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class ChunkingConfig:
    chunk_size: int = 100
    chunk_overlap: int = 20
    min_chunk_size: int = 20
    max_chunk_chars: Optional[int] = 500


class BaseChunker(ABC):
    """
    Abstract base class for document chunking.

    Subclasses implement load_parse_and_chunk() with their own:
    - Loading logic
    - Parsing logic
    - Chunking strategy
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    @abstractmethod
    def load_parse_and_chunk(
        self,
        source: Any,
        source_id: str,
        source_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Load, parse, and chunk a document.

        Args:
            source: File path, raw text, bytes, etc.
            source_id: Document identifier.
            source_type: Optional type hint.

        Returns:
            List of chunk dicts with keys:
                - chunk_id: str
                - original_id: str
                - text: str
                - chunk_index: int
                - (any additional metadata)
        """
        pass

    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        id_column: str,
        source_column: str,
        type_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Chunk all documents in a DataFrame.

        Args:
            df: The DataFrame containing the documents to chunk.
            id_column: The column containing the document IDs.
            source_column: The column containing the document sources.
            type_column: The column containing the document types.
        """

        def process_row(row):
            source_id = str(row[id_column])
            source = row[source_column]
            source_type = row[type_column] if type_column else None
            return self.load_parse_and_chunk(source, source_id, source_type)

        # Apply and explode
        df["_chunks"] = df.apply(process_row, axis=1)
        chunks_df = df["_chunks"].explode().apply(pd.Series)

        return chunks_df.reset_index(drop=True)


class TextChunker(BaseChunker):
    """Default chunker for plain text. Chunks by word count."""

    def load_parse_and_chunk(
        self,
        source: Any,
        source_id: str,
        source_type: Optional[str] = None,
    ) -> list[dict]:
        # Load
        text = self._load(source)

        # Chunk by words
        return self._chunk_by_words(text, source_id)

    def _load(self, source: Any) -> str:
        from pathlib import Path

        if isinstance(source, Path) and source.exists():
            return Path(source).read_text()
        if isinstance(source, str):
            if source.endswith(".txt"):
                return Path(source).read_text()
        return str(source)

    def _chunk_by_words(self, text: str, source_id: str) -> list[dict]:
        words = text.split()
        chunks = []

        step = self.config.chunk_size - self.config.chunk_overlap
        chunk_index = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.config.chunk_size]

            if len(chunk_words) < self.config.min_chunk_size:
                continue

            chunk_text = " ".join(chunk_words)
            if self.config.max_chunk_chars:
                chunk_text = chunk_text[: self.config.max_chunk_chars]

            chunks.append(
                {
                    "chunk_id": f"{source_id}_{chunk_index}",
                    "original_id": source_id,
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

        return chunks
