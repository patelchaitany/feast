from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class EmbeddingConfig:
    batch_size: int = 64
    show_progress: bool = True


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding generation.

    Supports multiple modalities via routing.
    Users can register custom modality handlers.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()

        # Registry: modality -> embedding function
        self._modality_handlers: dict[str, Callable[[list[Any]], np.ndarray]] = {}

        # Register default modalities (subclass can override)
        self._register_default_modalities()

    def _register_default_modalities(self) -> None:
        """Override in subclass to register default modality handlers."""
        pass

    def register_modality(
        self,
        modality: str,
        handler: Callable[[list[Any]], np.ndarray],
    ) -> None:
        """
        Register a handler for a modality.

        Args:
            modality: Name of modality ("text", "image", "video", etc.)
            handler: Function that takes list of inputs and returns embeddings.
        """
        self._modality_handlers[modality] = handler

    def supported_modalities(self) -> list[str]:
        """Return list of supported modalities."""
        return list(self._modality_handlers.keys())

    @abstractmethod
    def embed(self, inputs: list[Any], modality: str) -> np.ndarray:
        """
        Generate embeddings for inputs of a given modality.

        Args:
            inputs: List of inputs.
            modality: Type of content ("text", "image", "video", etc.)

        Returns:
            numpy array of shape (len(inputs), embedding_dim)
        """
        pass

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        column_mapping: dict[str, tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Add embeddings for multiple columns with modality routing.

        Args:
            df: Input DataFrame.
            column_mapping: Dict mapping source_column -> (modality, output_column).
                Example: {
                    "text": ("text", "text_embedding"),
                    "image_path": ("image", "image_embedding"),
                    "video_path": ("video", "video_embedding"),
                }
        """
        df = df.copy()

        for source_column, (modality, output_column) in column_mapping.items():
            inputs = df[source_column].tolist()
            embeddings = self.embed(inputs, modality)
            df[output_column] = pd.Series(
                [emb.tolist() for emb in embeddings], dtype=object
            )

        return df


class MultiModalEmbedder(BaseEmbedder):
    """
    Multi-modal embedder with built-in support for common modalities.

    Supports: text, image, video (extensible)
    """

    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        image_model: str = "openai/clip-vit-base-patch32",
        config: Optional[EmbeddingConfig] = None,
    ):
        self.text_model_name = text_model
        self.image_model_name = image_model

        # Lazy-loaded models
        self._text_model = None
        self._image_model = None
        self._image_processor = None

        super().__init__(config)

    def _register_default_modalities(self) -> None:
        """Register built-in modality handlers."""
        self.register_modality("text", self._embed_text)
        self.register_modality("image", self._embed_image)
        # Future: add more as needed
        # self.register_modality("video", self._embed_video)
        # self.register_modality("audio", self._embed_audio)

    def embed(self, inputs: list[Any], modality: str) -> np.ndarray:
        """Route to appropriate handler based on modality."""
        if modality not in self._modality_handlers:
            raise ValueError(
                f"Unsupported modality: '{modality}'. "
                f"Supported: {self.supported_modalities()}"
            )

        handler = self._modality_handlers[modality]
        return handler(inputs)

    # Text Embedding
    @property
    def text_model(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(self.text_model_name)
        return self._text_model

    def _embed_text(self, inputs: list[str]) -> np.ndarray:
        return self.text_model.encode(
            inputs,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress,
        )

    # Image Embedding
    @property
    def image_model(self):
        if self._image_model is None:
            from transformers import CLIPModel

            self._image_model = CLIPModel.from_pretrained(self.image_model_name)
        return self._image_model

    @property
    def image_processor(self):
        if self._image_processor is None:
            from transformers import CLIPProcessor

            self._image_processor = CLIPProcessor.from_pretrained(self.image_model_name)
        return self._image_processor

    def _embed_image(self, inputs: list[Any]) -> np.ndarray:
        from pathlib import Path

        from PIL import Image

        images = []
        for inp in inputs:
            if isinstance(inp, (str, Path)):
                images.append(Image.open(inp))
            else:
                images.append(inp)

        processed = self.image_processor(images=images, return_tensors="pt")
        embeddings = self.image_model.get_image_features(**processed)
        return embeddings.detach().numpy()
