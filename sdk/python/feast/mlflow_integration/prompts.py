"""Load ``@prompt`` templates from the MLflow Prompt Registry.

A prompt whose ``template`` is an MLflow prompt URI, such as
``prompts:/concierge@production`` or ``prompts:/concierge/3``, takes its wording
from MLflow each time it is rendered. Feast still reads the features, calls the
prompt function and cuts the template's variables to fit the token budget.
"""

import functools
from dataclasses import dataclass
from typing import Any, Dict, Optional

from feast.errors import FeastExtrasDependencyImportError


@dataclass(frozen=True)
class MlflowPromptTemplate:
    """One version of a text prompt from the MLflow Prompt Registry.

    Attributes:
        uri: The version loaded, as ``prompts:/<name>/<version>``, even when it
            was loaded through an alias.
        template: The prompt's wording, with ``{{ variable }}`` placeholders.
        model_config: The model settings MLflow stores with the version, if any.
    """

    uri: str
    template: str
    model_config: Optional[Dict[str, Any]] = None


def load_prompt_template(
    uri: str, tracking_uri: Optional[str] = None
) -> MlflowPromptTemplate:
    """Load a text prompt from the MLflow Prompt Registry.

    MLflow caches what it loads: a version for good, an alias for 60 seconds by
    default (``MLFLOW_ALIAS_PROMPT_CACHE_TTL_SECONDS``), so moving an alias
    reaches renders within that time.

    Args:
        uri: ``prompts:/<name>/<version>`` or ``prompts:/<name>@<alias>``.
        tracking_uri: The MLflow server. None uses ``MLFLOW_TRACKING_URI``, then
            MLflow's default.

    Raises:
        FeastExtrasDependencyImportError: MLflow is not installed.
        ValueError: The prompt is a chat prompt, which ``@prompt`` cannot render.
    """
    prompt_version = _client(tracking_uri).load_prompt(uri)
    # Chat prompts and model configs came after the first Prompt Registry
    # release, so older MLflow versions lack these attributes.
    resolved = f"prompts:/{prompt_version.name}/{prompt_version.version}"
    if not getattr(prompt_version, "is_text_prompt", True):
        raise ValueError(
            f"MLflow prompt {resolved} is a chat prompt. @prompt renders text "
            f"prompts only."
        )
    return MlflowPromptTemplate(
        uri=resolved,
        template=prompt_version.template,
        model_config=getattr(prompt_version, "model_config", None),
    )


@functools.lru_cache(maxsize=None)
def _client(tracking_uri: Optional[str]) -> Any:
    try:
        from mlflow import MlflowClient
    except ImportError as e:
        raise FeastExtrasDependencyImportError("mlflow", str(e)) from e
    if not hasattr(MlflowClient, "load_prompt"):
        raise ValueError(
            "Loading prompt templates from MLflow needs the MLflow Prompt Registry, "
            "added in MLflow 2.21. Upgrade MLflow."
        )
    return MlflowClient(tracking_uri=tracking_uri)
