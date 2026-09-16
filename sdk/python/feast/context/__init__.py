"""Token-budget-aware helpers for assembling LLM context in feature views.

Plain functions over immutable values, so an OnDemandFeatureView calling them
gets the same prompt offline and online.
"""

from feast.context.compress import (
    CompressFn,
    GroupBySpec,
    compress_count,
    compress_sample,
    compress_sequence,
    compress_summarize,
)
from feast.context.errors import (
    ContextError,
    RequiredSectionError,
    TemplateRenderError,
    TokenBudgetExceededError,
    TokenizerNotFoundError,
    TokenizerUnavailableError,
)
from feast.context.priority_select import (
    DEFAULT_SEPARATOR,
    Priority,
    PrioritySpec,
    Section,
    SectionOrder,
    SectionSpec,
    Selection,
    priority_select,
)
from feast.context.prompt_template import PromptTemplate
from feast.context.render import (
    DEFAULT_TRUNCATION_MARKER,
    OverflowStrategy,
    Rendering,
    TemplateSpec,
    TemplateVariable,
    Truncation,
    VariableSpec,
    render_with_budget,
    truncate,
)
from feast.context.token_budget import TokenBudget
from feast.context.tokenizer import (
    ApproximateTokenizer,
    Cl100kBaseTokenizer,
    O200kBaseTokenizer,
    TiktokenTokenizer,
    Tokenizer,
    TokenizerName,
    TokenizerSpec,
    get_tokenizer,
)

__all__ = [
    "ApproximateTokenizer",
    "Cl100kBaseTokenizer",
    "CompressFn",
    "ContextError",
    "DEFAULT_SEPARATOR",
    "DEFAULT_TRUNCATION_MARKER",
    "GroupBySpec",
    "O200kBaseTokenizer",
    "OverflowStrategy",
    "Priority",
    "PrioritySpec",
    "PromptTemplate",
    "Rendering",
    "RequiredSectionError",
    "Section",
    "SectionOrder",
    "SectionSpec",
    "Selection",
    "TemplateRenderError",
    "TemplateSpec",
    "TemplateVariable",
    "TiktokenTokenizer",
    "TokenBudget",
    "TokenBudgetExceededError",
    "Tokenizer",
    "TokenizerName",
    "TokenizerNotFoundError",
    "TokenizerSpec",
    "TokenizerUnavailableError",
    "Truncation",
    "VariableSpec",
    "compress_count",
    "compress_sample",
    "compress_sequence",
    "compress_summarize",
    "get_tokenizer",
    "priority_select",
    "render_with_budget",
    "truncate",
]
