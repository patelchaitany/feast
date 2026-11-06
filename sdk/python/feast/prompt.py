"""Prompts kept in the registry, versioned, and filled in from features.

``@prompt`` turns a function into a :class:`Prompt`. The function takes one
argument per source, a dict of that source's values for one entity row, and
returns the prompt::

    @prompt(sources=[user_profile, location_request], max_tokens=300)
    def concierge(user_profile, request):
        return (
            f"You are a concierge for {user_profile['name']}. "
            f"Suggest places in {request['location']}."
        )

``feast apply`` stores the prompt in the registry: the function's source text
and its settings. Every change is stored as a new version beside the old ones.
The prompt is never written to the online store; ``FeatureStore.render_prompt``
reads its features online and calls the function when it is asked for one.
"""

import inspect
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from google.protobuf.json_format import MessageToJson

from feast.base_feature_view import BaseFeatureView
from feast.context.errors import TokenBudgetExceededError
from feast.context.prompt_template import PromptTemplate
from feast.context.render import (
    OverflowStrategy,
    TemplateVariable,
    VariableSpec,
    _variable_settings,
)
from feast.context.token_budget import TokenBudget
from feast.context.tokenizer import TokenizerName, get_tokenizer
from feast.data_source import RequestSource
from feast.errors import RequestDataNotFoundInEntityRowsException
from feast.feature_view_projection import FeatureViewProjection
from feast.protos.feast.core.Prompt_pb2 import Prompt as PromptProto
from feast.protos.feast.core.Prompt_pb2 import PromptMeta as PromptMetaProto
from feast.protos.feast.core.Prompt_pb2 import PromptSource as PromptSourceProto
from feast.protos.feast.core.Prompt_pb2 import PromptSpec as PromptSpecProto
from feast.protos.feast.core.Prompt_pb2 import (
    PromptVariable as PromptVariableProto,
)
from feast.transformation.udf_rehydrate import (
    _strip_leading_decorators,
    rehydrate_udf_from_source,
)
from feast.utils import _utc_now
from feast.version_utils import parse_version, version_tag

#: What a prompt function reads: feature views, projections such as
#: ``view[["a", "b"]]``, and request sources.
PromptSourceSpec = Union[BaseFeatureView, FeatureViewProjection, RequestSource]

_BUILT_IN_TOKENIZERS = {name.value for name in TokenizerName}
# Names the rebuilt function finds in scope (see udf_rehydrate._exec_namespace).
_REHYDRATE_NAMES = {"pd", "pandas", "np", "numpy"}


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt filled in for one entity row.

    Reads as its text, so it drops straight into an f-string or a model call.

    Attributes:
        name: Name of the prompt.
        version: Version of the prompt that was rendered. None for a prompt
            that was never applied.
        text: The prompt.
        tokens: Tokens ``text`` takes, counted with the prompt's tokenizer.
        truncated_variables: The template variables cut to fit the budget.
    """

    name: str
    version: Optional[int]
    text: str
    tokens: int
    truncated_variables: Tuple[str, ...] = ()

    @property
    def reference(self) -> str:
        """The prompt and version rendered, as ``name@vN``, for logs and traces."""
        return _reference(self.name, self.version)

    def to_dict(self) -> Dict[str, Any]:
        """The rendering as plain data, as the feature server returns it."""
        return {
            "name": self.name,
            "version": self.version,
            "text": self.text,
            "tokens": self.tokens,
            "truncated_variables": list(self.truncated_variables),
        }

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class PromptSource:
    """One input of a prompt function: a feature view's features, or request fields.

    Attributes:
        name: The feature view's name (``view@vN`` when pinned), or the request
            source's name.
        fields: The features read from the view, or the fields read from the
            request.
        request: Whether the source is request data rather than a feature view.
    """

    name: str
    fields: Tuple[str, ...]
    request: bool = False

    def values(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """This source's values in ``row``, keyed by feature or field name."""
        if self.request:
            missing = [field for field in self.fields if field not in row]
            if missing:
                raise RequestDataNotFoundInEntityRowsException(missing)
            return {field: row[field] for field in self.fields}
        keys = {field: f"{self.name}__{field}" for field in self.fields}
        missing = [key for key in keys.values() if key not in row]
        if missing:
            raise ValueError(
                f"Missing features {missing}. Read a prompt's features with "
                f"full_feature_names=True."
            )
        return {field: row[key] for field, key in keys.items()}


class Prompt:
    """A prompt function the registry stores and versions, and what it reads.

    Build one with the :func:`prompt` decorator.

    Attributes:
        name: Unique name of the prompt within its project.
        sources: The function's inputs, one per parameter, in order.
        function_name: Name of the prompt function.
        function_source: Source text of the prompt function, from which Feast
            rebuilds it wherever the prompt is rendered.
        template: Jinja2 template filled with the dict the function returns,
            or None when the function returns the prompt itself.
        max_tokens: Most tokens the prompt may take, or None for no limit.
        tokenizer: Name of the tokenizer that counts tokens.
        variables: How each template variable gives way when the prompt runs
            over its budget.
        overflow_strategy: What gives way when the prompt runs over its budget.
        description: A human-readable description.
        tags: User-defined metadata.
        owner: The owner of the prompt, typically the email of its maintainer.
        version: Version number the registry gave this definition. None until
            the prompt is applied.
        created_timestamp: When this version was registered.
    """

    name: str
    sources: List[PromptSource]
    function_name: str
    function_source: str
    template: Optional[str]
    max_tokens: Optional[int]
    tokenizer: str
    variables: Dict[str, TemplateVariable]
    overflow_strategy: Optional[OverflowStrategy]
    description: str
    tags: Dict[str, str]
    owner: str
    version: Optional[int]
    created_timestamp: Optional[datetime]

    def __init__(
        self,
        *,
        name: str,
        sources: Sequence[PromptSourceSpec],
        function: Callable[..., Any],
        template: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tokenizer: str = TokenizerName.CL100K_BASE.value,
        variables: Optional[Mapping[str, VariableSpec]] = None,
        overflow_strategy: Optional[Union[OverflowStrategy, str]] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        owner: str = "",
    ):
        """Creates a Prompt object. Most code uses the :func:`prompt` decorator.

        Raises:
            ValueError: A setting is invalid, or the function cannot be stored
                and rebuilt from its source text.
        """
        prompt_sources = _prompt_sources(sources)
        self._setup(
            name=name,
            sources=prompt_sources,
            function_name=function.__name__,
            function_source=_function_source(function, len(prompt_sources)),
            template=template,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            variables=variables,
            overflow_strategy=overflow_strategy,
            description=description,
            tags=tags,
            owner=owner,
        )
        # Render with the function rebuilt from its source, as the feature
        # server will, so a prompt that renders here renders there.
        self._function()

    def _setup(
        self,
        *,
        name: str,
        sources: List[PromptSource],
        function_name: str,
        function_source: str,
        template: Optional[str],
        max_tokens: Optional[int],
        tokenizer: str,
        variables: Optional[Mapping[str, VariableSpec]],
        overflow_strategy: Optional[Union[OverflowStrategy, str]],
        description: str,
        tags: Optional[Dict[str, str]],
        owner: str,
    ) -> None:
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError(
                f"max_tokens must be a positive integer or None, got {max_tokens!r}."
            )
        if not isinstance(tokenizer, str) or not (
            tokenizer in _BUILT_IN_TOKENIZERS or tokenizer.endswith("Tokenizer")
        ):
            raise ValueError(
                f"tokenizer must be one of {', '.join(sorted(_BUILT_IN_TOKENIZERS))}, "
                f"or the class path of a Tokenizer subclass; got {tokenizer!r}."
            )
        self._template: Optional[PromptTemplate] = None
        if template is not None:
            if max_tokens is None:
                raise ValueError(
                    "A prompt with a template needs max_tokens: the budget its "
                    "variables are cut to fit."
                )
            # Compiles the template and refuses settings for variables it
            # does not use.
            self._template = PromptTemplate(
                name=name,
                template_string=template,
                variables=_variable_settings(variables or {}),
                overflow_strategy=OverflowStrategy(
                    overflow_strategy or OverflowStrategy.TRUNCATE_LONGEST
                ),
            )
        elif variables or overflow_strategy:
            raise ValueError(
                "variables and overflow_strategy say how a template's variables "
                "are cut; pass template= too."
            )
        self.name = name
        self.sources = sources
        self.function_name = function_name
        self.function_source = function_source
        self.template = template
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.variables = dict(self._template.variables) if self._template else {}
        self.overflow_strategy = (
            self._template.overflow_strategy if self._template else None
        )
        self.description = description
        self.tags = dict(tags or {})
        self.owner = owner
        self.version = None
        self.created_timestamp = None
        self._callable: Optional[Callable[..., Any]] = None

    @property
    def features(self) -> List[str]:
        """The feature references the prompt reads, as ``"view:feature"``."""
        return [
            f"{source.name}:{field}"
            for source in self.sources
            if not source.request
            for field in source.fields
        ]

    @property
    def reference(self) -> str:
        """The prompt as ``name@vN``, or its bare name before it is applied."""
        return _reference(self.name, self.version)

    def render(self, values: Mapping[str, Any]) -> RenderedPrompt:
        """Call the prompt function on one row of values.

        ``FeatureStore.render_prompt`` calls this with features read online.
        Call it directly on rows from ``get_historical_features`` with
        ``full_feature_names=True`` to build the same prompts for training.

        Args:
            values: The prompt's features under their full names
                (``view__feature``), and its request fields.

        Raises:
            TemplateRenderError: A template variable has no value.
            TokenBudgetExceededError: The prompt cannot fit its budget.
        """
        result = self._function()(*[source.values(values) for source in self.sources])
        if self._template is not None:
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"Prompt function '{self.function_name}' fills a template, so "
                    f"it must return the template's variables as a dict, got "
                    f"{type(result).__name__}."
                )
            assert self.max_tokens is not None
            rendering = self._template.render(
                result, TokenBudget.of(self.max_tokens, self.tokenizer)
            )
            return RenderedPrompt(
                name=self.name,
                version=self.version,
                text=rendering.text,
                tokens=rendering.budget.consumed,
                truncated_variables=rendering.truncated_variables,
            )
        if not isinstance(result, str):
            raise TypeError(
                f"Prompt function '{self.function_name}' must return the prompt "
                f"as a string, got {type(result).__name__}. To fill a template "
                f"with a dict instead, pass template= to @prompt."
            )
        tokens = get_tokenizer(self.tokenizer).count_tokens(result)
        if self.max_tokens is not None and tokens > self.max_tokens:
            raise TokenBudgetExceededError(
                tokens,
                self.max_tokens,
                self.max_tokens,
                message=(
                    f"Prompt '{self.name}' is {tokens} tokens, over its budget of "
                    f"{self.max_tokens}. Pass template= to let variables be cut."
                ),
            )
        return RenderedPrompt(
            name=self.name, version=self.version, text=result, tokens=tokens
        )

    def _function(self) -> Callable[..., Any]:
        """The prompt function, rebuilt from its source text once."""
        if self._callable is None:
            function = rehydrate_udf_from_source(
                self.function_source, preferred_name=self.function_name
            )
            if function is None:
                raise ValueError(
                    f"Cannot rebuild prompt function '{self.function_name}' from "
                    f"its source text."
                )
            self._callable = function
        return self._callable

    def to_proto(self) -> PromptProto:
        """Converts a prompt object to its protobuf representation."""
        meta = PromptMetaProto(version_number=self.version or 0)
        if self.created_timestamp:
            meta.created_timestamp.FromDatetime(self.created_timestamp)
        return PromptProto(spec=self._spec_proto(), meta=meta)

    @classmethod
    def from_proto(cls, prompt_proto: PromptProto) -> "Prompt":
        """Converts a protobuf representation of a prompt to a prompt object.

        The function is rebuilt from its source text when the prompt is first
        rendered, not here, so listing prompts runs no prompt code.
        """
        spec = prompt_proto.spec
        prompt = cls.__new__(cls)
        prompt._setup(
            name=spec.name,
            sources=[
                PromptSource(
                    name=source.name,
                    fields=tuple(source.fields),
                    request=source.request,
                )
                for source in spec.sources
            ],
            function_name=spec.function_name,
            function_source=spec.function_source,
            template=spec.template or None,
            max_tokens=spec.max_tokens or None,
            tokenizer=spec.tokenizer,
            variables={
                variable: {
                    "priority": settings.priority,
                    "max_tokens": settings.max_tokens or None,
                    "truncation": settings.truncation,
                }
                for variable, settings in spec.variables.items()
            },
            overflow_strategy=spec.overflow_strategy or None,
            description=spec.description,
            tags=dict(spec.tags),
            owner=spec.owner,
        )
        meta = prompt_proto.meta
        prompt.version = meta.version_number or None
        if meta.HasField("created_timestamp"):
            prompt.created_timestamp = meta.created_timestamp.ToDatetime().replace(
                tzinfo=timezone.utc
            )
        return prompt

    def _spec_proto(self) -> PromptSpecProto:
        return PromptSpecProto(
            name=self.name,
            sources=[
                PromptSourceProto(
                    name=source.name, fields=source.fields, request=source.request
                )
                for source in self.sources
            ],
            function_name=self.function_name,
            function_source=self.function_source,
            template=self.template or "",
            max_tokens=self.max_tokens or 0,
            tokenizer=self.tokenizer,
            variables={
                variable: PromptVariableProto(
                    priority=settings.to_dict()["priority"],
                    max_tokens=settings.max_tokens or 0,
                    truncation=settings.truncation.value,
                )
                for variable, settings in self.variables.items()
            },
            overflow_strategy=(
                self.overflow_strategy.value if self.overflow_strategy else ""
            ),
            description=self.description,
            tags=self.tags,
            owner=self.owner,
        )

    def __eq__(self, other: object) -> bool:
        # Two definitions are equal when their specs are: the registry assigns
        # the version and timestamp, so they say nothing about the definition.
        if not isinstance(other, Prompt):
            return NotImplemented
        return self._spec_proto() == other._spec_proto()

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Prompt(name={self.name!r}, version={self.version!r})"

    def __str__(self) -> str:
        return str(MessageToJson(self.to_proto()))


def prompt(
    *,
    sources: Sequence[PromptSourceSpec],
    template: Optional[str] = None,
    max_tokens: Optional[int] = None,
    tokenizer: str = TokenizerName.CL100K_BASE.value,
    variables: Optional[Mapping[str, VariableSpec]] = None,
    overflow_strategy: Optional[Union[OverflowStrategy, str]] = None,
    name: Optional[str] = None,
    description: str = "",
    tags: Optional[Dict[str, str]] = None,
    owner: str = "",
) -> Callable[[Callable[..., Any]], Prompt]:
    """Turn a function into a :class:`Prompt` the registry stores and versions.

    The function takes one argument per source, in order: a dict of that
    source's features, or of its request fields, for one entity row. It returns
    the prompt as a string, or, with ``template``, the template's variables as
    a dict::

        @prompt(sources=[user_profile, location_request], max_tokens=300)
        def concierge(user_profile, request):
            return (
                f"You are a concierge for {user_profile['name']}. "
                f"Suggest places in {request['location']}."
            )

    Feast stores the function's source text and rebuilds the function from it
    wherever the prompt is rendered, with only ``pandas`` and ``numpy`` in
    scope. Import anything else inside the function.

    Args:
        sources: Feature views, projections such as ``view[["a", "b"]]``, and
            request sources, one per function parameter.
        template: A Jinja2 template the function's dict fills. Its variables
            are cut to fit ``max_tokens``, least important first.
        max_tokens: Most tokens the prompt may take. Without a template, a
            prompt over it is refused rather than cut.
        tokenizer: A built-in tokenizer name, such as ``"cl100k_base"``, or the
            class path of a ``Tokenizer`` subclass.
        variables: With ``template``: priority, max_tokens and truncation per
            variable, as ``feast.context.TemplateVariable`` or mappings.
        overflow_strategy: With ``template``: what gives way over budget.
        name: Name of the prompt; the function's name by default.
        description: A human-readable description.
        tags: User-defined metadata.
        owner: The owner of the prompt.

    Raises:
        ValueError: A setting is invalid, or the function uses names from
            outside its body, has the wrong number of parameters, or cannot be
            read as source text.
    """

    def decorator(function: Callable[..., Any]) -> Prompt:
        return Prompt(
            name=name or function.__name__,
            sources=sources,
            function=function,
            template=template,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            variables=variables,
            overflow_strategy=overflow_strategy,
            description=description,
            tags=tags,
            owner=owner,
        )

    return decorator


def parse_prompt_reference(reference: str) -> Tuple[str, Optional[int]]:
    """Split ``"name"`` or ``"name@vN"`` into the name and the version number.

    Raises:
        ValueError: The version part is not ``latest``, ``vN`` or ``versionN``.
    """
    name, _, version = reference.partition("@")
    if not name:
        raise ValueError(f"Prompt reference {reference!r} has no name.")
    is_latest, number = parse_version(version)
    return name, None if is_latest else number


def _assign_version(prompt: Prompt, versions: Iterable[PromptProto]) -> bool:
    """Give ``prompt`` its registry version, and say whether to store it.

    A definition equal to the latest version takes that version and needs no
    storing; any other definition becomes the next version.
    """
    latest = max(versions, key=lambda proto: proto.meta.version_number, default=None)
    if latest is not None and Prompt.from_proto(latest) == prompt:
        prompt.version = latest.meta.version_number
        prompt.created_timestamp = latest.meta.created_timestamp.ToDatetime().replace(
            tzinfo=timezone.utc
        )
        return False
    prompt.version = latest.meta.version_number + 1 if latest is not None else 1
    prompt.created_timestamp = _utc_now()
    return True


def _reference(name: str, version: Optional[int]) -> str:
    return f"{name}@{version_tag(version)}" if version else name


def _prompt_sources(sources: Sequence[PromptSourceSpec]) -> List[PromptSource]:
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise TypeError(
            f"sources must be a list of feature views and request sources, got "
            f"{sources!r}."
        )
    prompt_sources = []
    for source in sources:
        if isinstance(source, RequestSource):
            prompt_sources.append(
                PromptSource(
                    name=source.name,
                    fields=tuple(field.name for field in source.schema),
                    request=True,
                )
            )
            continue
        projection = (
            source.projection if isinstance(source, BaseFeatureView) else source
        )
        if not isinstance(projection, FeatureViewProjection):
            raise TypeError(
                f"Expected a feature view, a projection or a request source, got "
                f"{source!r}."
            )
        if not projection.features:
            raise ValueError(
                f"Feature view {projection.name!r} has no features yet; its schema "
                f"is inferred when it is applied. Declare its schema, or select "
                f"features with {projection.name}[[...]]."
            )
        prompt_sources.append(
            PromptSource(
                name=projection.name_to_use(),
                fields=tuple(feature.name for feature in projection.features),
            )
        )
    if all(source.request for source in prompt_sources):
        raise ValueError("A prompt reads at least one feature view.")
    names = [source.name for source in prompt_sources]
    if len(set(names)) != len(names):
        raise ValueError(f"Sources are listed more than once: {names}.")
    return prompt_sources


def _function_source(function: Callable[..., Any], source_count: int) -> str:
    """The function's source text, once it is sure to rebuild on its own."""
    name = getattr(function, "__name__", None)
    if not inspect.isfunction(function) or name == "<lambda>":
        raise ValueError(
            f"@prompt needs a function defined with def, got {function!r}."
        )
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Cannot read the source of prompt function '{name}'. Define it in a "
            f"file, not an interactive session."
        ) from e

    parameters = [
        parameter
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(parameters) != source_count:
        raise ValueError(
            f"Prompt function '{name}' takes {len(parameters)} arguments, but "
            f"@prompt has {source_count} sources. It takes one argument per "
            f"source, in order."
        )

    closure = inspect.getclosurevars(function)
    outside = sorted(
        set(closure.nonlocals) | (set(closure.globals) - _REHYDRATE_NAMES - {name})
    )
    if outside:
        raise ValueError(
            f"Prompt function '{name}' uses {', '.join(outside)} from outside its "
            f"body. Feast rebuilds the function from its source text when it "
            f"renders the prompt, so import or define them inside the function."
        )
    return _strip_leading_decorators(textwrap.dedent(source))
