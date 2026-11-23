"""Rendering prompt templates within a token budget.

A template decides what a prompt says; the budget decides how much of it a
model gets to read. :func:`render_with_budget` renders a Jinja2 template and,
when the prompt runs past the budget, cuts its variables down rather than the
words around them::

    rendering = render_with_budget(
        template="templates/recommendation.jinja2",
        context={
            "user_name": user_profile["name"],
            "watch_history": lambda: format_history(watch_events),
            "candidates": format_items(candidate_items),
        },
        budget=TokenBudget(max_tokens=4096),
        overflow_strategy="truncate_longest",
    )
    prompt = rendering.text

Cuts fall on token boundaries as the budget's own tokenizer counts them, and
nothing but the template, the context and the budget decides them, so an
OnDemandFeatureView renders the same prompt during training as it does at
serving time.

Jinja2 is needed only to render; the rest of ``feast.context`` works without it.
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from feast.context.errors import TemplateRenderError, TokenBudgetExceededError
from feast.context.priority_select import Priority
from feast.context.token_budget import TokenBudget
from feast.context.tokenizer import Tokenizer
from feast.errors import FeastExtrasDependencyImportError

#: Put in place of the text a truncation removed.
DEFAULT_TRUNCATION_MARKER = "..."

#: Compiled inline templates kept for reuse, keyed by their source.
_TEMPLATE_CACHE_SIZE = 256

#: A one-line string without Jinja2 syntax that ends like this names a file.
_FILE_SUFFIX = re.compile(r"\.[A-Za-z0-9]+$")
_JINJA_DELIMITERS = ("{{", "{%", "{#")

#: The least share of a value's text the prompt must show for the value to be
#: cut. One the template only tests or measures, say with ``length``, stays.
_MIN_RENDERED_SHARE = 0.5


class Truncation(str, Enum):
    """Which part of a text too long for its allowance is cut."""

    #: Cut the end, keeping the start: for text that leads with what matters,
    #: such as a history listed newest first.
    TAIL = "tail"
    #: Cut the middle, keeping the start and the end: for a document whose
    #: opening and conclusion carry it.
    MIDDLE = "middle"
    #: Cut the start, keeping the end: for a history listed oldest first, or a
    #: transcript whose latest turns matter most.
    HEAD = "head"


class OverflowStrategy(str, Enum):
    """What gives way when a rendered template runs past its budget."""

    #: Cut the longest text variable first, down to the next longest, and so
    #: on until the prompt fits.
    TRUNCATE_LONGEST = "truncate_longest"
    #: Cut text variables from the lowest priority up, emptying each priority
    #: before touching the next, the longest first among equals.
    TRUNCATE_LOWEST_PRIORITY = "truncate_lowest_priority"
    #: Cut nothing, and fail the render instead.
    ERROR = "error"


@dataclass(frozen=True)
class TemplateVariable:
    """How a template variable gives way when a prompt runs over its budget.

    Attributes:
        priority: How readily the variable is cut. A ``critical`` variable is
            never cut to make room; the render fails instead.
        max_tokens: The most tokens the variable's text may take, however much
            room the prompt has. None leaves it uncapped.
        truncation: Which part of the text is cut when it is.
    """

    priority: Priority = Priority.MEDIUM
    max_tokens: Optional[int] = None
    truncation: Truncation = Truncation.TAIL

    def __post_init__(self) -> None:
        object.__setattr__(self, "priority", Priority.parse(self.priority))
        object.__setattr__(self, "truncation", Truncation(self.truncation))
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError(
                f"max_tokens must be a positive integer or None, got "
                f"{self.max_tokens!r}."
            )

    @classmethod
    def of(cls, spec: "VariableSpec") -> "TemplateVariable":
        """Normalize a variable, or a mapping of its settings."""
        if isinstance(spec, TemplateVariable):
            return spec
        if isinstance(spec, Mapping):
            unknown = [key for key in spec if key not in _VARIABLE_SETTINGS]
            if unknown:
                raise ValueError(
                    f"Unknown variable settings {', '.join(map(repr, unknown))}. "
                    f"Settings are: {', '.join(_VARIABLE_SETTINGS)}."
                )
            return cls(**spec)
        raise TypeError(
            f"Expected a TemplateVariable or a mapping of its settings, got {spec!r}."
        )

    def to_dict(self) -> dict[str, Any]:
        """The settings as plain data, which :meth:`of` reads back."""
        return {
            "priority": str(self.priority),
            "max_tokens": self.max_tokens,
            "truncation": self.truncation.value,
        }


_VARIABLE_SETTINGS = ("priority", "max_tokens", "truncation")

#: The settings a variable has when none are given.
_DEFAULT_VARIABLE = TemplateVariable()

#: A variable's settings, or the mapping shorthand for them.
VariableSpec = Union[TemplateVariable, Mapping[str, Any]]

#: A template: Jinja2 source, or the path of a file that holds it.
TemplateSpec = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class Rendering:
    """The outcome of a render: the prompt, what it cost, and what gave way.

    Reads as its text, so a rendering drops straight into an f-string::

        prompt = f"{rendering}"

    Attributes:
        text: The rendered prompt.
        budget: The budget after charging :attr:`text`.
        truncated_variables: The variables cut to fit, in context order.
    """

    text: str
    budget: TokenBudget
    truncated_variables: tuple[str, ...] = ()

    @property
    def truncated(self) -> bool:
        """Whether any variable was cut."""
        return bool(self.truncated_variables)

    def __str__(self) -> str:
        return self.text


def truncate(
    text: str,
    budget: TokenBudget,
    *,
    truncation: Truncation = Truncation.TAIL,
    marker: str = DEFAULT_TRUNCATION_MARKER,
) -> str:
    """``text`` cut down to what remains of ``budget``; unchanged if it fits.

    The cut falls on token boundaries and ``marker`` takes the place of what was
    removed. The marker is paid for out of the budget too, so the result always
    fits: when the budget cannot hold the marker and some of the text besides,
    the text is cut without one, and a spent budget leaves nothing.

    The budget itself is not charged. Pass the result to
    :meth:`TokenBudget.consume` to spend it.

    Args:
        text: The text to fit.
        budget: Only what remains of it is available.
        truncation: Which part of the text is cut.
        marker: Put where text was removed; ``""`` for nothing.

    Raises:
        TypeError: ``text`` or ``marker`` is not a string.
        ValueError: ``truncation`` is not a known name.
    """
    return _fit(text, budget, Truncation(truncation), marker)[0]


def render_with_budget(
    template: TemplateSpec,
    context: Mapping[str, Any],
    budget: TokenBudget,
    *,
    overflow_strategy: OverflowStrategy = OverflowStrategy.TRUNCATE_LONGEST,
    variables: Optional[Mapping[str, VariableSpec]] = None,
    marker: str = DEFAULT_TRUNCATION_MARKER,
) -> Rendering:
    """Render ``template`` with ``context`` within what remains of ``budget``.

    ``template`` is Jinja2 source or a template file. A path, or a string on one
    line without Jinja2 syntax that ends in a suffix such as ``.jinja2``, is
    read from that file, relative to the working directory, and can include
    templates beside it. Source is compiled once and cached.

    Rendering is strict: a variable the context lacks raises rather than
    rendering blank, though ``default`` and ``is defined`` still work. Loops,
    conditionals and filters render structured feature values.

    A callable in ``context`` is a lazy variable: called once, with no
    arguments, when the template refers to it, and never when it does not.

    Variables give way to the budget in two steps. Each variable with a
    ``max_tokens`` is first cut to that many tokens. Then, while the prompt
    still runs over, ``overflow_strategy`` decides what gives way:

    - ``truncate_longest``: the longest text variable is cut first, down to the
      length of the next longest, and so on until the prompt fits.
    - ``truncate_lowest_priority``: variables are cut from the lowest priority
      up, each priority emptied before the next is touched, the longest first
      among equals.
    - ``error``: nothing is cut, and the render fails.

    Only text the prompt shows gives way. The template's own words, values that
    are not strings, and ``critical`` variables are never cut to make room: if
    the prompt cannot fit without cutting them, the render fails. Each cut
    leaves ``marker`` in place of what it removed.

    Args:
        template: Jinja2 source, or a template file.
        context: The variables the template refers to; callables are lazy.
        budget: The allowance to fill. Anything already consumed on it stays
            consumed; only what remains is available here.
        overflow_strategy: What gives way when the prompt runs over.
        variables: Priority, max_tokens and truncation per variable, as
            :class:`TemplateVariable` or mappings of its settings. A variable
            left out has medium priority, no cap, and loses its tail.
        marker: Put where text was cut; ``""`` for nothing.

    Returns:
        A :class:`Rendering` of the prompt, the budget left after it, and the
        variables cut to fit.

    Raises:
        FileNotFoundError: ``template`` names a file that does not exist.
        TemplateRenderError: The template does not compile, or rendering it
            fails, as it does on an undefined variable.
        TokenBudgetExceededError: The prompt runs over and ``overflow_strategy``
            is ``error``, or it runs over even with every variable that may be
            cut emptied.
        TypeError: ``template`` is neither source nor a path, ``max_tokens`` is
            set on a value that is not text, or ``marker`` is not a string.
        ValueError: ``overflow_strategy`` or a variable setting is invalid.
    """
    # TODO: Make room by compressing a variable, not only by cutting its text.
    # An LLM summarizer would rewrite it shorter while keeping what it says,
    # and bucketing would fold its detail into coarser ranges, both of which
    # keep more meaning per token than dropping the tail. Either costs more
    # than truncation, and a summarizer is neither free nor deterministic, so
    # it would be a strategy a caller asks for rather than the default.
    strategy = OverflowStrategy(overflow_strategy)
    settings = _variable_settings(variables or {})
    if not isinstance(marker, str):
        raise TypeError(f"marker must be a string, got {marker!r}.")
    compiled, referenced = _load(template)
    return _render(compiled, referenced, context, budget, strategy, settings, marker)


def _render(
    compiled: Any,
    referenced: Callable[[], Optional[frozenset[str]]],
    context: Mapping[str, Any],
    budget: TokenBudget,
    strategy: OverflowStrategy,
    settings: Mapping[str, TemplateVariable],
    marker: str,
) -> Rendering:
    """Render a compiled template, cutting variables until the prompt fits."""
    # Finding what a file template reads means parsing it; do that once at most.
    referenced_once = lru_cache(maxsize=None)(referenced)
    originals = _resolve_lazy(context, referenced_once)
    values = dict(originals)

    # Caps come first: a variable's own limit holds however much room is left.
    allowances: dict[str, int] = {}
    for name, setting in settings.items():
        if setting.max_tokens is None or name not in values:
            continue
        if not isinstance(values[name], str):
            raise TypeError(
                f"Variable '{name}' has max_tokens, which caps text, but its value "
                f"is {type(values[name]).__name__}."
            )
        _cut(
            values,
            originals,
            {name: setting.max_tokens},
            settings,
            budget.tokenizer,
            marker,
        )
        if values[name] is not originals[name]:
            allowances[name] = setting.max_tokens

    text = _render_text(compiled, values)
    tokens = budget.count_tokens(text)
    if tokens > budget.remaining:
        if strategy is OverflowStrategy.ERROR:
            raise TokenBudgetExceededError(
                tokens,
                budget.remaining,
                budget.max_tokens,
                message=(
                    f"The prompt needs {tokens} tokens with only {budget.remaining} "
                    f"of {budget.max_tokens} left, and overflow_strategy is 'error'."
                ),
            )
        text, tokens = _give_way(
            compiled,
            text,
            tokens,
            values,
            originals,
            allowances,
            settings,
            budget,
            strategy,
            marker,
            referenced_once(),
        )
    return Rendering(
        text=text,
        budget=budget.consume_tokens(tokens),
        truncated_variables=tuple(name for name in originals if name in allowances),
    )


def _give_way(
    compiled: Any,
    text: str,
    tokens: int,
    values: dict[str, Any],
    originals: Mapping[str, Any],
    allowances: dict[str, int],
    settings: Mapping[str, TemplateVariable],
    budget: TokenBudget,
    strategy: OverflowStrategy,
    marker: str,
    referenced: Optional[frozenset[str]],
) -> tuple[str, int]:
    """Cut variables as ``strategy`` orders until the prompt fits."""
    shares = _rendered_shares(compiled, text, values, settings, referenced)
    sizes = {name: budget.count_tokens(values[name]) for name in shares}
    while True:
        limits = _allocate(sizes, shares, tokens - budget.remaining, strategy, settings)
        exhausted = limits is None
        if limits is None:
            limits = dict.fromkeys(sizes, 0)
        # Only what actually shrinks: a variable in a priority the strategy has
        # not reached, or already under the level, keeps the text it has.
        cuts = {name: limit for name, limit in limits.items() if limit < sizes[name]}
        if not cuts:
            # Nothing left to give: rendering again would repeat this round for
            # ever. A tokenizer that counts some non-empty text as no tokens at
            # all lands here on the first round.
            raise _cannot_fit(tokens, budget)
        sizes.update(cuts)
        allowances.update(cuts)
        _cut(values, originals, cuts, settings, budget.tokenizer, marker)
        text = _render_text(compiled, values)
        tokens = budget.count_tokens(text)
        if tokens <= budget.remaining:
            return text, tokens
        if exhausted:
            raise _cannot_fit(tokens, budget)


def _rendered_shares(
    compiled: Any,
    text: str,
    values: Mapping[str, Any],
    settings: Mapping[str, TemplateVariable],
    referenced: Optional[frozenset[str]],
) -> dict[str, float]:
    """How much of each cuttable value the prompt shows, per character of it.

    Rendering with a value emptied measures what it adds, which counts a value
    written twice double and one in an untaken branch not at all.
    """
    shares = {}
    for name, value in values.items():
        if (
            not isinstance(value, str)
            or not value
            or (referenced is not None and name not in referenced)
            or settings.get(name, _DEFAULT_VARIABLE).priority is Priority.CRITICAL
        ):
            continue
        emptied = _render_text(compiled, {**values, name: ""})
        share = (len(text) - len(emptied)) / len(value)
        if share >= _MIN_RENDERED_SHARE:
            shares[name] = share
    return shares


def _allocate(
    sizes: Mapping[str, int],
    shares: Mapping[str, float],
    needed: int,
    strategy: OverflowStrategy,
    settings: Mapping[str, TemplateVariable],
) -> Optional[dict[str, int]]:
    """Token limits that save ``needed`` tokens, or None if emptying cannot."""
    if strategy is OverflowStrategy.TRUNCATE_LOWEST_PRIORITY:
        priority = {
            name: settings.get(name, _DEFAULT_VARIABLE).priority for name in sizes
        }
        tiers = [
            [name for name in sizes if priority[name] is level]
            for level in sorted(set(priority.values()))
        ]
    else:
        tiers = [list(sizes)]

    limits = dict(sizes)
    remaining_need: float = needed
    for tier in tiers:
        capacity = sum(shares[name] * sizes[name] for name in tier)
        if capacity >= remaining_need:
            level = _water_level(
                [sizes[name] for name in tier],
                [shares[name] for name in tier],
                remaining_need,
            )
            for name in tier:
                limits[name] = min(sizes[name], level)
            return limits
        for name in tier:
            limits[name] = 0
        remaining_need -= capacity
    return None


def _water_level(sizes: list[int], shares: list[float], needed: float) -> int:
    """The highest level that cutting every size down to still saves ``needed``.

    Cutting to a level shortens the longest first, down to the next longest,
    which is truncate_longest. Assumes cutting to zero saves enough.
    """

    def saved(level: int) -> float:
        return sum(share * max(0, size - level) for size, share in zip(sizes, shares))

    enough, too_little = 0, max(sizes)
    while too_little - enough > 1:
        middle = (enough + too_little) // 2
        if saved(middle) >= needed:
            enough = middle
        else:
            too_little = middle
    return enough


def _cut(
    values: dict[str, Any],
    originals: Mapping[str, Any],
    limits: Mapping[str, int],
    settings: Mapping[str, TemplateVariable],
    tokenizer: Tokenizer,
    marker: str,
) -> None:
    # Always from the original, so markers from earlier cuts never pile up.
    for name, limit in limits.items():
        values[name] = _fit(
            originals[name],
            TokenBudget(max_tokens=limit, tokenizer=tokenizer),
            settings.get(name, _DEFAULT_VARIABLE).truncation,
            marker,
        )[0]


def _cannot_fit(tokens: int, budget: TokenBudget) -> TokenBudgetExceededError:
    return TokenBudgetExceededError(
        tokens,
        budget.remaining,
        budget.max_tokens,
        message=(
            f"The prompt needs {tokens} tokens with only {budget.remaining} of "
            f"{budget.max_tokens} left, even with every variable that may be cut "
            f"emptied. Its own text, critical variables and values that are not "
            f"text do not fit."
        ),
    )


def _resolve_lazy(
    context: Mapping[str, Any], referenced: Callable[[], Optional[frozenset[str]]]
) -> dict[str, Any]:
    values = dict(context)
    lazy = [name for name, value in values.items() if callable(value)]
    if lazy:
        used = referenced()
        for name in lazy:
            if used is None or name in used:
                values[name] = values[name]()
    return values


def _variable_settings(variables: Mapping[str, Any]) -> dict[str, TemplateVariable]:
    if not isinstance(variables, Mapping):
        raise TypeError(
            f"variables must map variable names to settings, got "
            f"{type(variables).__name__}."
        )
    settings = {}
    for name, spec in variables.items():
        if not isinstance(name, str):
            raise TypeError(f"Variable names must be strings, got {name!r}.")
        settings[name] = TemplateVariable.of(spec)
    return settings


def _load(
    template: TemplateSpec,
) -> tuple[Any, Callable[[], Optional[frozenset[str]]]]:
    """The compiled template, and how to find the variables it reads."""
    path = _template_file(template)
    if path is None:
        source = str(template)
        return _compile_source(source), partial(_template_variables, source)

    environment = _file_environment(os.path.abspath(path.parent))
    jinja2 = _jinja2()
    try:
        compiled = environment.get_template(path.name)
    except jinja2.TemplateNotFound as e:
        if e.name == path.name:
            raise FileNotFoundError(f"Template file '{path}' does not exist.") from None
        raise TemplateRenderError(e) from e
    except jinja2.TemplateError as e:
        raise TemplateRenderError(e) from e
    return compiled, partial(_file_template_variables, environment, path.name)


def _template_file(template: TemplateSpec) -> Optional[Path]:
    if isinstance(template, os.PathLike):
        return Path(template)
    if not isinstance(template, str):
        raise TypeError(
            f"template must be Jinja2 source or a path, got {type(template).__name__}."
        )
    name = template.strip()
    if (
        "\n" in name
        or "\r" in name
        or any(delimiter in name for delimiter in _JINJA_DELIMITERS)
        or not _FILE_SUFFIX.search(name)
    ):
        return None
    return Path(name)


def _jinja2() -> Any:
    try:
        import jinja2
        import jinja2.meta
    except ImportError as e:
        raise FeastExtrasDependencyImportError("llm", str(e)) from e
    return jinja2


def _environment_options(jinja2: Any) -> dict[str, Any]:
    # Prompts are not HTML, so nothing is escaped, and a variable missing from
    # the context fails the render rather than dropping out of the prompt.
    return {"undefined": jinja2.StrictUndefined, "autoescape": False}


@lru_cache(maxsize=None)
def _inline_environment() -> Any:
    jinja2 = _jinja2()
    return jinja2.Environment(**_environment_options(jinja2))


@lru_cache(maxsize=64)
def _file_environment(directory: str) -> Any:
    # The loader caches compiled files and reloads one that has changed.
    jinja2 = _jinja2()
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(directory), **_environment_options(jinja2)
    )


@lru_cache(maxsize=_TEMPLATE_CACHE_SIZE)
def _compile_source(source: str) -> Any:
    # Compiling costs far more than rendering, and a feature view renders the
    # same source for every row it transforms.
    jinja2 = _jinja2()
    try:
        return _inline_environment().from_string(source)
    except jinja2.TemplateError as e:
        raise TemplateRenderError(e) from e


@lru_cache(maxsize=_TEMPLATE_CACHE_SIZE)
def _template_variables(source: str) -> frozenset[str]:
    """The variables ``source`` reads from its context.

    Names the template binds itself, such as loop variables, and Jinja2's
    globals are not among them.
    """
    jinja2 = _jinja2()
    try:
        ast = _inline_environment().parse(source)
    except jinja2.TemplateError as e:
        raise TemplateRenderError(e) from e
    return frozenset(jinja2.meta.find_undeclared_variables(ast))


def _file_template_variables(environment: Any, name: str) -> Optional[frozenset[str]]:
    """The variables a template file and those it includes read, if knowable.

    None when a template it includes is chosen at render time.
    """
    jinja2 = _jinja2()
    found: set[str] = set()
    pending, seen = [name], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            source, _, _ = environment.loader.get_source(environment, current)
            ast = environment.parse(source)
        except jinja2.TemplateError as e:
            raise TemplateRenderError(e) from e
        found |= jinja2.meta.find_undeclared_variables(ast)
        for included in jinja2.meta.find_referenced_templates(ast):
            if included is None:
                return None
            pending.append(included)
    return frozenset(found)


def _render_text(compiled: Any, values: Mapping[str, Any]) -> str:
    jinja2 = _jinja2()
    try:
        return compiled.render(values)
    except jinja2.TemplateError as e:
        raise TemplateRenderError(e) from e


def _fit(
    text: str, budget: TokenBudget, truncation: Truncation, marker: str
) -> tuple[str, int]:
    """``text`` cut to fit ``budget``, and the tokens the result costs."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {text!r}.")
    if not isinstance(marker, str):
        raise TypeError(f"marker must be a string, got {marker!r}.")

    tokens = budget.count_tokens(text)
    allowance = budget.remaining
    if tokens <= allowance:
        return text, tokens

    marker_tokens = budget.count_tokens(marker)
    if marker_tokens >= allowance:
        # A marker with no text beside it says nothing; cut without one.
        marker, marker_tokens = "", 0
    kept = allowance - marker_tokens
    while kept > 0:
        fitted = _cut_text(budget.tokenizer, text, kept, truncation, marker)
        tokens = budget.count_tokens(fitted)
        if tokens <= allowance:
            return fitted, tokens
        # Text can tokenize differently once joined across a cut, costing more
        # than its pieces did apart; give up the excess and cut again.
        kept -= tokens - allowance
    return "", 0


def _cut_text(
    tokenizer: Tokenizer, text: str, tokens: int, truncation: Truncation, marker: str
) -> str:
    """Keep ``tokens`` tokens of ``text`` around the cut ``truncation`` makes."""
    if truncation is Truncation.TAIL:
        return tokenizer.prefix(text, tokens) + marker
    if truncation is Truncation.HEAD:
        return marker + tokenizer.suffix(text, tokens)
    start = tokenizer.prefix(text, (tokens + 1) // 2)
    # The end is taken from what the start left, so the two cannot overlap.
    end = tokenizer.suffix(text[len(start) :], tokens // 2)
    return start + marker + end
