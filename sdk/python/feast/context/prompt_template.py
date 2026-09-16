"""Reusable prompt templates with a name, a version, and a budget per variable.

A :class:`PromptTemplate` turns a Jinja2 prompt into a value a feature
repository defines once, versions for A/B tests, and hands to other systems::

    template = PromptTemplate(
        name="recommendation_v2",
        template_string=(
            "You are a recommendation engine for {{ user_name }}. "
            "Recent activity: {{ watch_history }}. "
            "Please rank these candidates: {{ candidates }}"
        ),
        variables={
            "user_name": {"priority": "critical", "max_tokens": 50},
            "watch_history": {"priority": "high", "max_tokens": 2000},
            "candidates": {"priority": "high", "max_tokens": 1500},
        },
    )
    rendering = template.render(context=features, budget=TokenBudget(max_tokens=4096))

It serializes to JSON or YAML, so a model server's transformer can load the
template from its configuration and render the same prompt a feature view
rendered for training.
"""

import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from feast.context.render import (
    DEFAULT_TRUNCATION_MARKER,
    OverflowStrategy,
    Rendering,
    TemplateVariable,
    VariableSpec,
    _compile_source,
    _render,
    _template_variables,
    _variable_settings,
)
from feast.context.token_budget import TokenBudget

_REQUIRED_FIELDS = ("name", "template_string")
_OPTIONAL_FIELDS = ("version", "variables", "overflow_strategy")


@dataclass(frozen=True, kw_only=True)
class PromptTemplate:
    """A named, versioned Jinja2 prompt, and how its variables give way to a budget.

    Attributes:
        name: Identifies the prompt, in experiment tracking for instance.
        template_string: Jinja2 source, rendered as strictly as
            :func:`render_with_budget` renders it.
        version: Tells revisions of the prompt apart, such as the arms of an
            A/B test. None leaves the template unversioned.
        variables: Priority, max_tokens and truncation per variable, given as
            :class:`TemplateVariable` or as mappings of its settings. Variables
            left out take the defaults; each one named must appear in the
            template.
        overflow_strategy: What gives way when a render runs past its budget.
    """

    name: str
    template_string: str = field(repr=False)
    version: Optional[str] = None
    variables: Mapping[str, TemplateVariable] = field(default_factory=dict)
    overflow_strategy: OverflowStrategy = OverflowStrategy.TRUNCATE_LONGEST

    def __post_init__(self) -> None:
        _check_text("name", self.name)
        _check_text("template_string", self.template_string)
        if self.version is not None:
            _check_text("version", self.version)
        object.__setattr__(
            self, "overflow_strategy", OverflowStrategy(self.overflow_strategy)
        )
        object.__setattr__(self, "variables", _variable_settings(self.variables))
        # Fail on a broken template when it is defined, not when it is served.
        _compile_source(self.template_string)
        unused = sorted(
            self.variables.keys() - _template_variables(self.template_string)
        )
        if unused:
            raise ValueError(
                f"Prompt template '{self.name}' sets variables its template does not "
                f"use: {', '.join(unused)}."
            )

    @classmethod
    def from_file(
        cls,
        path: Union[str, "os.PathLike[str]"],
        *,
        name: str,
        version: Optional[str] = None,
        variables: Optional[Mapping[str, VariableSpec]] = None,
        overflow_strategy: OverflowStrategy = OverflowStrategy.TRUNCATE_LONGEST,
    ) -> "PromptTemplate":
        """Build a template from the Jinja2 source in a file.

        The file is read here, once: the template keeps its text, so it
        serializes whole and renders the same wherever it is loaded. For that
        reason the file cannot include or extend other templates.
        """
        return cls(
            name=name,
            template_string=Path(path).read_text(encoding="utf-8"),
            version=version,
            variables=_variable_settings(variables or {}),
            overflow_strategy=overflow_strategy,
        )

    def render(self, context: Mapping[str, Any], budget: TokenBudget) -> Rendering:
        """Render the prompt with ``context`` within what remains of ``budget``.

        Renders as :func:`render_with_budget` does, with this template's
        variables and overflow strategy: callables in ``context`` are lazy, caps
        apply first, and ``overflow_strategy`` decides what gives way after.

        Raises:
            TemplateRenderError: Rendering failed, as it does on a variable
                missing from ``context``.
            TokenBudgetExceededError: The prompt cannot fit the budget.
            TypeError: A variable with ``max_tokens`` has a value that is not
                text.
        """
        return _render(
            _compile_source(self.template_string),
            partial(_template_variables, self.template_string),
            context,
            budget,
            self.overflow_strategy,
            self.variables,
            DEFAULT_TRUNCATION_MARKER,
        )

    def to_dict(self) -> dict[str, Any]:
        """The template as plain data, which :meth:`from_dict` reads back."""
        return {
            "name": self.name,
            "version": self.version,
            "template_string": self.template_string,
            "variables": {
                variable: settings.to_dict()
                for variable, settings in self.variables.items()
            },
            "overflow_strategy": self.overflow_strategy.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptTemplate":
        """Build a template from :meth:`to_dict` output or hand-written config.

        Raises:
            TypeError: ``data`` is not a mapping.
            ValueError: A required field is missing, a field is not one a
                template has, or a value is invalid.
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                f"A prompt template is a mapping of its fields, got "
                f"{type(data).__name__}."
            )
        missing = [name for name in _REQUIRED_FIELDS if name not in data]
        if missing:
            raise ValueError(f"Prompt template config is missing {', '.join(missing)}.")
        known = _REQUIRED_FIELDS + _OPTIONAL_FIELDS
        unknown = [key for key in data if key not in known]
        if unknown:
            raise ValueError(
                f"Prompt template config has unknown fields "
                f"{', '.join(map(repr, unknown))}. Fields are: {', '.join(known)}."
            )
        fields = dict(data)
        if fields.get("variables") is None:
            fields.pop("variables", None)
        return cls(**fields)

    def to_json(self) -> str:
        """The template as a single line of JSON, fit for an environment variable."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "PromptTemplate":
        """Build a template from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))

    def to_yaml(self) -> str:
        """The template as YAML.

        A multi-line template is written as a literal block, so it reads as it
        was typed, unless a line ends in a space; YAML then quotes it instead.
        """
        return yaml.dump(
            self.to_dict(), Dumper=_TemplateDumper, sort_keys=False, allow_unicode=True
        )

    @classmethod
    def from_yaml(cls, text: str) -> "PromptTemplate":
        """Build a template from YAML, refusing tags that construct objects."""
        return cls.from_dict(yaml.safe_load(text))

    def __hash__(self) -> int:
        # By hand, since variables is a dict; the generated hash would raise.
        return hash(
            (
                self.name,
                self.version,
                self.template_string,
                frozenset(self.variables.items()),
                self.overflow_strategy,
            )
        )


class _TemplateDumper(yaml.SafeDumper):
    """Writes multi-line strings as literal blocks rather than escaped lines."""


def _represent_str(dumper: yaml.SafeDumper, text: str) -> yaml.ScalarNode:
    style = "|" if "\n" in text else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", text, style=style)


_TemplateDumper.add_representer(str, _represent_str)


def _check_text(attribute: str, value: object) -> None:
    if isinstance(value, str) and value.strip():
        return
    hint = ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # YAML reads an unquoted version: 2.1 as a number.
        hint = f" Quote it in YAML: {attribute}: '{value}'."
    raise ValueError(f"{attribute} must be a non-empty string, got {value!r}.{hint}")
