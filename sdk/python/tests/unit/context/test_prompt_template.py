import dataclasses
from typing import Any

import pytest
import yaml

from feast.context import (
    OverflowStrategy,
    Priority,
    PromptTemplate,
    TemplateVariable,
    TokenBudget,
    Truncation,
)
from feast.context.errors import TemplateRenderError, TokenBudgetExceededError
from feast.context.tokenizer import ApproximateTokenizer

# Four characters per token: "abcd" is one token under the estimate.
ESTIMATING = ApproximateTokenizer()

RECOMMENDATION = """You are a recommendation engine for {{user_name}}.

Recent activity:
{{watch_history}}

Please rank these candidates:
{{candidates}}"""

VARIABLES = {
    "user_name": {"priority": "critical", "max_tokens": 50},
    "watch_history": {"priority": "high", "max_tokens": 2000},
    "candidates": {"priority": "high", "max_tokens": 1500},
}

CONTEXT = {
    "user_name": "Ann",
    "watch_history": "- Stalker\n- Solaris",
    "candidates": "- Mirror\n- Nostalghia",
}


def budget(max_tokens: int = 1000) -> TokenBudget:
    """A budget on the deterministic estimate, so counts need no tiktoken."""
    return TokenBudget(max_tokens=max_tokens, tokenizer=ESTIMATING)


def recommendation(**overrides: Any) -> PromptTemplate:
    """The template from the story, with any field swapped out."""
    fields: dict[str, Any] = {
        "name": "recommendation_v2",
        "template_string": RECOMMENDATION,
        "variables": VARIABLES,
    }
    fields.update(overrides)
    return PromptTemplate(**fields)


class TestConstruction:
    def test_holds_its_fields(self):
        template = recommendation()
        assert template.name == "recommendation_v2"
        assert template.template_string == RECOMMENDATION
        assert template.version is None
        assert template.overflow_strategy is OverflowStrategy.TRUNCATE_LONGEST

    def test_resolves_variable_settings_into_template_variables(self):
        assert recommendation().variables == {
            "user_name": TemplateVariable(Priority.CRITICAL, 50),
            "watch_history": TemplateVariable(Priority.HIGH, 2000),
            "candidates": TemplateVariable(Priority.HIGH, 1500),
        }

    def test_accepts_template_variables_as_they_are(self):
        variables = {"user_name": TemplateVariable(Priority.CRITICAL, 50)}
        assert recommendation(variables=variables).variables == variables

    def test_carries_a_version_for_ab_tests(self):
        assert recommendation(version="short-context").version == "short-context"

    def test_equal_templates_hash_alike(self):
        reordered = recommendation(variables=dict(reversed(list(VARIABLES.items()))))
        assert len({recommendation(), reordered}) == 1

    def test_copies_the_variables(self):
        variables = dict(VARIABLES)
        template = recommendation(variables=variables)
        variables["extra"] = {"priority": "low"}
        assert "extra" not in template.variables

    def test_accepts_an_overflow_strategy_by_name(self):
        template = recommendation(overflow_strategy="truncate_lowest_priority")
        assert template.overflow_strategy is OverflowStrategy.TRUNCATE_LOWEST_PRIORITY

    def test_rejects_an_unknown_overflow_strategy(self):
        with pytest.raises(ValueError, match="truncate_randomly"):
            recommendation(overflow_strategy="truncate_randomly")

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            recommendation().version = "2"  # type: ignore[misc]

    def test_takes_keyword_arguments_only(self):
        with pytest.raises(TypeError):
            PromptTemplate("recommendation_v2", RECOMMENDATION)  # type: ignore[misc]

    @pytest.mark.parametrize("field", ["name", "template_string", "version"])
    def test_rejects_a_blank_text_field(self, field):
        with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
            recommendation(**{field: "  "})

    def test_a_numeric_version_is_told_to_quote_it(self):
        with pytest.raises(ValueError, match="Quote it in YAML: version: '2.1'"):
            recommendation(version=2.1)

    def test_rejects_a_template_that_does_not_compile(self):
        with pytest.raises(TemplateRenderError, match="at line 1"):
            PromptTemplate(name="broken", template_string="{{ oops }")

    def test_rejects_settings_for_a_variable_the_template_does_not_use(self):
        with pytest.raises(ValueError, match="does not use: watch_histroy"):
            recommendation(variables={"watch_histroy": {"priority": "low"}})

    def test_variables_without_settings_take_the_defaults(self):
        template = recommendation(variables={})
        assert template.render(CONTEXT, budget()).text.startswith(
            "You are a recommendation engine for Ann."
        )

    def test_rejects_invalid_variable_settings(self):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            recommendation(variables={"candidates": {"max_tokens": 0}})


class TestRender:
    def test_renders_the_prompt(self):
        rendering = recommendation().render(context=CONTEXT, budget=budget())
        assert rendering.text == (
            "You are a recommendation engine for Ann.\n\n"
            "Recent activity:\n- Stalker\n- Solaris\n\n"
            "Please rank these candidates:\n- Mirror\n- Nostalghia"
        )
        assert rendering.truncated is False

    def test_caps_each_variable_at_its_max_tokens(self):
        template = recommendation(
            variables={"watch_history": {"priority": "high", "max_tokens": 3}}
        )
        rendering = template.render({**CONTEXT, "watch_history": "w" * 40}, budget())
        assert "Recent activity:\nwwwwwwww...\n" in rendering.text
        assert rendering.truncated_variables == ("watch_history",)

    def test_cuts_the_lowest_priority_first_when_the_template_says_so(self):
        template = PromptTemplate(
            name="notes",
            template_string="{{ low }}{{ high }}",
            variables={"low": {"priority": "low"}, "high": {"priority": "high"}},
            overflow_strategy=OverflowStrategy.TRUNCATE_LOWEST_PRIORITY,
        )
        rendering = template.render({"low": "l" * 16, "high": "h" * 40}, budget(8))
        assert rendering.text == "h" * 28 + "..."

    def test_never_cuts_a_critical_variable(self):
        template = recommendation(variables={"user_name": {"priority": "critical"}})
        # Emptying the history and the candidates cannot make room for the name.
        with pytest.raises(TokenBudgetExceededError, match="every variable that may"):
            template.render({**CONTEXT, "user_name": "u" * 400}, budget(20))

    def test_calls_lazy_variables(self):
        rendering = recommendation().render(
            {**CONTEXT, "watch_history": lambda: "- Stalker"}, budget()
        )
        assert "Recent activity:\n- Stalker\n" in rendering.text

    def test_a_missing_variable_fails_the_render(self):
        with pytest.raises(TemplateRenderError, match="'candidates' is undefined"):
            recommendation().render({"user_name": "Ann", "watch_history": ""}, budget())

    def test_treats_its_template_as_source_even_when_it_looks_like_a_file(self):
        template = PromptTemplate(name="literal", template_string="notes.txt")
        assert template.render({}, budget()).text == "notes.txt"


class TestFromFile:
    def test_reads_the_template_from_a_file(self, tmp_path):
        path = tmp_path / "recommendation.jinja2"
        path.write_text(RECOMMENDATION, encoding="utf-8")
        template = PromptTemplate.from_file(
            path, name="recommendation_v2", version="2", variables=VARIABLES
        )
        assert template == recommendation(version="2")


class TestSerialization:
    @pytest.fixture
    def template(self):
        # Unicode, indentation, a line ending in a space and a trailing newline:
        # everything a serializer could quietly alter.
        return PromptTemplate(
            name="offer_email",
            version="3.0.1",
            template_string=(
                "  Écris à {{ customer }} 👋 \n\n"
                "{% for item in basket %}- {{ item }}\n{% endfor %}"
                "Budget: {{ spend }}\n"
            ),
            variables={
                "customer": {"priority": "critical", "max_tokens": 20},
                "basket": {"priority": "low", "truncation": "head"},
            },
            overflow_strategy=OverflowStrategy.TRUNCATE_LOWEST_PRIORITY,
        )

    def test_to_dict_holds_every_field(self, template):
        assert template.to_dict() == {
            "name": "offer_email",
            "version": "3.0.1",
            "template_string": template.template_string,
            "variables": {
                "customer": {
                    "priority": "critical",
                    "max_tokens": 20,
                    "truncation": "tail",
                },
                "basket": {"priority": "low", "max_tokens": None, "truncation": "head"},
            },
            "overflow_strategy": "truncate_lowest_priority",
        }

    @pytest.mark.parametrize(
        "dump, load",
        [
            ("to_dict", "from_dict"),
            ("to_json", "from_json"),
            ("to_yaml", "from_yaml"),
        ],
    )
    def test_round_trips_exactly(self, template, dump, load):
        loaded = getattr(PromptTemplate, load)(getattr(template, dump)())
        assert loaded == template
        assert loaded.template_string == template.template_string
        assert loaded.variables["basket"].truncation is Truncation.HEAD

    def test_a_loaded_template_renders_the_same_prompt(self):
        template = recommendation(version="2")
        loaded = PromptTemplate.from_yaml(template.to_yaml())
        assert loaded.render(CONTEXT, budget(30)) == template.render(
            CONTEXT, budget(30)
        )

    def test_json_fits_on_one_line(self, template):
        assert "\n" not in template.to_json()

    def test_yaml_writes_a_multiline_template_as_a_block(self):
        assert "template_string: |" in recommendation().to_yaml()

    def test_reads_hand_written_yaml(self):
        template = PromptTemplate.from_yaml(
            "name: greeting\n"
            "version: '1'\n"
            "template_string: |\n"
            "  Hello {{ name }}.\n"
            "variables:\n"
            "  name:\n"
            "    priority: critical\n"
            "    max_tokens: 10\n"
            "overflow_strategy: error\n"
        )
        assert template.render({"name": "Ann"}, budget()).text == "Hello Ann."
        assert template.variables["name"] == TemplateVariable(Priority.CRITICAL, 10)
        assert template.overflow_strategy is OverflowStrategy.ERROR

    def test_optional_fields_take_their_defaults(self):
        template = PromptTemplate.from_dict(
            {"name": "greeting", "template_string": "Hi"}
        )
        assert (template.version, template.variables) == (None, {})
        assert template.overflow_strategy is OverflowStrategy.TRUNCATE_LONGEST

    def test_treats_empty_yaml_variables_as_none_given(self):
        template = PromptTemplate.from_yaml(
            "name: greeting\ntemplate_string: Hi\nvariables:\n"
        )
        assert template.variables == {}

    def test_rejects_config_missing_a_required_field(self):
        with pytest.raises(ValueError, match="missing template_string"):
            PromptTemplate.from_dict({"name": "greeting"})

    def test_rejects_config_with_an_unknown_field(self):
        with pytest.raises(ValueError, match="unknown fields 'templat'"):
            PromptTemplate.from_dict(
                {"name": "a", "template_string": "Hi", "templat": "x"}
            )

    def test_rejects_config_that_is_not_a_mapping(self):
        with pytest.raises(TypeError, match="mapping of its fields, got list"):
            PromptTemplate.from_json('["greeting", "Hi"]')

    def test_an_unquoted_yaml_version_is_told_to_quote_it(self):
        with pytest.raises(ValueError, match="Quote it in YAML"):
            PromptTemplate.from_yaml("name: a\nversion: 2.1\ntemplate_string: Hi\n")

    def test_yaml_cannot_construct_python_objects(self):
        with pytest.raises(yaml.YAMLError):
            PromptTemplate.from_yaml(
                "name: !!python/object/apply:builtins.print ['should not run']\n"
                "template_string: Hi\n"
            )
