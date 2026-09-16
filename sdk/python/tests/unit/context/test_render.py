import builtins
import dataclasses
import os

import pytest

from feast.context import (
    DEFAULT_TRUNCATION_MARKER,
    OverflowStrategy,
    Priority,
    TemplateVariable,
    TokenBudget,
    Truncation,
    compress_sequence,
    priority_select,
    render_with_budget,
    truncate,
)
from feast.context import render as render_module
from feast.context.errors import (
    ContextError,
    TemplateRenderError,
    TokenBudgetExceededError,
)
from feast.context.tokenizer import ApproximateTokenizer, Tokenizer, TokenizerName
from feast.errors import FeastExtrasDependencyImportError

# Four characters per token: "abcd" is one token under the estimate, and the
# default "..." marker is one token too.
ESTIMATING = ApproximateTokenizer()


class SeamTokenizer(Tokenizer):
    """A token a character, and three more wherever text runs into a marker.

    Stands in for tokenizers that merge text differently across a cut, which
    makes the pieces cost more joined than apart.
    """

    @property
    def name(self) -> str:
        return "seam"

    def count_tokens(self, text: str) -> int:
        return len(text) + (3 if "x..." in text else 0)


def budget(max_tokens: int = 100, consumed: int = 0) -> TokenBudget:
    """A budget on the deterministic estimate, so counts need no tiktoken."""
    return TokenBudget(max_tokens=max_tokens, tokenizer=ESTIMATING, consumed=consumed)


def text(tokens: int, letter: str = "a") -> str:
    """Content costing exactly ``tokens`` tokens under the estimate."""
    return letter * (tokens * 4)


def cut(tokens: int, letter: str = "a") -> str:
    """What :func:`text` becomes when its tail is cut to ``tokens`` tokens."""
    return text(tokens - 1, letter) + DEFAULT_TRUNCATION_MARKER


@pytest.fixture
def no_jinja2(monkeypatch):
    """Make ``import jinja2`` fail, as it does when Jinja2 is not installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jinja2" or name.startswith("jinja2."):
            raise ImportError("No module named 'jinja2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    render_module._compile_source.cache_clear()
    render_module._inline_environment.cache_clear()
    yield
    render_module._compile_source.cache_clear()
    render_module._inline_environment.cache_clear()


class TestTruncate:
    def test_leaves_text_that_fits_unchanged(self):
        assert truncate(text(5), budget(5)) == text(5)

    def test_cuts_the_tail_by_default(self):
        result = truncate(text(5, "s") + text(5, "e"), budget(6))
        assert result == text(5, "s") + DEFAULT_TRUNCATION_MARKER

    def test_cuts_the_head_to_keep_the_end(self):
        result = truncate(
            text(5, "s") + text(5, "e"), budget(6), truncation=Truncation.HEAD
        )
        assert result == DEFAULT_TRUNCATION_MARKER + text(5, "e")

    def test_cuts_the_middle_to_keep_both_ends(self):
        result = truncate(
            text(4, "s") + text(4, "m") + text(4, "e"),
            budget(5),
            truncation=Truncation.MIDDLE,
        )
        assert result == text(2, "s") + DEFAULT_TRUNCATION_MARKER + text(2, "e")

    def test_a_middle_cut_gives_the_odd_token_to_the_start(self):
        result = truncate(
            text(4, "s") + text(4, "e"), budget(4), truncation=Truncation.MIDDLE
        )
        assert result == text(2, "s") + DEFAULT_TRUNCATION_MARKER + text(1, "e")

    def test_accepts_a_truncation_by_name(self):
        result = truncate(text(5, "s") + text(5, "e"), budget(6), truncation="head")  # type: ignore[arg-type]
        assert result.endswith(text(5, "e"))

    def test_rejects_an_unknown_truncation(self):
        with pytest.raises(ValueError, match="sideways"):
            truncate(text(10), budget(5), truncation="sideways")  # type: ignore[arg-type]

    @pytest.mark.parametrize("truncation", list(Truncation))
    @pytest.mark.parametrize("max_tokens", range(0, 12))
    def test_the_result_always_fits(self, truncation, max_tokens):
        result = truncate(text(12), budget(max_tokens), truncation=truncation)
        assert ESTIMATING.count_tokens(result) <= max_tokens

    def test_cuts_to_what_remains_of_a_used_budget(self):
        assert truncate(text(10), budget(10, consumed=4)) == text(5) + "..."

    def test_a_spent_budget_leaves_nothing(self):
        assert truncate(text(10), budget(10, consumed=10)) == ""

    def test_marks_the_cut_with_a_custom_marker(self):
        # " [...] " is seven characters: two tokens, leaving four for the text.
        assert truncate(text(10), budget(6), marker=" [...] ") == text(4) + " [...] "

    def test_an_empty_marker_leaves_no_trace(self):
        assert truncate(text(10), budget(6), marker="") == text(6)

    def test_drops_the_marker_when_no_text_would_fit_beside_it(self):
        assert truncate(text(10), budget(1)) == text(1)

    def test_cuts_again_when_the_pieces_cost_more_joined(self):
        seam = TokenBudget(max_tokens=10, tokenizer=SeamTokenizer())
        # Seven characters and a marker fill the budget apart, but joined they
        # pay three tokens more; the text gives those up.
        assert truncate("x" * 20, seam) == "xxxx..."

    def test_rejects_non_string_text(self):
        with pytest.raises(TypeError, match="text must be a string"):
            truncate(None, budget())  # type: ignore[arg-type]

    def test_rejects_a_non_string_marker(self):
        with pytest.raises(TypeError, match="marker must be a string"):
            truncate(text(10), budget(5), marker=None)  # type: ignore[arg-type]


class TestRenderWithBudget:
    def test_renders_the_template_with_the_context(self):
        rendering = render_with_budget("Hello {{ name }}", {"name": "Ann"}, budget())
        assert rendering.text == "Hello Ann"
        assert rendering.truncated is False
        assert rendering.truncated_variables == ()

    def test_charges_the_budget_for_the_prompt(self):
        rendering = render_with_budget("Hello {{ name }}", {"name": "Ann"}, budget())
        assert rendering.budget.consumed == ESTIMATING.count_tokens("Hello Ann")

    def test_spends_only_what_is_left_of_a_used_budget(self):
        rendering = render_with_budget(
            "{{ notes }}", {"notes": text(10)}, budget(10, consumed=4)
        )
        assert rendering.text == cut(6)
        assert rendering.budget.remaining == 0

    def test_leaves_the_original_budget_untouched(self):
        original = budget(10)
        render_with_budget("{{ notes }}", {"notes": text(20)}, original)
        assert original.consumed == 0

    def test_renders_the_same_prompt_every_time(self):
        def render():
            return render_with_budget(
                "{{ a }}|{{ b }}", {"a": text(10), "b": text(6)}, budget(9)
            )

        assert render() == render()


class TestTemplates:
    def test_renders_loops_conditionals_and_filters(self):
        rendering = render_with_budget(
            "{% for item in candidates %}{{ loop.index }}. {{ item.title | upper }}\n"
            "{% endfor %}{% if user.premium %}Premium{% else %}Free{% endif %}",
            {
                "candidates": [{"title": "dark"}, {"title": "lost"}],
                "user": {"premium": True},
            },
            budget(),
        )
        assert rendering.text == "1. DARK\n2. LOST\nPremium"

    def test_an_undefined_variable_raises(self):
        with pytest.raises(TemplateRenderError, match="'question' is undefined"):
            render_with_budget("Q: {{ question }}", {}, budget())

    def test_an_optional_variable_can_take_a_default(self):
        rendering = render_with_budget("{{ tier | default('free') }}", {}, budget())
        assert rendering.text == "free"

    def test_a_syntax_error_raises_with_its_line(self):
        with pytest.raises(TemplateRenderError, match="at line 2"):
            render_with_budget("Hello\n{{ name }", {"name": "Ann"}, budget())

    def test_template_errors_are_context_errors(self):
        assert issubclass(TemplateRenderError, ContextError)

    def test_does_not_escape_html(self):
        rendering = render_with_budget("{{ q }}", {"q": "<b>&</b>"}, budget())
        assert rendering.text == "<b>&</b>"

    def test_compiles_a_source_once(self):
        render_module._compile_source.cache_clear()
        for name in ["Ann", "Bo"]:
            render_with_budget("Hello {{ name }}", {"name": name}, budget())
        info = render_module._compile_source.cache_info()
        assert (info.misses, info.hits) == (1, 1)

    def test_rejects_a_template_that_is_neither_source_nor_a_path(self):
        with pytest.raises(TypeError, match="Jinja2 source or a path"):
            render_with_budget(42, {}, budget())  # type: ignore[arg-type]


class TestTemplateFiles:
    @pytest.fixture
    def templates(self, tmp_path, monkeypatch):
        (tmp_path / "templates" / "partials").mkdir(parents=True)
        (tmp_path / "templates" / "greeting.jinja2").write_text("Hello {{ name }}")
        (tmp_path / "templates" / "page.jinja2").write_text(
            'Hi {{ name }}. {% include "partials/history.jinja2" %}'
        )
        (tmp_path / "templates" / "partials" / "history.jinja2").write_text(
            "Watched {{ history }}"
        )
        monkeypatch.chdir(tmp_path)
        return tmp_path / "templates"

    def test_reads_a_file_named_by_a_relative_path(self, templates):
        rendering = render_with_budget(
            "templates/greeting.jinja2", {"name": "Ann"}, budget()
        )
        assert rendering.text == "Hello Ann"

    def test_reads_a_file_given_as_a_path(self, templates):
        rendering = render_with_budget(
            templates / "greeting.jinja2", {"name": "Ann"}, budget()
        )
        assert rendering.text == "Hello Ann"

    def test_a_file_can_include_the_templates_beside_it(self, templates):
        rendering = render_with_budget(
            "templates/page.jinja2", {"name": "Ann", "history": "Dark"}, budget()
        )
        assert rendering.text == "Hi Ann. Watched Dark"

    def test_picks_up_a_file_that_changed(self, templates):
        path = templates / "greeting.jinja2"
        assert render_with_budget(path, {"name": "Ann"}, budget()).text == "Hello Ann"
        path.write_text("Goodbye {{ name }}")
        later = os.stat(path).st_mtime + 10
        os.utime(path, (later, later))
        assert render_with_budget(path, {"name": "Ann"}, budget()).text == "Goodbye Ann"

    @pytest.mark.parametrize("name", ["templates/missing.jinja2", "notes.txt"])
    def test_a_missing_file_raises_rather_than_rendering_its_name(
        self, templates, name
    ):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            render_with_budget(name, {}, budget())

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("Hello", "Hello"),
            ("Read {{ name }}.txt", "Read Ann.txt"),
            ("Line one\nline two.txt", "Line one\nline two.txt"),
        ],
    )
    def test_any_other_string_is_the_template_itself(self, templates, source, expected):
        assert render_with_budget(source, {"name": "Ann"}, budget()).text == expected


class TestOverflowStrategies:
    def test_truncate_longest_cuts_the_longest_variable_first(self):
        rendering = render_with_budget(
            "{{ a }}|{{ b }}",
            {"a": text(10, "a"), "b": text(4, "b")},
            budget(12),
            overflow_strategy=OverflowStrategy.TRUNCATE_LONGEST,
        )
        assert rendering.text == cut(7, "a") + "|" + text(4, "b")
        assert rendering.truncated_variables == ("a",)

    def test_truncate_longest_cuts_down_to_the_next_longest(self):
        rendering = render_with_budget(
            "{{ a }}{{ b }}", {"a": text(10, "a"), "b": text(8, "b")}, budget(12)
        )
        assert rendering.text == cut(6, "a") + cut(6, "b")
        assert rendering.truncated_variables == ("a", "b")

    def test_truncate_lowest_priority_empties_the_least_important_first(self):
        rendering = render_with_budget(
            "{{ low }}{{ high }}",
            {"low": text(4, "l"), "high": text(10, "h")},
            budget(8),
            overflow_strategy="truncate_lowest_priority",  # type: ignore[arg-type]
            variables={"low": {"priority": "low"}, "high": {"priority": "high"}},
        )
        assert rendering.text == cut(8, "h")
        assert rendering.truncated_variables == ("low", "high")

    def test_truncate_longest_ignores_priorities_below_critical(self):
        rendering = render_with_budget(
            "{{ low }}{{ high }}",
            {"low": text(4, "l"), "high": text(10, "h")},
            budget(8),
            variables={"low": {"priority": "low"}, "high": {"priority": "high"}},
        )
        assert rendering.text == text(4, "l") + cut(4, "h")

    def test_truncate_lowest_priority_cuts_the_longest_among_equals(self):
        rendering = render_with_budget(
            "{{ a }}{{ b }}{{ c }}",
            {"a": text(10, "a"), "b": text(4, "b"), "c": text(4, "c")},
            budget(15),
            overflow_strategy=OverflowStrategy.TRUNCATE_LOWEST_PRIORITY,
            variables={
                "a": {"priority": "low"},
                "b": {"priority": "low"},
                "c": {"priority": "high"},
            },
        )
        assert rendering.text == cut(7, "a") + text(4, "b") + text(4, "c")

    def test_error_refuses_to_cut(self):
        with pytest.raises(
            TokenBudgetExceededError, match="overflow_strategy is 'error'"
        ):
            render_with_budget(
                "{{ a }}",
                {"a": text(10)},
                budget(5),
                overflow_strategy=OverflowStrategy.ERROR,
            )

    def test_error_renders_a_prompt_that_fits(self):
        rendering = render_with_budget(
            "{{ a }}",
            {"a": text(5)},
            budget(5),
            overflow_strategy=OverflowStrategy.ERROR,
        )
        assert rendering.text == text(5)

    def test_rejects_an_unknown_strategy(self):
        with pytest.raises(ValueError, match="truncate_randomly"):
            render_with_budget(
                "{{ a }}",
                {"a": "x"},
                budget(),
                overflow_strategy="truncate_randomly",  # type: ignore[arg-type]
            )


class TestWhatGivesWay:
    def test_never_cuts_a_critical_variable(self):
        rendering = render_with_budget(
            "{{ rules }}{{ notes }}",
            {"rules": text(10, "r"), "notes": text(4, "n")},
            budget(12),
            variables={"rules": {"priority": Priority.CRITICAL}},
        )
        assert rendering.text == text(10, "r") + cut(2, "n")

    def test_fails_when_what_may_not_be_cut_runs_over(self):
        with pytest.raises(TokenBudgetExceededError, match="every variable that may"):
            render_with_budget(
                "{{ rules }}{{ notes }}",
                {"rules": text(10, "r"), "notes": text(4, "n")},
                budget(9),
                variables={"rules": {"priority": "critical"}},
            )

    def test_keeps_the_template_text_whole(self):
        rendering = render_with_budget(
            "Answer briefly. {{ notes }} Thanks.", {"notes": text(40)}, budget(12)
        )
        assert rendering.text.startswith("Answer briefly. ")
        assert rendering.text.endswith(" Thanks.")
        assert ESTIMATING.count_tokens(rendering.text) <= 12

    def test_never_cuts_values_that_are_not_text(self):
        rendering = render_with_budget(
            "{% for title in titles %}{{ title }};{% endfor %}{{ notes }}",
            {"titles": ["Dark", "Lost"], "notes": text(10, "n")},
            budget(10),
        )
        assert rendering.text == "Dark;Lost;" + cut(7, "n")

    def test_counts_a_variable_written_twice_twice(self):
        rendering = render_with_budget("{{ a }}|{{ a }}", {"a": text(10)}, budget(15))
        # Each token cut saves two, so seven of the ten tokens can stay.
        assert rendering.text == cut(7) + "|" + cut(7)

    def test_leaves_a_variable_in_an_untaken_branch_whole(self):
        rendering = render_with_budget(
            "{% if show_bio %}{{ bio }}{% endif %}{{ notes }}",
            {"show_bio": False, "bio": text(20, "b"), "notes": text(10, "n")},
            budget(8),
        )
        assert rendering.text == cut(8, "n")
        assert rendering.truncated_variables == ("notes",)

    def test_leaves_a_value_the_template_only_measures_whole(self):
        rendering = render_with_budget(
            "{{ items | length }} items. {{ notes }}",
            {"items": text(20, "i"), "notes": text(10, "n")},
            budget(10),
        )
        assert rendering.text == "80 items. " + cut(7, "n")

    def test_leaves_variables_the_template_never_mentions(self):
        rendering = render_with_budget(
            "{{ notes }}", {"notes": text(10), "unused": text(50)}, budget(8)
        )
        assert rendering.truncated_variables == ("notes",)

    def test_cuts_where_a_variable_says(self):
        rendering = render_with_budget(
            "{{ history }}",
            {"history": text(5, "s") + text(5, "e")},
            budget(6),
            variables={"history": {"truncation": "head"}},
        )
        assert rendering.text == DEFAULT_TRUNCATION_MARKER + text(5, "e")

    def test_leaves_the_marker_given(self):
        rendering = render_with_budget("{{ a }}", {"a": text(10)}, budget(6), marker="")
        assert rendering.text == text(6)

    def test_rejects_a_non_string_marker(self):
        with pytest.raises(TypeError, match="marker must be a string"):
            render_with_budget("{{ a }}", {"a": "x"}, budget(), marker=None)  # type: ignore[arg-type]


class TestMaxTokens:
    def test_caps_a_variable_even_when_the_prompt_fits(self):
        rendering = render_with_budget(
            "{{ bio }}",
            {"bio": text(10)},
            budget(),
            variables={"bio": {"max_tokens": 3}},
        )
        assert rendering.text == cut(3)
        assert rendering.truncated_variables == ("bio",)

    def test_caps_a_critical_variable_too(self):
        rendering = render_with_budget(
            "{{ name }}",
            {"name": text(10)},
            budget(),
            variables={"name": {"priority": "critical", "max_tokens": 3}},
        )
        assert rendering.text == cut(3)

    def test_leaves_a_variable_under_its_cap_whole(self):
        rendering = render_with_budget(
            "{{ bio }}",
            {"bio": text(3)},
            budget(),
            variables={"bio": {"max_tokens": 3}},
        )
        assert rendering.truncated is False

    def test_caps_before_the_strategy_cuts_the_rest(self):
        rendering = render_with_budget(
            "{{ a }}{{ b }}",
            {"a": text(10, "a"), "b": text(10, "b")},
            budget(12),
            variables={"a": {"max_tokens": 4}},
        )
        assert rendering.text == cut(4, "a") + cut(8, "b")

    def test_rejects_a_cap_on_a_value_that_is_not_text(self):
        with pytest.raises(TypeError, match="caps text"):
            render_with_budget(
                "{{ items }}",
                {"items": [1, 2]},
                budget(),
                variables={"items": {"max_tokens": 2}},
            )


class TestLazyVariables:
    def test_renders_what_a_lazy_variable_returns(self):
        rendering = render_with_budget(
            "Watched {{ history }}", {"history": lambda: "Dark"}, budget()
        )
        assert rendering.text == "Watched Dark"

    def test_calls_it_once_even_when_the_prompt_is_cut(self):
        calls = []

        def history():
            calls.append("history")
            return text(20)

        rendering = render_with_budget("{{ history }}", {"history": history}, budget(8))
        assert rendering.text == cut(8)
        assert calls == ["history"]

    def test_never_calls_one_the_template_does_not_mention(self):
        calls = []
        render_with_budget(
            "{{ name }}", {"name": "Ann", "unused": lambda: calls.append(1)}, budget()
        )
        assert calls == []

    def test_calls_one_an_included_template_mentions(self, tmp_path):
        (tmp_path / "page.jinja2").write_text('{% include "history.jinja2" %}')
        (tmp_path / "history.jinja2").write_text("Watched {{ history }}")
        rendering = render_with_budget(
            tmp_path / "page.jinja2", {"history": lambda: "Dark"}, budget()
        )
        assert rendering.text == "Watched Dark"


class TestTemplateVariable:
    def test_defaults_to_medium_priority_no_cap_and_cutting_the_tail(self):
        assert TemplateVariable() == TemplateVariable(
            Priority.MEDIUM, None, Truncation.TAIL
        )

    def test_of_builds_from_a_mapping_of_settings(self):
        assert TemplateVariable.of(
            {"priority": "critical", "max_tokens": 50}
        ) == TemplateVariable(Priority.CRITICAL, 50)

    def test_of_passes_a_variable_through(self):
        variable = TemplateVariable(max_tokens=5)
        assert TemplateVariable.of(variable) is variable

    def test_round_trips_through_a_dict(self):
        variable = TemplateVariable(Priority.HIGH, 2000, Truncation.HEAD)
        assert variable.to_dict() == {
            "priority": "high",
            "max_tokens": 2000,
            "truncation": "head",
        }
        assert TemplateVariable.of(variable.to_dict()) == variable

    def test_rejects_an_unknown_setting(self):
        with pytest.raises(ValueError, match="'priorty'"):
            TemplateVariable.of({"priorty": "high"})

    @pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True, "50"])
    def test_rejects_max_tokens_that_is_not_a_positive_integer(self, max_tokens):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            TemplateVariable(max_tokens=max_tokens)

    def test_rejects_an_unknown_priority(self):
        with pytest.raises(ValueError, match="urgent"):
            TemplateVariable.of({"priority": "urgent"})

    def test_rejects_settings_that_are_neither(self):
        with pytest.raises(TypeError, match="mapping of its settings"):
            TemplateVariable.of("high")  # type: ignore[arg-type]


class TestRendering:
    @pytest.fixture
    def rendering(self):
        return render_with_budget("Hello {{ name }}", {"name": "Ann"}, budget())

    def test_reads_as_its_text(self, rendering):
        assert f"{rendering}" == "Hello Ann"

    def test_is_immutable(self, rendering):
        with pytest.raises(dataclasses.FrozenInstanceError):
            rendering.text = "other"


class TestJinja2IsOptional:
    def test_rendering_asks_for_the_llm_extra(self, no_jinja2):
        with pytest.raises(FeastExtrasDependencyImportError, match=r"feast\[llm\]"):
            render_with_budget("Hello {{ name }}", {"name": "Ann"}, budget())

    def test_the_rest_of_the_module_works_without_it(self, no_jinja2):
        assert budget(10).consume("abcd").remaining == 9
        assert priority_select([("high", "abcd")], budget(10)).render() == "abcd"
        assert compress_sequence(events=[{"a": 1}] * 2, template="{count}") == ["2"]
        assert truncate(text(10), budget(6)) == cut(6)


class TestTiktoken:
    # Characters both encodings spell over several tokens, so cuts split some.
    TEXT = (
        "You are a recommendation assistant. The user watched 北京の天気 and 龘 "
        "twice 👩‍👩‍👧‍👦, then <|endoftext|> in Ελληνικά 🦜. "
    ) * 8 + "Question: what should they watch next?"
    MARKER = " <<cut>> "

    @pytest.mark.parametrize("truncation", list(Truncation))
    @pytest.mark.parametrize(
        "encoding", [TokenizerName.CL100K_BASE.value, TokenizerName.O200K_BASE.value]
    )
    def test_truncate_fits_an_exact_budget(self, encoding, truncation):
        pytest.importorskip("tiktoken")
        for max_tokens in range(0, 80, 3):
            exact = TokenBudget.of(max_tokens, encoding, fallback=False)
            result = truncate(
                self.TEXT, exact, truncation=truncation, marker=self.MARKER
            )
            assert exact.count_tokens(result) <= max_tokens
            # What survives is cut from the text as it was, split nowhere else.
            if truncation is Truncation.TAIL:
                assert self.TEXT.startswith(result.removesuffix(self.MARKER))
            elif truncation is Truncation.HEAD:
                assert self.TEXT.endswith(result.removeprefix(self.MARKER))
            elif self.MARKER in result:
                start, end = result.split(self.MARKER)
                assert self.TEXT.startswith(start) and self.TEXT.endswith(end)

    @pytest.mark.parametrize(
        "strategy", ["truncate_longest", "truncate_lowest_priority"]
    )
    @pytest.mark.parametrize(
        "encoding", [TokenizerName.CL100K_BASE.value, TokenizerName.O200K_BASE.value]
    )
    def test_the_prompt_fits_an_exact_budget(self, encoding, strategy):
        pytest.importorskip("tiktoken")
        template = (
            "You are a recommendation engine for {{ user_name }}.\n\n"
            "Recent activity:\n{{ watch_history }}\n\n"
            "Please rank these candidates:\n{{ candidates }}"
        )
        context = {
            "user_name": "Ann",
            "watch_history": self.TEXT,
            "candidates": "\n".join(f"- 候補 {n}: a title 🎬" for n in range(40)),
        }
        variables = {
            "user_name": {"priority": "critical"},
            "watch_history": {"priority": "low", "truncation": "head"},
            "candidates": {"priority": "high"},
        }
        for max_tokens in [60, 150, 400]:
            rendering = render_with_budget(
                template,
                context,
                TokenBudget.of(max_tokens, encoding, fallback=False),
                overflow_strategy=strategy,  # type: ignore[arg-type]
                variables=variables,
            )
            assert rendering.budget.count_tokens(rendering.text) <= max_tokens
            assert rendering.text.startswith("You are a recommendation engine for Ann.")
