import copy
from typing import Any

import numpy as np
import pytest

from feast.context import (
    compress_count,
    compress_sample,
    compress_sequence,
    compress_summarize,
)

WATCH_EVENTS = [
    {"title": "Breaking Bad", "season": 1, "episode": 1, "ts": "2024-01-01"},
    {"title": "Breaking Bad", "season": 1, "episode": 2, "ts": "2024-01-01"},
    {"title": "Breaking Bad", "season": 1, "episode": 3, "ts": "2024-01-02"},
    *(
        {"title": "Breaking Bad", "season": 1, "episode": episode, "ts": "2024-01-02"}
        for episode in range(4, 9)
    ),
    {"title": "The Office", "season": 3, "episode": 7, "ts": "2024-01-03"},
]


def titles(*names: str) -> list[dict[str, Any]]:
    """One event per title, for histories where only the grouping matters."""
    return [{"title": name} for name in names]


class TestCompressSequence:
    def test_compresses_a_watch_history(self):
        compressed = compress_sequence(
            events=WATCH_EVENTS,
            group_by="title",
            compress_fn="count",
            template="Watched {count} episodes of {title} (Season {season})",
            max_groups=5,
        )
        assert compressed == [
            "Watched 8 episodes of Breaking Bad (Season 1)",
            "Watched 1 episodes of The Office (Season 3)",
        ]

    def test_only_groups_neighbours(self):
        compressed = compress_sequence(
            events=titles("Dark", "Dark", "Lost", "Dark"),
            group_by="title",
            template="{title} x{count}",
        )
        assert compressed == ["Dark x2", "Lost x1", "Dark x1"]

    def test_groups_by_several_fields(self):
        events = [
            {"title": "Dark", "season": 1},
            {"title": "Dark", "season": 1},
            {"title": "Dark", "season": 2},
        ]
        compressed = compress_sequence(
            events=events,
            group_by=["title", "season"],
            template="{title} S{season} x{count}",
        )
        assert compressed == ["Dark S1 x2", "Dark S2 x1"]

    def test_groups_by_a_function(self):
        compressed = compress_sequence(
            events=titles("Dark", "DARK", "Lost"),
            group_by=lambda event: event["title"].lower(),
            template="{title} x{count}",
        )
        assert compressed == ["DARK x2", "Lost x1"]

    def test_without_group_by_groups_equal_events(self):
        compressed = compress_sequence(
            events=[{"page": "pricing"}] * 5 + [{"page": "docs"}],
            template="Viewed {page} ({count} times)",
        )
        assert compressed == ["Viewed pricing (5 times)", "Viewed docs (1 times)"]

    def test_writes_a_record_as_its_fields_without_a_template(self):
        compressed = compress_sequence(events=WATCH_EVENTS[-1:], group_by="title")
        assert compressed == [
            "title: The Office, season: 3, episode: 7, ts: 2024-01-03, count: 1"
        ]

    def test_the_template_can_format_values_and_escape_braces(self):
        compressed = compress_sequence(
            events=WATCH_EVENTS[:8],
            group_by="title",
            template="{{{title}}}: {count:02d}",
        )
        assert compressed == ["{Breaking Bad}: 08"]

    def test_compresses_nothing_from_nothing(self):
        assert compress_sequence(events=[]) == []

    def test_accepts_events_in_any_iterable(self):
        # A pandas feature view can hold a history as a numpy object array.
        compressed = compress_sequence(
            events=np.array(WATCH_EVENTS, dtype=object),
            group_by="title",
            template="{title} x{count}",
        )
        assert compressed == ["Breaking Bad x8", "The Office x1"]

    @pytest.mark.parametrize("compress_fn", ["count", "summarize", "sample"])
    def test_leaves_the_events_untouched(self, compress_fn):
        before = copy.deepcopy(WATCH_EVENTS)
        compress_sequence(
            events=WATCH_EVENTS, group_by="title", compress_fn=compress_fn
        )
        assert WATCH_EVENTS == before


class TestMaxGroups:
    def test_keeps_the_most_recent_groups(self):
        compressed = compress_sequence(
            events=titles("A", "A", "B", "C", "C", "C", "D"),
            group_by="title",
            template="{title} x{count}",
            max_groups=2,
        )
        assert compressed == ["C x3", "D x1"]

    def test_keeps_every_group_when_there_are_fewer(self):
        compressed = compress_sequence(
            events=titles("A", "B"),
            group_by="title",
            template="{title}",
            max_groups=5,
        )
        assert compressed == ["A", "B"]

    def test_does_not_compress_the_groups_it_leaves_out(self):
        compressed_groups = []

        def compress(group):
            compressed_groups.append(group[0]["title"])
            return group[0]["title"]

        compress_sequence(
            events=titles("A", "B", "C"),
            group_by="title",
            compress_fn=compress,
            max_groups=1,
        )
        assert compressed_groups == ["C"]

    @pytest.mark.parametrize("max_groups", [0, -1])
    def test_rejects_max_groups_below_one(self, max_groups):
        with pytest.raises(ValueError, match="max_groups must be at least 1"):
            compress_sequence(events=titles("A"), max_groups=max_groups)


class TestBuiltIns:
    def test_count_is_the_latest_event_with_the_group_size(self):
        assert compress_count(WATCH_EVENTS[:3]) == {
            "title": "Breaking Bad",
            "season": 1,
            "episode": 3,
            "ts": "2024-01-02",
            "count": 3,
        }

    def test_count_overwrites_a_count_field(self):
        assert compress_count([{"count": 99}]) == {"count": 1}

    def test_summarize_adds_how_every_field_ran(self):
        record = compress_summarize(WATCH_EVENTS[:8])
        assert record["count"] == 8
        assert record["episode"] == 8
        assert (record["first_episode"], record["last_episode"]) == (1, 8)
        assert (record["min_episode"], record["max_episode"]) == (1, 8)
        assert (record["first_ts"], record["last_ts"]) == ("2024-01-01", "2024-01-02")

    def test_summarize_leaves_out_missing_values(self):
        record = compress_summarize(
            [
                {"title": "A", "rating": None},
                {"title": "A", "rating": 4},
                {"title": "A", "rating": 2},
                {"title": "A"},
            ]
        )
        assert (record["min_rating"], record["max_rating"]) == (2, 4)
        assert (record["first_rating"], record["last_rating"]) == (None, None)

    def test_summarize_gives_none_for_values_that_cannot_be_ordered(self):
        record = compress_summarize([{"value": 1}, {"value": "one"}])
        assert (record["min_value"], record["max_value"]) == (None, None)

    def test_summarize_through_a_template(self):
        compressed = compress_sequence(
            events=WATCH_EVENTS,
            group_by="title",
            compress_fn="summarize",
            template=(
                "Watched episodes {min_episode}-{max_episode} of {title} "
                "({first_ts} to {last_ts})"
            ),
        )
        assert compressed == [
            "Watched episodes 1-8 of Breaking Bad (2024-01-01 to 2024-01-02)",
            "Watched episodes 7-7 of The Office (2024-01-03 to 2024-01-03)",
        ]

    def test_sample_keeps_the_first_and_latest_events(self):
        assert compress_sample(WATCH_EVENTS[:8]) == [
            {**WATCH_EVENTS[0], "count": 8},
            {**WATCH_EVENTS[7], "count": 8},
        ]

    def test_sample_keeps_a_lone_event_once(self):
        assert compress_sample(WATCH_EVENTS[-1:]) == [{**WATCH_EVENTS[-1], "count": 1}]

    def test_sample_through_a_template(self):
        compressed = compress_sequence(
            events=WATCH_EVENTS,
            group_by="title",
            compress_fn="sample",
            template="Watched S{season}E{episode} of {title}",
        )
        assert compressed == [
            "Watched S1E1 of Breaking Bad",
            "Watched S1E8 of Breaking Bad",
            "Watched S3E7 of The Office",
        ]

    @pytest.mark.parametrize(
        "name, function",
        [
            ("count", compress_count),
            ("summarize", compress_summarize),
            ("sample", compress_sample),
        ],
    )
    def test_a_name_stands_for_its_function(self, name, function):
        def compress(compress_fn):
            return compress_sequence(
                events=WATCH_EVENTS, group_by="title", compress_fn=compress_fn
            )

        assert compress(name) == compress(function)


class TestCustomCompressFn:
    def test_writes_the_records_it_returns_through_the_template(self):
        compressed = compress_sequence(
            events=WATCH_EVENTS,
            group_by="title",
            compress_fn=lambda group: {
                "title": group[0]["title"],
                "minutes": 45 * len(group),
            },
            template="{title}: {minutes} minutes",
        )
        assert compressed == ["Breaking Bad: 360 minutes", "The Office: 45 minutes"]

    def test_keeps_the_lines_it_returns_as_they_are(self):
        # Where a summarizing model would write the line itself.
        compressed = compress_sequence(
            events=WATCH_EVENTS,
            group_by="title",
            compress_fn=lambda group: f"A {len(group)}-episode stretch of drama",
            template="ignored for lines: {title}",
        )
        assert compressed == [
            "A 8-episode stretch of drama",
            "A 1-episode stretch of drama",
        ]

    def test_can_return_several_records_and_lines(self):
        compressed = compress_sequence(
            events=WATCH_EVENTS[:8],
            group_by="title",
            compress_fn=lambda group: ["Started a binge:", group[0], group[-1]],
            template="S{season}E{episode}",
        )
        assert compressed == ["Started a binge:", "S1E1", "S1E8"]

    def test_is_given_each_group_in_order(self):
        groups = []
        compress_sequence(
            events=WATCH_EVENTS,
            group_by="title",
            compress_fn=lambda group: groups.append(group) or "",
        )
        assert groups == [tuple(WATCH_EVENTS[:8]), tuple(WATCH_EVENTS[8:])]


class TestValidation:
    @pytest.mark.parametrize(
        "events, kind", [(WATCH_EVENTS[0], "mapping"), ("title", "string")]
    )
    def test_rejects_a_single_event_or_string(self, events, kind):
        with pytest.raises(TypeError, match=f"not a single {kind}"):
            compress_sequence(events=events)

    def test_rejects_an_event_that_is_not_a_mapping(self):
        with pytest.raises(TypeError, match="Events must be mappings, got str"):
            compress_sequence(events=[{"title": "A"}, "B"])  # type: ignore[list-item]

    def test_names_a_missing_group_by_field_but_no_values(self):
        with pytest.raises(ValueError, match="Cannot group by titel") as excinfo:
            compress_sequence(events=WATCH_EVENTS, group_by="titel")
        assert "title, season, episode, ts" in str(excinfo.value)
        assert "Breaking Bad" not in str(excinfo.value)

    @pytest.mark.parametrize("group_by", [5, [], ["title", 3]])
    def test_rejects_group_by_that_is_neither_fields_nor_a_function(self, group_by):
        with pytest.raises(TypeError, match="group_by must be a field name"):
            compress_sequence(events=WATCH_EVENTS, group_by=group_by)

    def test_rejects_an_unknown_compress_fn_name(self):
        with pytest.raises(ValueError, match="'count', 'summarize', 'sample'"):
            compress_sequence(events=WATCH_EVENTS, compress_fn="mean")

    def test_rejects_a_compress_fn_that_is_not_a_function(self):
        with pytest.raises(TypeError, match="compress_fn must name a built-in"):
            compress_sequence(events=WATCH_EVENTS, compress_fn=42)  # type: ignore[arg-type]

    @pytest.mark.parametrize("result", [8, [{"title": "A"}, 8], None])
    def test_rejects_a_result_that_is_not_records_or_lines(self, result):
        with pytest.raises(TypeError, match="must return a string, a mapping"):
            compress_sequence(events=WATCH_EVENTS, compress_fn=lambda group: result)

    def test_rejects_a_malformed_template(self):
        with pytest.raises(ValueError, match="not a valid format string"):
            compress_sequence(events=WATCH_EVENTS, template="Watched {title")

    @pytest.mark.parametrize("template", ["{}", "{0}", "{title.upper}", "{title[0]}"])
    def test_rejects_template_fields_that_are_not_plain_names(self, template):
        with pytest.raises(ValueError, match="must be plain names"):
            compress_sequence(events=WATCH_EVENTS, template=template)

    def test_rejects_a_template_that_is_not_a_string(self):
        with pytest.raises(TypeError, match="template must be a string"):
            compress_sequence(events=WATCH_EVENTS, template=5)  # type: ignore[arg-type]

    def test_names_a_template_field_a_record_lacks_but_no_values(self):
        with pytest.raises(ValueError, match=r"uses \{episodes\}") as excinfo:
            compress_sequence(
                events=WATCH_EVENTS, group_by="title", template="Watched {episodes}"
            )
        assert "title, season, episode, ts, count" in str(excinfo.value)
        assert "Breaking Bad" not in str(excinfo.value)

    def test_checks_its_arguments_before_any_event(self):
        with pytest.raises(ValueError, match="Unknown compress_fn"):
            compress_sequence(events=[], compress_fn="mean")
