"""Compressing runs of similar events into a few lines each.

Activity histories repeat themselves: eight episodes of the same show, one page
viewed five times over. Listed in full, a run spends tokens on what a model
reads as a single fact. :func:`compress_sequence` groups consecutive events,
compresses each group, and writes what is left through a template::

    watch_events = [
        {"title": "Breaking Bad", "season": 1, "episode": 1, "ts": "2024-01-01"},
        {"title": "Breaking Bad", "season": 1, "episode": 2, "ts": "2024-01-01"},
        {"title": "Breaking Bad", "season": 1, "episode": 3, "ts": "2024-01-02"},
        {"title": "The Office", "season": 3, "episode": 7, "ts": "2024-01-03"},
        {"title": "The Office", "season": 3, "episode": 8, "ts": "2024-01-03"},
    ]
    compress_sequence(
        events=watch_events,
        group_by="title",
        compress_fn="count",
        template="Watched {count} episodes of {title} (Season {season})",
        max_groups=5,
    )
    # ["Watched 3 episodes of Breaking Bad (Season 1)",
    #  "Watched 2 episodes of The Office (Season 3)"]

Only neighbouring events are grouped, so the history keeps the order things
happened in, and the events alone decide the result: an OnDemandFeatureView
compresses the same history the same way offline and online.
"""

import string
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar, Union

#: An event: a mapping of fields. Bound rather than fixed, so functions typed
#: for the caller's own events, such as ``dict[str, Any]``, are accepted.
EventT = TypeVar("EventT", bound=Mapping[str, Any])

#: The names a template can use, standing for one line.
Record = Mapping[str, Any]

#: What compressing a group yields: records for the template to write, lines
#: already written, or a list of either.
Compressed = Union[str, Record, Sequence[Union[str, Record]]]

#: Compresses a group, given its events in order.
CompressFn = Callable[[Sequence[EventT]], Compressed]

#: What consecutive events must share to be grouped: a field, several fields,
#: or whatever a function computes from the event.
GroupBySpec = Union[str, Sequence[str], Callable[[EventT], Any]]


def compress_sequence(
    events: Iterable[EventT],
    *,
    group_by: Optional[GroupBySpec[EventT]] = None,
    compress_fn: Union[str, CompressFn[EventT]] = "count",
    template: Optional[str] = None,
    max_groups: Optional[int] = None,
) -> list[str]:
    """Group consecutive events, compress each group, and write it as lines.

    Consecutive events with the same ``group_by`` value form a group.
    ``compress_fn`` turns each group into records, and ``template`` writes
    each record as a line.

    ``group_by`` is a field name, a list of them, or a function of an event;
    naming a field an event lacks is an error. Without it, consecutive events
    are grouped when they are equal.

    ``compress_fn`` names a built-in, or is a function given a group's events
    that returns records, finished lines of text, or a list of either:

    - ``"count"``, :func:`compress_count`: the group's latest event, with
      ``count``, the number of events in the group.
    - ``"summarize"``, :func:`compress_summarize`: the count record, plus the
      first, last, smallest and largest value of every field.
    - ``"sample"``, :func:`compress_sample`: the group's first and latest
      events, each with ``count``.

    ``template`` is a :meth:`str.format` string of plain field names, such as
    ``"Watched {count} episodes of {title}"``. Without it, a record is written
    as its fields: ``"title: The Office, season: 3, count: 2"``.

    ``max_groups`` keeps only the most recent groups, the last ones in a history
    listed oldest first. Groups left out are never compressed.

    Args:
        events: The history, oldest event first.
        group_by: What consecutive events must share to be grouped.
        compress_fn: How each group is compressed.
        template: How each record is written.
        max_groups: The most groups to keep, counting back from the latest.

    Returns:
        The lines written, in the order of the history.

    Raises:
        TypeError: ``events`` is a single mapping, an event is not a mapping,
            ``group_by`` or ``compress_fn`` is of the wrong kind, or
            ``compress_fn`` returns something other than records or lines.
        ValueError: An event lacks a ``group_by`` field, ``compress_fn`` names
            no built-in, ``template`` is malformed or names a field a record
            lacks, or ``max_groups`` is less than one.
    """
    if isinstance(events, (str, Mapping)):
        kind = "string" if isinstance(events, str) else "mapping"
        raise TypeError(f"events must be an iterable of mappings, not a single {kind}.")
    group_key = _group_key(group_by)
    compress = _compress_function(compress_fn)
    if template is not None:
        _check_template(template)
    if max_groups is not None and max_groups < 1:
        raise ValueError(f"max_groups must be at least 1, got {max_groups}.")

    groups = _groups(events, group_key)
    if max_groups is not None:
        groups = groups[-max_groups:]
    return [
        _write(item, template) for group in groups for item in _items(compress(group))
    ]


def compress_count(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compress a group to its latest event, with ``count``, the group's size.

    The built-in behind ``compress_fn="count"``. A field of the event named
    ``count`` is overwritten.
    """
    return {**events[-1], "count": len(events)}


def compress_summarize(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compress a group to its :func:`compress_count` record and field ranges.

    The built-in behind ``compress_fn="summarize"``. For every field of the
    group, ``first_<field>`` and ``last_<field>`` are its values in the first
    and latest events, and ``min_<field>`` and ``max_<field>`` its smallest and
    largest, leaving out missing values. They are None when no event has a
    value, or when the values cannot be ordered, such as numbers mixed with
    text.
    """
    record = compress_count(events)
    for field in dict.fromkeys(field for event in events for field in event):
        present = [event[field] for event in events if event.get(field) is not None]
        record[f"first_{field}"] = events[0].get(field)
        record[f"last_{field}"] = events[-1].get(field)
        record[f"min_{field}"], record[f"max_{field}"] = _extremes(present)
    return record


def compress_sample(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compress a group to its first and latest events, each with ``count``.

    The built-in behind ``compress_fn="sample"``. A group of one event yields
    that event once.
    """
    ends = events[:1] if len(events) == 1 else [events[0], events[-1]]
    return [{**event, "count": len(events)} for event in ends]


_BUILT_IN_COMPRESS_FNS: dict[str, Callable[[Sequence[Mapping[str, Any]]], Any]] = {
    "count": compress_count,
    "summarize": compress_summarize,
    "sample": compress_sample,
}


def _group_key(group_by: Optional[GroupBySpec[Any]]) -> Callable[[Any], Any]:
    if group_by is None:
        return lambda event: event
    if callable(group_by):
        return group_by
    if isinstance(group_by, str):
        fields: tuple[str, ...] = (group_by,)
    elif (
        isinstance(group_by, Sequence)
        and group_by
        and all(isinstance(field, str) for field in group_by)
    ):
        fields = tuple(group_by)
    else:
        raise TypeError(
            f"group_by must be a field name, a list of field names, or a "
            f"function, got {group_by!r}."
        )

    def read_fields(event: Mapping[str, Any]) -> tuple[Any, ...]:
        missing = [field for field in fields if field not in event]
        if missing:
            # Field names only: the values are feature data.
            raise ValueError(
                f"Cannot group by {', '.join(missing)}: an event has no such "
                f"field. Its fields are: {', '.join(map(str, event))}."
            )
        return tuple(event[field] for field in fields)

    return read_fields


def _compress_function(compress_fn: Union[str, CompressFn[Any]]) -> Callable[..., Any]:
    if isinstance(compress_fn, str):
        if compress_fn not in _BUILT_IN_COMPRESS_FNS:
            raise ValueError(
                f"Unknown compress_fn {compress_fn!r}. Use one of "
                f"{', '.join(map(repr, _BUILT_IN_COMPRESS_FNS))}, or pass a "
                f"function."
            )
        return _BUILT_IN_COMPRESS_FNS[compress_fn]
    if callable(compress_fn):
        return compress_fn
    raise TypeError(
        f"compress_fn must name a built-in or be a function, got {compress_fn!r}."
    )


def _check_template(template: str) -> None:
    if not isinstance(template, str):
        raise TypeError(f"template must be a string, got {type(template).__name__}.")
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(template)]
    except ValueError as e:
        raise ValueError(f"template is not a valid format string: {e}.") from e
    for field in fields:
        # Plain names only: no positions, and no attribute or index lookups
        # reaching into the values.
        if field is not None and not field.isidentifier():
            raise ValueError(
                f"template fields must be plain names such as {{title}}, got "
                f"{{{field}}}."
            )


def _groups(
    events: Iterable[Any], group_key: Callable[[Any], Any]
) -> list[tuple[Any, ...]]:
    groups: list[tuple[Any, ...]] = []
    group: list[Any] = []
    group_value: Any = None
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError(f"Events must be mappings, got {type(event).__name__}.")
        value = group_key(event)
        if group and value != group_value:
            groups.append(tuple(group))
            group = []
        group.append(event)
        group_value = value
    if group:
        groups.append(tuple(group))
    return groups


def _items(compressed: object) -> list[Union[str, Record]]:
    if isinstance(compressed, (str, Mapping)):
        return [compressed]
    if isinstance(compressed, (list, tuple)) and all(
        isinstance(item, (str, Mapping)) for item in compressed
    ):
        return list(compressed)
    raise TypeError(
        f"compress_fn must return a string, a mapping, or a list of them, got "
        f"{type(compressed).__name__}."
    )


def _write(item: Union[str, Record], template: Optional[str]) -> str:
    if isinstance(item, str):
        return item
    if template is None:
        return ", ".join(f"{field}: {value}" for field, value in item.items())
    try:
        return template.format_map(item)
    except KeyError as e:
        raise ValueError(
            f"template uses {{{e.args[0]}}}, which a record from compress_fn does "
            f"not have. Its fields are: {', '.join(map(str, item))}."
        ) from None


def _extremes(values: list[Any]) -> tuple[Any, Any]:
    if not values:
        return None, None
    try:
        return min(values), max(values)
    except (TypeError, ValueError):
        # Unorderable values, or numpy arrays, which refuse to compare as one.
        return None, None
