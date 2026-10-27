# LLM Context Assembly

`feast.context` turns feature values into text for a large language model (LLM) prompt, and keeps that text inside a token budget. It is a set of plain Python functions that you call from an [on demand feature view](../../reference/beta-on-demand-feature-view.md). The feature view's output is the prompt, or a section of it.

{% hint style="warning" %}
`feast.context` is new. Nothing in Feast calls it for you yet: you call it from your own on demand feature view. It needs the `llm` extra (`feast[llm]`) for exact token counts with `tiktoken`.
{% endhint %}

## Why build prompts in a feature view

An LLM only knows what is in its prompt. To personalize an answer, an application adds facts about the user, such as their name, location or recent activity, to the prompt. Those facts are features, so Feast already stores, refreshes and serves them.

Building the prompt inside an on demand feature view gives two guarantees:

* **The same prompt offline and online.** The functions are deterministic: no clocks, no unordered iteration, and no shared state. Training data built with `get_historical_features` contains the exact prompt the model sees in production.
* **A prompt that always fits.** Feature values vary in length. A long history can push a prompt past the model's context window. `feast.context` counts tokens with the model's tokenizer and cuts the least important text first.

## Where it runs

`feast.context` runs wherever the on demand feature view runs:

| Retrieval | When the prompt is built |
|---|---|
| `get_online_features`, default (`write_to_online_store=False`) | On every read, after the source features are read from the online store |
| `get_online_features` with `write_to_online_store=True` | Once, when data is written by `materialize`, `push` or `write_to_online_store`. Reads return the stored prompt |
| `get_historical_features` | Always, after the point-in-time join |

Keep the default when the prompt uses request data, such as the user's current location: that data only exists at read time. Use `write_to_online_store=True` only when the prompt depends on stored features alone and read latency matters more than freshness.

## Building blocks

| Function or class | What it does |
|---|---|
| `TokenBudget` | An immutable token budget. `TokenBudget.of(4096, "cl100k_base")` builds one; `consume()` returns a new, smaller budget. |
| Tokenizers | `cl100k_base` and `o200k_base` (`tiktoken`), plus a character estimate used when `tiktoken` is missing. Any `Tokenizer` subclass can be passed by its class path. |
| `priority_select()` | Keeps as many prompt sections as fit, dropping `low` priority first, then `medium`, then `high`. A `critical` section is never dropped. |
| `render_with_budget()` | Renders a Jinja2 template and cuts its variables, never the template's own words, until the prompt fits. |
| `compress_sequence()` | Turns a list of events into a few summary lines, for example "Watched 8 episodes of Breaking Bad". |
| `PromptTemplate` | A named, versioned template with per-variable settings that saves to YAML or JSON, so training and serving use the same template. |

When a prompt cannot fit without cutting a `critical` variable or the template's own words, `render_with_budget()` raises `TokenBudgetExceededError` instead of serving a prompt with its instructions missing.

## Example

This on demand feature view builds a restaurant concierge's system prompt from the user's profile and their location in the request. The visit history is cut first when the prompt runs over 256 tokens; the user's name is never cut.

```python
from typing import Any

from feast import Field, RequestSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import String

location_request = RequestSource(
    name="location_request",
    schema=[Field(name="location", dtype=String)],
)


@on_demand_feature_view(
    sources=[user_profile_fv, location_request],
    schema=[Field(name="system_prompt", dtype=String)],
    mode="python",
)
def restaurant_prompt(inputs: dict[str, Any]) -> dict[str, Any]:
    # Feast rebuilds this function from its source text when it loads the
    # registry, so imports and constants must live inside the body.
    from feast.context import TokenBudget, render_with_budget

    template = """You are a concierge that recommends restaurants.
You are serving {{ name }}. Address them by name.
Suggest restaurants in {{ location }}.
Places they visited recently: {{ recent_visits }}"""

    prompts = []
    for name, location, visits in zip(
        inputs["name"], inputs["location"], inputs["recent_visits"]
    ):
        rendering = render_with_budget(
            template=template,
            context={"name": name, "location": location, "recent_visits": visits},
            budget=TokenBudget.of(256, "cl100k_base"),
            variables={"name": {"priority": "critical"}},
        )
        prompts.append(rendering.text)
    return {"system_prompt": prompts}
```

At serving time, the application passes only the entity key and the request data:

```python
store.get_online_features(
    features=["restaurant_prompt:system_prompt"],
    entity_rows=[{"user_id": "user3", "location": "Chicago"}],
)
```

Two details in this example matter:

* **Imports go inside the function.** When Feast loads a Python mode on demand feature view from the registry, it runs the function's source text again with only `pandas` and `numpy` in scope. A module-level import or constant is missing at serving time.
* **The function loops over rows.** With `mode="python"`, inputs are lists of values, one per entity row, both online and in `get_historical_features`. Do not use `singleton=True` here: historical retrieval passes whole columns to the function.

## Code

The module lives in `sdk/python/feast/context/`. Each building block has its own file: `token_budget.py`, `tokenizer.py`, `priority_select.py`, `render.py`, `compress.py` and `prompt_template.py`.
