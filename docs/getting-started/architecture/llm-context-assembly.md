# LLM Context Assembly

`feast.context` turns feature values into text for a large language model (LLM) prompt, and keeps that text inside a token budget. The `@prompt` decorator puts that to work: it turns a function into a prompt that Feast stores in the registry, versions, and fills in from features on request.

Feast only builds the prompt. Calling the model, running an agent and choosing tools stay with your application, in whatever framework it uses.

{% hint style="warning" %}
`feast.context` is new. It needs the `llm` extra (`feast[llm]`) for exact token counts with `tiktoken`.
{% endhint %}

## Prompts are definitions, features are data

An LLM only knows what is in its prompt. To personalize an answer, an application adds facts about the user, such as their name, location or recent activity, to the prompt. Those facts are features, so Feast already stores, refreshes and serves them. The prompt around them is not a feature: it does not change over time per user, it changes by version when someone edits it.

| Thing | What it is | Where it lives |
|---|---|---|
| The prompt: your function, its sources, its token budget | A versioned definition | The registry, next to feature views and feature services |
| The feature values it reads | Data that changes over time | The online and offline stores |

So a prompt is never written to the online store, and there is nothing to materialize for it. Feast calls your function when it is asked for a prompt.

## Defining a prompt

This prompt is a restaurant concierge's system prompt. It reads the user's name and recent visits from a feature view, and their location from the request.

```python
from feast import Field, RequestSource
from feast.prompt import prompt
from feast.types import String

location_request = RequestSource(
    name="location_request",
    schema=[Field(name="location", dtype=String)],
)


@prompt(sources=[user_profile, location_request], max_tokens=256)
def restaurant_prompt(user_profile, request):
    return (
        "You are a concierge that recommends restaurants.\n"
        f"You are serving {user_profile['name']}. Address them by name.\n"
        f"Suggest restaurants in {request['location']}.\n"
        f"Places they visited recently: {user_profile['recent_visits']}"
    )
```

* The function takes one argument per source, in order. Each is a dict for one entity row: a feature view's features, or a request source's fields. It returns the prompt.
* `sources` takes feature views, projections such as `user_profile[["name"]]`, and request sources.
* Feast stores the function's source text and rebuilds the function from it wherever the prompt is rendered, with only `pandas` and `numpy` in scope. Import anything else inside the function. A function that uses a module-level import or variable fails when it is defined, not when it is served.
* With `max_tokens`, a prompt that runs over is refused with `TokenBudgetExceededError`.

To have Feast cut the prompt to fit instead, return the template's variables and pass a `template`. The variables are cut to fit `max_tokens`, least important first; a `critical` one is never cut:

```python
@prompt(
    sources=[user_profile, location_request],
    template="""You are a concierge that recommends restaurants.
You are serving {{ name }}. Address them by name.
Suggest restaurants in {{ location }}.
Places they visited recently: {{ recent_visits }}""",
    max_tokens=256,
    variables={"name": {"priority": "critical"}},
)
def restaurant_prompt(user_profile, request):
    return {
        "name": user_profile["name"],
        "location": request["location"],
        "recent_visits": user_profile["recent_visits"],
    }
```

Apply it with the rest of the repository: `feast apply`.

## Versions

Every time `feast apply` sees a changed prompt, whether its function, sources or settings, it stores the new definition as the next version and keeps the old ones.

* `restaurant_prompt` means the latest version; `restaurant_prompt@v2` pins version 2.
* Applying an unchanged prompt keeps its version. Going back to an old function stores it again as a new version.
* `feast prompts list-versions restaurant_prompt` and `store.list_prompt_versions("restaurant_prompt")` list every version.
* Each rendered prompt carries its name and version. Log its `reference` (for example `restaurant_prompt@v2`) with each model call to tell outputs of different versions apart.

## Using a prompt

| From | Call | Returns |
|---|---|---|
| Python | `store.render_prompt("restaurant_prompt", [{"user_id": "user3", "location": "Chicago"}])` | One `RenderedPrompt` per entity row: `text`, `version`, `tokens`, `truncated_variables` |
| Any language | `POST /render-prompt` on the feature server, with `{"prompt": "restaurant_prompt@v2", "entities": {"user_id": ["user3"], "location": ["Chicago"]}}` | The same, in JSON |
| Definition only | `store.get_prompt("restaurant_prompt@v1")`, `GET /prompts/restaurant_prompt@v1`, or `feast prompts describe restaurant_prompt@v1` | The function's source and settings |
| Training and evaluation | `get_historical_features(entity_df, features=prompt.features, full_feature_names=True)`, then `prompt.render(row)` for each row | The prompt each row would have been served, from point-in-time features |

`render_prompt` reads the prompt's features with `get_online_features`, then calls the function on each row. Entity rows carry the entity keys and the request fields:

```python
[rendered] = store.render_prompt(
    "restaurant_prompt", [{"user_id": "user3", "location": "Chicago"}]
)
send_to_model(system=rendered.text, prompt_version=rendered.reference)
```

The prompt is rendered wherever `render_prompt` runs. Behind the feature server, that is the server, so its image needs `feast[llm]`.

### Registries and permissions

* The file registry and the SQL registry store prompts. The SQL registry keeps them in a `prompts` table, one row per version; with `schema_mode: verify`, create it first with `feast registry create-schema`.
* Remote and Snowflake registries do not store prompts yet: they list none, and applying or reading one raises an error.
* On the feature server, reading or rendering a prompt requires `READ_ONLINE` on every feature view the prompt reads.

## Building blocks

| Function or class | What it does |
|---|---|
| `@prompt` | Turns a function into a `Prompt`: a versioned registry object, rendered on request from the features it reads. |
| `TokenBudget` | An immutable token budget. `TokenBudget.of(4096, "cl100k_base")` builds one; `consume()` returns a new, smaller budget. |
| Tokenizers | `cl100k_base` and `o200k_base` (`tiktoken`), plus a character estimate used when `tiktoken` is missing. Any `Tokenizer` subclass can be passed by its class path. |
| `priority_select()` | Keeps as many prompt sections as fit, dropping `low` priority first, then `medium`, then `high`. A `critical` section is never dropped. |
| `render_with_budget()` | Renders a Jinja2 template and cuts its variables, never the template's own words, until the prompt fits. |
| `compress_sequence()` | Turns a list of events into a few summary lines, for example "Watched 8 episodes of Breaking Bad". |
| `PromptTemplate` | A named template with per-variable settings that saves to YAML or JSON. `@prompt` with a `template` renders with one. |

## Building a prompt inside an on demand feature view

When the prompt must be a feature column, for example to store it at write time, the building blocks also work inside an [on demand feature view](../../reference/beta-on-demand-feature-view.md). The view then runs on every `get_online_features` read by default, once at write time with `write_to_online_store=True`, and always after the point-in-time join in `get_historical_features`.

```python
from typing import Any

from feast.on_demand_feature_view import on_demand_feature_view


@on_demand_feature_view(
    sources=[user_profile_fv, location_request],
    schema=[Field(name="system_prompt", dtype=String)],
    mode="python",
)
def restaurant_prompt_view(inputs: dict[str, Any]) -> dict[str, Any]:
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

Two details matter here:

* **Imports go inside the function.** When Feast loads a Python mode on demand feature view from the registry, it runs the function's source text again with only `pandas` and `numpy` in scope. A module-level import or constant is missing at serving time.
* **The function loops over rows.** With `mode="python"`, inputs are lists of values, one per entity row, both online and in `get_historical_features`. Do not use `singleton=True` here: historical retrieval passes whole columns to the function.

A prompt built this way is a feature, not a `@prompt`: it has no versions of its own, and is read with `get_online_features`.

## Code

* `sdk/python/feast/prompt.py`: the `@prompt` decorator, `Prompt` and `RenderedPrompt`.
* `protos/feast/core/Prompt.proto`: one message per prompt version, in `Registry.prompts`.
* Registry: `apply_prompt`, `get_prompt`, `list_prompts`, `list_prompt_versions` and `delete_prompt` in `infra/registry/base_registry.py`, `registry.py` (file) and `sql.py`, with shared readers in `proto_registry_utils.py`.
* `feature_store.py`: `get_prompt`, `list_prompts`, `list_prompt_versions` and `render_prompt`. `feature_server.py`: `GET /prompts/{name}` and `POST /render-prompt`. `cli/prompts.py`: `feast prompts`.
* `sdk/python/feast/context/`: the building blocks, one file each: `token_budget.py`, `tokenizer.py`, `priority_select.py`, `render.py`, `compress.py` and `prompt_template.py`.
