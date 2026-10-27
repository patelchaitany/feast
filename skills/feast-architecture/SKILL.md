---
name: feast-architecture
description: Internals of the Feast codebase — how each component works, where the key abstractions live, and the data flow through the system. Use when asked how feast apply works, how the registry stores data, how materialization moves data, how get_online_features retrieves features, how the feature server or MCP server works, how feast.context assembles token-budgeted LLM prompts, how the Kubernetes operator manages deployments, or when navigating the codebase to understand where to make a change.
license: Apache-2.0
compatibility: Works with Claude Code, OpenAI Codex, and any Agent Skills compatible tool.
metadata:
  author: feast-dev
  version: "1.0"
---

# Feast Architecture Internals

## Full Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│                     Feast Deployment Modes                       │
│                                                                   │
│  Local / Python SDK          Kubernetes (feast-operator)          │
│  ─────────────────           ─────────────────────────────────   │
│  feature_store.yaml    ←──   FeatureStore CR (CRD)               │
│        │                            │                             │
│        ▼                            ▼                             │
│  FeatureStore (Python)      Operator deploys services:            │
│    ├── Registry               - feature-server (Go or Python)     │
│    ├── Provider               - offline-store-server              │
│    │   ├── OnlineStore        - registry-server                   │
│    │   └── OfflineStore     + manages feature_store.yaml config   │
│    └── FeatureServer                                              │
│        (Python FastAPI or                                         │
│         Go gRPC/HTTP)                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Python SDK Core

### FeatureStore — the orchestrator

**File**: `sdk/python/feast/feature_store.py`

`FeatureStore` is the single entry point for all operations. It never reads/writes data
directly — it delegates to the registry (for metadata) and the provider (for infrastructure
and data movement).

```python
FeatureStore(repo_path=".")   # loads feature_store.yaml
store.apply(objects)           # register feature definitions
store.materialize(...)         # offline → online
store.get_online_features(...) # serve
store.get_historical_features(...) # training data
```

**File**: `sdk/python/feast/repo_config.py` — parses `feature_store.yaml` into typed `RepoConfig`.
All component classes are loaded dynamically from the `type:` string via the maps in
`repo_config.py`: `ONLINE_STORE_CLASS_FOR_TYPE`, `OFFLINE_STORE_CLASS_FOR_TYPE`,
`REGISTRY_CLASS_FOR_TYPE`, `BATCH_ENGINE_CLASS_FOR_TYPE` and `FEATURE_SERVER_CONFIG_CLASS_FOR_TYPE`.
A `type:` that is not a short name is taken as a fully-qualified class path (it must end in the
expected suffix, e.g. `...OnlineStore`).

---

### Registry

**Purpose**: Metadata store — persists definitions of entities, feature views, data sources,
feature services, permissions.

**Backends and their files:**

| Backend | File | Notes |
|---|---|---|
| File/GCS/S3 (default) | `infra/registry/registry.py` | Single proto blob, cached in memory |
| SQL | `infra/registry/sql.py` | Per-object tables via SQLAlchemy |
| Snowflake | `infra/registry/snowflake.py` | Snowflake tables |
| Remote | `infra/registry/remote.py` | Delegates to a remote registry server over gRPC |

**How the proto/file backend works:**
1. All metadata is serialized into one `Registry` protobuf (`protos/feast/core/Registry.proto`)
2. The proto blob is stored at the configured `registry:` path
3. In-memory `cached_registry_proto` is refreshed on a TTL (`cache_ttl_seconds`, default 600s).
   `cache_mode: sync` (default) refreshes on the first registry read after expiry and right after
   each write from this process; `thread` refreshes in a background thread every `cache_ttl_seconds`
4. Writes re-serialize and overwrite the full blob — no partial updates

**Key pattern — apply:**
```python
# Python object → proto → stored in registry blob
registry.apply_feature_view(feature_view, project)
# → feature_view.to_proto()
# → upserts into cached_registry_proto.feature_views
# → registry_store.update_registry_proto(proto)
```

**How the SQL backend works:**
- Per-object tables, each storing that object's serialized proto in a binary column.
- **Binary proto columns must use `ProtoBytes`, not `LargeBinary` directly** (defined at the top of `infra/registry/sql.py`). `ProtoBytes` emits `LONGBLOB` on MySQL and MariaDB and falls back to `LargeBinary`'s default on every other dialect (`BLOB` on SQLite, `BYTEA` on PostgreSQL). Plain `LargeBinary` maps to MySQL `BLOB` (64 KB cap), which silently truncates large protos (e.g. a `FeatureView`) and later fails to deserialize. Any new serialized-proto/blob-metadata column reintroduces that bug if it uses `LargeBinary`.
- All tables are declared on the module-level `metadata` object in `sql.py` (the source of truth). `metadata.create_all` only creates missing tables — it never widens existing columns, so schema changes to existing registries require a manual migration (see `docs/reference/registries/sql.md`). On MySQL/MariaDB, `SqlRegistry._warn_if_narrow_blob_columns` logs an error at startup for any registry proto column still typed as the narrow `BLOB` — a new column is covered automatically as long as it's typed `ProtoBytes` (the diagnostic selects columns by `column.type is ProtoBytes`); a column typed as plain `LargeBinary` would be silently missed.

**Supporting files:**
- `infra/registry/base_registry.py` — abstract interface
- `infra/registry/proto_registry_utils.py` — proto serialization helpers
- `infra/registry/caching_registry.py` — adds TTL caching on top of any backend

---

### Provider

**Purpose**: Infrastructure lifecycle — creates/updates/tears down online store tables.
Also dispatches `online_write_batch`, `get_historical_features` and materialization.

**Files**: `sdk/python/feast/infra/provider.py` (interface, `PROVIDERS_CLASS_FOR_TYPE`),
`sdk/python/feast/infra/passthrough_provider.py` (the implementation almost everyone uses)

Built-in providers (set via `provider:` in `feature_store.yaml`): `local`, `gcp`, `aws` and `azure`
all resolve to `PassthroughProvider`, which builds the online store, offline store and batch
compute engine from config. The provider name does not pick stores: when `online_store` /
`offline_store` are omitted, every provider defaults to SQLite online and Dask (`file`) offline.
`chronon` and `unity_catalog` map to their own provider classes.

Custom providers extend `Provider` and override `update_infra` / `teardown_infra`.

---

### Online Store

**Purpose**: Low-latency feature serving. Stores the latest feature values per entity key.

**Interface**: `sdk/python/feast/infra/online_stores/online_store.py`
**Implementations**: `sdk/python/feast/infra/online_stores/` (redis, dynamodb, sqlite, bigtable, postgres, snowflake, …)

Key methods:
- `online_write_batch` — write entity→feature values
- `online_read` — read by entity keys
- `update` — provision/deprovision tables on `feast apply`
- `teardown` — clean up on `feast teardown`

---

### Offline Store

**Purpose**: Historical feature retrieval and training data generation (point-in-time joins).

**Interface**: `sdk/python/feast/infra/offline_stores/offline_store.py`
**Implementations**: `sdk/python/feast/infra/offline_stores/` (bigquery, snowflake, redshift, duckdb, file, …)

Returns a `RetrievalJob` (lazy) — no data moves until `.to_df()` or `.to_arrow()` is called.

PIT join logic (shared): `sdk/python/feast/infra/offline_stores/offline_utils.py`

**Key methods to implement for a new backend:**
```python
class MyOfflineStore(OfflineStore):
    def get_historical_features(self, config, feature_views, feature_refs,
                                entity_df, registry, project, ...) -> RetrievalJob: ...
    def pull_latest_from_table_or_query(self, config, data_source,
                                        join_key_columns, feature_name_columns,
                                        timestamp_field, created_timestamp_column,
                                        start_date, end_date) -> RetrievalJob: ...
    def pull_all_from_table_or_query(self, config, data_source, join_key_columns,
                                     feature_name_columns, timestamp_field,
                                     start_date, end_date) -> RetrievalJob: ...
    def write_logged_features(self, config, data, source, logging_config,
                              registry) -> None: ...  # optional
```

**Config class**: subclass `FeastConfigBaseModel` with a `type` Literal (short alias + full dotted path). Register in `OFFLINE_STORE_CLASS_FOR_TYPE` in `sdk/python/feast/repo_config.py`.

**Data source**: each offline store backend pairs with a `DataSource` subclass (e.g. `BigQuerySource`, `FileSource`), which lives beside the store (e.g. `infra/offline_stores/bigquery_source.py`, `infra/offline_stores/contrib/<name>/<name>_source.py`). Register its proto `SourceType` in `_DATA_SOURCE_OPTIONS` in `sdk/python/feast/data_source.py`.

---

### LLM Context Assembly (`feast.context`)

**Purpose**: Turn feature values into LLM prompt text that fits a token budget, identically
offline and online. Plain functions over immutable values; nothing in Feast calls them yet —
users call them from an `OnDemandFeatureView` whose output is the prompt.

**Directory**: `sdk/python/feast/context/` · **Extra**: `feast[llm]` (`tiktoken`, `Jinja2`)

| File | Provides |
|---|---|
| `tokenizer.py` | `Tokenizer` ABC; `cl100k_base` / `o200k_base` (tiktoken) and `approximate`. Resolved via `TOKENIZER_CLASS_FOR_TYPE`, else as a class path through `feast.importer.import_class`. Falls back to `approximate` with a warning when tiktoken is missing (`fallback=False` makes it fatal) |
| `token_budget.py` | `TokenBudget` — frozen dataclass; `consume()` returns a new budget, `try_consume()` returns `None` instead of raising |
| `priority_select.py` | `priority_select()` — keeps `Section`s by `Priority` (`LOW`=10 … `CRITICAL`=40); critical sections are never dropped |
| `render.py` | `render_with_budget()` — renders Jinja2 and cuts template *variables* (never the template's words, non-string values, or `critical` variables) per `OverflowStrategy`; returns a `Rendering` |
| `compress.py` | `compress_sequence()` — groups consecutive events and writes each group as a line (`count`, `summarize`, `sample`, or a custom function) |
| `prompt_template.py` | `PromptTemplate` — named, versioned template with variable settings; YAML/JSON round trip |
| `errors.py` | `ContextError` and subclasses, e.g. `TokenBudgetExceededError` |

**Where it runs**: wherever the calling ODFV runs (see the `get_online_features` and
`get_historical_features` flows below) — on every online read by default, at write time with
`write_to_online_store=True`, and always after the PIT join offline. Prompts that use request
data must stay read-time.

**Determinism is the contract**: no clocks, no iteration over unordered collections, no mutation
of the budget passed in. Any change here must keep the same inputs producing the same prompt,
or training and serving prompts diverge.

**ODFV gotcha**: Python-mode UDFs are rebuilt from their source text on registry load
(`transformation/udf_rehydrate.py`, which exposes only `pd`/`np`), so `feast.context` imports and
template constants must be inside the UDF body. Inputs are column lists both online and offline,
so loop over rows rather than relying on `singleton=True`.

---

## Data Flows

### `feast apply`
```
feast apply  (CLI → repo_operations.py)
  ├── Parse Python files → collect FeastObjects
  ├── store.apply(objects)
  │     ├── diff against registry (diff/registry_diff.py)
  │     ├── update registry metadata for changed objects
  │     └── provider.update_infra(tables_to_keep, tables_to_delete)
  │           └── online_store.update(...)  ← create/drop tables
  └── Write updated registry to storage
```

### `feast materialize`
```
store.materialize(start_date, end_date)
  ├── Load feature views from registry (online, enabled ones)
  ├── ODFVs with write_to_online_store=True → _materialize_odfv()
  │     └── read source FVs, run the transform, write the output online
  ├── Regular feature views:
  │     ├── batch_engine.supports_batch → _materialize_fvs_batch(...)
  │     └── else, per view: provider.materialize_single_feature_view(...)
  │           └── batch_engine.materialize(registry, task)   ← compute engine
  │                 └── DAG: read offline source → transform → online_store.online_write_batch(...)
  └── registry.apply_materialization(...)  ← record the materialized interval
```
Compute engines (`batch_engine:` in `feature_store.yaml`) live in `infra/compute_engines/`
(`local`, `spark.engine`, `ray.engine`, `flink.engine`, `snowflake.engine`, `lambda`, `k8s`, …).

### `get_online_features`
```
store.get_online_features(features, entity_rows)
  └── provider.get_online_features(...) → OnlineStore.get_online_features(...)
        ├── _prepare_entities_to_read_from_online_store (utils.py)
        │     ├── resolve refs / FeatureService → grouped FVs, ODFVs, needed request fields (cached)
        │     └── split entity_rows: join keys vs. ODFV request data (missing request data → error)
        ├── FeatureService(precompute_online=True) and no ODFVs → read one precomputed vector
        │   else → _read_features_per_fv: per FV, dedupe entities → online_read → populate values + status
        ├── _augment_response_with_on_demand_transforms  ← ODFVs with write_to_online_store=False run HERE
        ├── _drop_unneeded_columns  ← source features fetched only to feed an ODFV are removed
        └── Return OnlineResponse
  └── optional MLflow auto-log of the retrieval (FeatureStore.get_online_features)
```
A stale registry cache is refreshed synchronously on the first call after the TTL expires.

### `get_historical_features`
```
store.get_historical_features(entity_df, features)
  ├── Resolve feature refs → FeatureViews from registry
  ├── offline_store.get_historical_features(config, feature_views, entity_df)
  │     └── Point-in-time join:
  │           for each entity row, find latest values where
  │           event_timestamp ≤ entity_df.event_timestamp
  │           (prevents data leakage in training)
  └── Returns RetrievalJob → .to_df() / .to_arrow()
        └── when the job executes, every requested ODFV runs on the joined table
            (RetrievalJob in offline_store.py), regardless of write_to_online_store
```

---

## Feature Servers

### Python Feature Server (FastAPI)

**File**: `sdk/python/feast/feature_server.py`

A FastAPI app that wraps `FeatureStore`. Started with `feast serve`.

Endpoints:
- `POST /get-online-features` — online feature retrieval
- `POST /search`, `POST /retrieve-online-documents` — vector / document retrieval
- `GET /v1/vector_stores`, `GET /v1/vector_stores/{id}`, `POST /v1/vector_stores/{id}/search` — vector store API
- `POST /push` — push features to online/offline store
- `POST /write-to-online-store` — write a DataFrame directly to the online store
- `POST /materialize`, `POST /materialize-incremental` — trigger materialization
- `GET /health` — health check

The app loads `feature_store.yaml` at startup, creates a `FeatureStore`, and periodically
refreshes the registry in the background (async timer).

### MCP servers

Feast exposes tools to LLM agents over the Model Context Protocol in two ways. Both expose
generic tools (e.g. one `get_online_features` taking feature refs and entity values) — there is
no per-feature-view tool, so the agent discovers feature names and supplies entity keys itself.

| | In-process | Standalone (`feast mcp`) |
|---|---|---|
| Code | `infra/mcp_servers/mcp_server.py` | `sdk/python/feast/mcp/` (`server.py`, `features.py`, `registry.py`) |
| Built on | `fastapi_mcp`, mounted on the feature server app | `fastmcp`, a separate process |
| Enabled by | `feature_server: {type: mcp, mcp_enabled: true}` | `feast mcp` CLI, `feast_mcp.yaml` |
| Tools come from | the feature server's FastAPI routes | hand-written tools proxying over HTTP to the feature server and REST registry server |
| Extra | `feast[mcp]` | `feast[mcp-server]` |

### Go Feature Server

**Directory**: `go/`
**Entry point**: `go/main.go`

A high-performance alternative to the Python feature server, written in Go.
Supports HTTP, HTTPS, and gRPC transports:

```bash
go run go/main.go -type http  -port 6566
go run go/main.go -type grpc  -port 6566
```

Key packages:
- `go/internal/feast/` — Go port of FeatureStore (reads feature_store.yaml, calls online store)
- `go/internal/feast/server/` — HTTP and gRPC server implementations
- `go/internal/feast/server/logging/` — feature logging to offline store

The Go server reads the registry directly (proto file or remote) and calls the online store.
It does not support `feast apply` or materialization — those remain Python-only.

---

## Feast Operator (Kubernetes)

**Directory**: `infra/feast-operator/`
**Language**: Go (controller-runtime / kubebuilder)

The operator manages the full lifecycle of a Feast deployment on Kubernetes via a
`FeatureStore` Custom Resource Definition (CRD).

### CRD: FeatureStore

**API version**: `feast.dev/v1`
**File**: `infra/feast-operator/api/v1/featurestore_types.go`

```yaml
apiVersion: feast.dev/v1
kind: FeatureStore
metadata:
  name: my-feast
spec:
  feastProjectName: my_project
  services:
    offlineStore:
      persistence:
        file:
          type: dask
    onlineStore:
      persistence:
        store:
          type: redis
          secretRef:
            name: redis-credentials
    registry:
      local:
        persistence:
          file:
            path: /data/registry.db
```

### What the operator manages

| Service | What it deploys |
|---|---|
| **Online Store server** | Deployment + Service for the feature server (Go or Python) |
| **Offline Store server** | Deployment + Service for the offline feature server |
| **Registry server** | Deployment + Service for the registry gRPC server (optional REST API) |
| **UI** | Deployment + Service for the Feast web UI (`spec.services.ui`) |
| **MCP server** | Standalone `feast mcp` container (`spec.services.mcpServer`); needs the online store or the registry REST API as its upstream |
| **feature_store.yaml** | ConfigMap auto-generated from the CR spec |
| **Materialization jobs** | CronJob (`spec.services.onlineStore.cronJob`) |
| **TLS** | Certificate management via `spec.services.*.tls` |
| **Auth** | OIDC / Kubernetes RBAC via `spec.authz` |

### Reconcile loop

**File**: `infra/feast-operator/internal/controller/featurestore_controller.go`

The `FeatureStoreReconciler.Reconcile` method runs on every CR change:
1. Fetches the `FeatureStore` CR
2. Calls `deployFeast()` → creates/updates Deployments, Services, ConfigMaps
3. Updates CR status conditions (`OfflineStore`, `OnlineStore`, `Registry`, `UI`, `McpServer` ready conditions)
4. Watches owned resources; re-reconciles on any change

Supporting services logic: `infra/feast-operator/internal/controller/services/`

---

## Serialization Layer

All persistent metadata and the feature server wire format use **Protocol Buffers**.

```
Python object (FeatureView, Entity, ...)
    ├── .to_proto()   → Protobuf message → stored in registry or sent over gRPC
    └── .from_proto() ← Protobuf message

Proto definitions:
  protos/feast/core/    # registry objects (FeatureView, Entity, DataSource, …)
  protos/feast/serving/ # serving API (GetOnlineFeaturesRequest/Response)
  protos/feast/types/   # Value, EntityKey, Field
```

When adding a new field to a Feast object:
1. Update the `.proto` file
2. Run `make compile-protos-python` (and `make compile-protos-go` if applicable)
3. Update `.to_proto()` and `.from_proto()` in the Python class

---

## Key Files Quick Reference

| Concern | Key file(s) |
|---|---|
| User-facing Python API | `sdk/python/feast/feature_store.py` |
| Config parsing | `sdk/python/feast/repo_config.py` |
| `feast apply` CLI logic | `sdk/python/feast/repo_operations.py` |
| Registry diff | `sdk/python/feast/diff/registry_diff.py` |
| Registry (proto/file) | `sdk/python/feast/infra/registry/registry.py` |
| Registry (SQL) | `sdk/python/feast/infra/registry/sql.py` |
| PIT join | `sdk/python/feast/infra/offline_stores/offline_utils.py` |
| Online store interface | `sdk/python/feast/infra/online_stores/online_store.py` |
| Entity key serialization | `sdk/python/feast/infra/online_stores/helpers.py` |
| Python feature server | `sdk/python/feast/feature_server.py` |
| Materialization (compute engines) | `sdk/python/feast/infra/compute_engines/` |
| ODFV online transforms | `sdk/python/feast/utils.py` (`_augment_response_with_on_demand_transforms`) |
| ODFV UDF rehydration | `sdk/python/feast/transformation/udf_rehydrate.py` |
| MCP (in-process / standalone) | `sdk/python/feast/infra/mcp_servers/`, `sdk/python/feast/mcp/` |
| LLM context assembly | `sdk/python/feast/context/` |
| Go feature server | `go/main.go`, `go/internal/feast/server/` |
| Operator CRD types | `infra/feast-operator/api/v1/featurestore_types.go` |
| Operator controller | `infra/feast-operator/internal/controller/featurestore_controller.go` |
| Operator services | `infra/feast-operator/internal/controller/services/` |
| Proto definitions | `protos/feast/` |
| Web UI | `ui/` (React) |

---

## Official Architecture Documentation

The `docs/` directory contains user-facing architecture documentation that complements this skill:

| Topic | Doc |
|---|---|
| Architecture overview | `docs/getting-started/architecture/overview.md` |
| Push vs pull model | `docs/getting-started/architecture/push-vs-pull-model.md` |
| Write patterns | `docs/getting-started/architecture/write-patterns.md` |
| Feature transformation | `docs/getting-started/architecture/feature-transformation.md` |
| RBAC / authorization | `docs/getting-started/architecture/rbac.md` |
| LLM context assembly | `docs/getting-started/architecture/llm-context-assembly.md` |
| Online store component | `docs/getting-started/components/online-store.md` |
| Offline store component | `docs/getting-started/components/offline-store.md` |
| Registry component | `docs/getting-started/components/registry.md` |
| Feature server component | `docs/getting-started/components/feature-server.md` |
| Provider component | `docs/getting-started/components/provider.md` |
| Compute engine | `docs/getting-started/components/compute-engine.md` |
| ADRs (design decisions) | `docs/adr/` |
