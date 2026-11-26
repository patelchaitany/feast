"""Features MCP sub-server.

Exposes tools that proxy to the Feast feature server (online feature
retrieval, LLM prompt assembly, vector search, push, online writes, and
materialization).
"""

from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from feast.mcp.auth import get_auth_token
from feast.mcp.client import FeastClient, path_segment


def _one_row(entity_keys: Dict[str, Any]) -> Dict[str, List[Any]]:
    """The feature server's column form of one entity row."""
    return {key: [value] for key, value in entity_keys.items()}


def _check_token_budget(
    text: str, max_tokens: Optional[int], tokenizer: str, source: str
) -> None:
    """Refuse a prompt longer than the caller's budget, saying by how much."""
    if max_tokens is None:
        return
    from feast.context.errors import ContextError
    from feast.context.tokenizer import get_tokenizer

    try:
        tokens = get_tokenizer(tokenizer, fallback=False).count_tokens(text)
    except (ContextError, TypeError) as e:
        raise ToolError(f"Cannot count tokens with {tokenizer!r}: {e}") from e
    if tokens > max_tokens:
        raise ToolError(
            f"The prompt from {source} is {tokens} tokens ({tokenizer}), over "
            f"max_tokens={max_tokens}. Raise max_tokens, or shorten the prompt "
            "where it is built."
        )


def create_features_mcp(client: FeastClient) -> FastMCP:
    features_mcp = FastMCP("feast-features")

    @features_mcp.tool
    async def get_online_features(
        features: List[str],
        entities: Dict[str, List[Any]],
        feature_service: Optional[str] = None,
        full_feature_names: bool = False,
        include_feature_view_version_metadata: bool = False,
    ) -> Any:
        """Retrieve online feature values for a set of entities.

        Args:
            features: Feature references in 'feature_view:feature' format.
            entities: Entity key-value map — each key maps to a list of values.
            feature_service: Optional feature service name (used instead of features list).
            full_feature_names: If true, response keys include the feature view name.
            include_feature_view_version_metadata: If true, the response metadata
                says which feature view version served each feature.
        """
        body: Dict[str, Any] = {
            "features": features,
            "entities": entities,
            "full_feature_names": full_feature_names,
        }
        if feature_service is not None:
            body["feature_service"] = feature_service
        if include_feature_view_version_metadata:
            body["include_feature_view_version_metadata"] = True
        return await client.request(
            "POST", "/get-online-features", token=get_auth_token(), json=body
        )

    @features_mcp.tool
    async def assemble_context(
        feature_service: str,
        entity_keys: Dict[str, Any],
        max_tokens: Optional[int] = None,
        tokenizer: str = "cl100k_base",
    ) -> str:
        """Assemble the LLM prompt a text-mode feature service builds for one entity.

        Works with feature services defined with output_format="text" (list
        them with registry_list_prompt_services). Returns the prompt as plain
        text, ready to send to a model.

        Args:
            feature_service: Name of a feature service with output_format="text".
            entity_keys: One value per entity key and request field the service
                reads, e.g. {"user_id": "u123"}.
            max_tokens: Your context budget. A longer prompt is refused with its
                token count, never cut.
            tokenizer: Tokenizer that counts tokens for max_tokens:
                "cl100k_base", "o200k_base" or "approximate".
        """
        text = await client.request(
            "POST",
            "/get-online-features",
            token=get_auth_token(),
            json={
                "features": [],
                "entities": _one_row(entity_keys),
                "feature_service": feature_service,
            },
            accept="text/plain",
        )
        if not isinstance(text, str):
            raise ToolError(
                f"Feature service {feature_service!r} does not return text. Use "
                'one defined with output_format="text" (see '
                "registry_list_prompt_services), or read its features with "
                "features_get_online_features."
            )
        _check_token_budget(
            text, max_tokens, tokenizer, f"feature service {feature_service!r}"
        )
        return text

    @features_mcp.tool
    async def get_prompt(prompt: str, entity_keys: Dict[str, Any]) -> str:
        """Render a versioned @prompt from the registry for one entity.

        Returns the prompt as plain text. The prompt's own max_tokens applies.

        Args:
            prompt: Prompt name: "concierge" for its latest version, or
                "concierge@v2" to pin version 2.
            entity_keys: One value per entity key and request field the prompt
                reads, e.g. {"user_id": "u123"}.
        """
        result = await client.request(
            "POST",
            "/render-prompt",
            token=get_auth_token(),
            json={"prompt": prompt, "entities": _one_row(entity_keys)},
        )
        return result["prompts"][0]["text"]

    @features_mcp.tool
    async def search(
        features: List[str],
        top_k: int = 5,
        query: Optional[List[float]] = None,
        query_string: Optional[str] = None,
        feature_service: Optional[str] = None,
        distance_metric: Optional[str] = None,
        full_feature_names: bool = False,
        api_version: int = 2,
        filters: Optional[Dict[str, Any]] = None,
        include_feature_view_version_metadata: bool = False,
    ) -> Any:
        """Vector similarity search against online document embeddings.

        Args:
            features: Feature references to retrieve.
            top_k: Number of nearest results to return.
            query: Query embedding vector (list of floats).
            query_string: Text query (used when the server handles embedding).
            feature_service: Optional feature service name.
            distance_metric: Optional distance metric override.
            full_feature_names: If true, response keys include the feature view name.
            api_version: API version for the search endpoint. Filters need 2.
            filters: Optional metadata filter, applied together with the vector
                match. A comparison is {"type": "eq"|"ne"|"gt"|"gte"|"lt"|"lte"|"in"|"nin",
                "key": "<field>", "value": ...}; combine several with
                {"type": "and"|"or", "filters": [...]}.
            include_feature_view_version_metadata: If true, the response metadata
                says which feature view version served each feature.
        """
        body: Dict[str, Any] = {
            "features": features,
            "top_k": top_k,
            "full_feature_names": full_feature_names,
            "api_version": api_version,
        }
        if query is not None:
            body["query"] = query
        if query_string is not None:
            body["query_string"] = query_string
        if feature_service is not None:
            body["feature_service"] = feature_service
        if distance_metric is not None:
            body["distance_metric"] = distance_metric
        if filters is not None:
            body["filters"] = filters
        if include_feature_view_version_metadata:
            body["include_feature_view_version_metadata"] = True
        return await client.request(
            "POST", "/search", token=get_auth_token(), json=body
        )

    @features_mcp.tool
    async def list_vector_stores() -> Any:
        """List all available vector stores."""
        return await client.request("GET", "/v1/vector_stores", token=get_auth_token())

    @features_mcp.tool
    async def get_vector_store(vector_store_id: str) -> Any:
        """Get details of a specific vector store.

        Args:
            vector_store_id: Identifier of the vector store.
        """
        return await client.request(
            "GET",
            f"/v1/vector_stores/{path_segment(vector_store_id)}",
            token=get_auth_token(),
        )

    @features_mcp.tool
    async def vector_store_search(
        vector_store_id: str,
        query: Union[str, List[str]],
        max_num_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        features_to_retrieve: Optional[List[str]] = None,
    ) -> Any:
        """OpenAI-compatible vector store search.

        Args:
            vector_store_id: Identifier of the vector store to search.
            query: Text query or list of text queries.
            max_num_results: Maximum number of results to return.
            filters: Optional metadata filter, applied together with the vector
                match. A comparison is {"type": "eq"|"ne"|"gt"|"gte"|"lt"|"lte"|"in"|"nin",
                "key": "<field>", "value": ...}; combine several with
                {"type": "and"|"or", "filters": [...]}.
            features_to_retrieve: Feature names to return in each result's
                attributes (all non-vector features if omitted).
        """
        body: Dict[str, Any] = {
            "query": query,
            "max_num_results": max_num_results,
        }
        if filters is not None:
            body["filters"] = filters
        if features_to_retrieve is not None:
            body["metadata"] = {"features_to_retrieve": features_to_retrieve}
        return await client.request(
            "POST",
            f"/v1/vector_stores/{path_segment(vector_store_id)}/search",
            token=get_auth_token(),
            json=body,
        )

    @features_mcp.tool
    async def push(
        push_source_name: str,
        df: Dict[str, Any],
        to: str = "online",
        allow_registry_cache: bool = True,
        transform_on_write: bool = True,
    ) -> str:
        """Push features into the online or offline store.

        Args:
            push_source_name: Name of the push source defined in the feature repo.
            df: Column-oriented dict representing the DataFrame to push.
            to: Target store — 'online', 'offline', or 'online_and_offline'.
            allow_registry_cache: Allow using the cached registry.
            transform_on_write: Apply on-demand transforms before writing.
        """
        body: Dict[str, Any] = {
            "push_source_name": push_source_name,
            "df": df,
            "to": to,
            "allow_registry_cache": allow_registry_cache,
            "transform_on_write": transform_on_write,
        }
        await client.request("POST", "/push", token=get_auth_token(), json=body)
        return "ok"

    @features_mcp.tool
    async def write_to_online_store(
        feature_view_name: str,
        df: Dict[str, Any],
        allow_registry_cache: bool = True,
        transform_on_write: bool = True,
    ) -> str:
        """Write rows straight into a feature view in the online store.

        Unlike push, this targets a feature view by name and needs no push
        source, which suits values an agent produces itself (for example memory).

        Args:
            feature_view_name: Name of the feature view to write to.
            df: Column-oriented dict representing the DataFrame to write. Include
                the entity columns and the event timestamp column.
            allow_registry_cache: Allow using the cached registry.
            transform_on_write: Apply on-demand transforms before writing.
        """
        body: Dict[str, Any] = {
            "feature_view_name": feature_view_name,
            "df": df,
            "allow_registry_cache": allow_registry_cache,
            "transform_on_write": transform_on_write,
        }
        await client.request(
            "POST", "/write-to-online-store", token=get_auth_token(), json=body
        )
        return "ok"

    def _materialize_params(async_mode: bool, force: bool) -> Optional[Dict[str, str]]:
        # The feature server reads these from the URL query, not the JSON body.
        params = {}
        if async_mode:
            params["async"] = "true"
        if force:
            params["force"] = "true"
        return params or None

    @features_mcp.tool
    async def materialize(
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        feature_views: Optional[List[str]] = None,
        disable_event_timestamp: bool = False,
        full_feature_names: bool = False,
        version: Optional[str] = None,
        async_mode: bool = False,
        force: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Materialize features from the offline store to the online store.

        Args:
            start_ts: Start timestamp (ISO-8601).
            end_ts: End timestamp (ISO-8601).
            feature_views: Specific feature views to materialize (all if omitted).
            disable_event_timestamp: Materialize all rows, ignoring event
                timestamps (start_ts and end_ts are then optional).
            full_feature_names: Use 'feature_view__feature' names.
            version: Feature view version to materialize (for example 'v2').
                Needs exactly one feature view and online feature view versioning.
            async_mode: Start the job in the background and return at once with
                {"status": "accepted", "feature_views": [...]}. Use it for long
                jobs that would outlast the MCP request timeout.
            force: With async_mode, restart feature views stuck in the
                materializing state instead of failing with HTTP 409.
        """
        body: Dict[str, Any] = {}
        if start_ts is not None:
            body["start_ts"] = start_ts
        if end_ts is not None:
            body["end_ts"] = end_ts
        if feature_views is not None:
            body["feature_views"] = feature_views
        if disable_event_timestamp:
            body["disable_event_timestamp"] = True
        if full_feature_names:
            body["full_feature_names"] = True
        if version is not None:
            body["version"] = version
        result = await client.request(
            "POST",
            "/materialize",
            token=get_auth_token(),
            json=body,
            params=_materialize_params(async_mode, force),
        )
        return result if async_mode else "ok"

    @features_mcp.tool
    async def materialize_incremental(
        end_ts: str,
        feature_views: Optional[List[str]] = None,
        full_feature_names: bool = False,
        version: Optional[str] = None,
        async_mode: bool = False,
        force: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Run incremental materialization up to the given timestamp.

        Args:
            end_ts: End timestamp (ISO-8601).
            feature_views: Specific feature views to materialize (all if omitted).
            full_feature_names: Use 'feature_view__feature' names.
            version: Feature view version to materialize (for example 'v2').
                Needs exactly one feature view and online feature view versioning.
            async_mode: Start the job in the background and return at once with
                {"status": "accepted", "feature_views": [...]}. Use it for long
                jobs that would outlast the MCP request timeout.
            force: With async_mode, restart feature views stuck in the
                materializing state instead of failing with HTTP 409.
        """
        body: Dict[str, Any] = {"end_ts": end_ts}
        if feature_views is not None:
            body["feature_views"] = feature_views
        if full_feature_names:
            body["full_feature_names"] = True
        if version is not None:
            body["version"] = version
        result = await client.request(
            "POST",
            "/materialize-incremental",
            token=get_auth_token(),
            json=body,
            params=_materialize_params(async_mode, force),
        )
        return result if async_mode else "ok"

    @features_mcp.tool
    async def health() -> Any:
        """Check the health of the Feast feature server."""
        return await client.request("GET", "/health", token=get_auth_token())

    return features_mcp
