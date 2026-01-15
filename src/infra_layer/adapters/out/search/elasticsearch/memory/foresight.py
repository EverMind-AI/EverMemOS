# Import retained for type annotations and field definitions
from elasticsearch.dsl import field as e_field
from core.tenants.tenantize.oxm.es.tenant_aware_async_document import (
    TenantAwareAliasDoc,
)
from core.tenants.tenantize.oxm.es.tenant_aware_async_document import (
    TenantAwareAliasDoc,
)
from core.oxm.es.analyzer import (
    completion_analyzer,
    lower_keyword_analyzer,
    edge_analyzer,
    whitespace_lowercase_trim_stop_analyzer,
)


class ForesightDoc(
    TenantAwareAliasDoc("foresight", number_of_shards=1, number_of_replicas=0)
):
    """
    Foresight Elasticsearch document

    Uses a separate foresight index.
    """

    class CustomMeta:
        # Specify the field name used to automatically populate meta.id
        id_source_field = "id"

    # Basic identification fields
    # The id field is automatically extracted from kwargs via CustomMeta.id_source_field and set as meta.id
    user_id = e_field.Keyword()
    user_name = e_field.Keyword()

    # Timestamp field
    timestamp = e_field.Date(required=True)

    # Core content fields
    foresight = e_field.Text(
        required=True,
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
        fields={"keyword": e_field.Keyword()},
    )
    evidence = e_field.Text(
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
        fields={"keyword": e_field.Keyword()},
    )

    # BM25 retrieval core field
    search_content = e_field.Text(
        multi=True,
        required=True,
        analyzer="standard",
        fields={
            "original": e_field.Text(
                analyzer=lower_keyword_analyzer, search_analyzer=lower_keyword_analyzer
            )
        },
    )

    # Categorization and tagging fields
    group_id = e_field.Keyword()  # Group ID
    group_name = e_field.Keyword()  # Group name
    participants = e_field.Keyword(multi=True)

    type = e_field.Keyword()  # Conversation/Email/Notion, etc.

    # Parent info
    parent_type = e_field.Keyword()  # Parent memory type (e.g., memcell)
    parent_id = e_field.Keyword()  # Parent memory ID

    # Extension field
    extend = e_field.Object(dynamic=True)  # Flexible extension field

    # Audit fields
    created_at = e_field.Date()
    updated_at = e_field.Date()
