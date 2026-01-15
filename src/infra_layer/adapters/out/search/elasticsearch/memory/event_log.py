# Import retained for type annotations and field definitions
from elasticsearch.dsl import field as e_field
from core.tenants.tenantize.oxm.es.tenant_aware_async_document import (
    TenantAwareAliasDoc,
)
from core.oxm.es.analyzer import (
    completion_analyzer,
    lower_keyword_analyzer,
    edge_analyzer,
    whitespace_lowercase_trim_stop_analyzer,
)


class EventLogDoc(
    TenantAwareAliasDoc("event-log", number_of_shards=1, number_of_replicas=0)
):
    """
    Event log Elasticsearch document

    Uses a separate event-log index.
    """

    class CustomMeta:
        # Specify the field name used to automatically populate meta.id
        id_source_field = "id"

    # Basic identification fields
    # id field is automatically extracted from kwargs via CustomMeta.id_source_field and set as meta.id
    user_id = e_field.Keyword()
    user_name = e_field.Keyword()

    # Timestamp field
    timestamp = e_field.Date(required=True)

    # BM25 retrieval core field - supports multi-value storage for search content
    # Application layer can store multiple related search terms or phrases
    search_content = e_field.Text(
        multi=True,
        required=True,
        # star
        analyzer="standard",
        fields={
            # Original content field - for exact matching, lowercase processing
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

    # Core content field
    atomic_fact = e_field.Text(
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
    )

    # Parent info
    parent_type = e_field.Keyword()  # Parent memory type (e.g., memcell)
    parent_id = e_field.Keyword()  # Parent memory ID

    # Extension field
    extend = e_field.Object(dynamic=True)  # Flexible extension field

    # Audit fields
    created_at = e_field.Date()
    updated_at = e_field.Date()
