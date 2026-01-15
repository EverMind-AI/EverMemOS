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


class EpisodicMemoryDoc(
    TenantAwareAliasDoc("episodic-memory", number_of_shards=1, number_of_replicas=0)
):
    """
    Episodic memory Elasticsearch document

    Based on MongoDB EpisodicMemory model, used for efficient BM25 text retrieval.
    Main retrieval field is the concatenated content of title and episode.

    Field descriptions:
    - event_id: Event unique identifier (corresponds to MongoDB _id)
    - user_id: User ID (required, used for filtering)
    - user_name: User name
    - timestamp: Event occurrence time
    - title: Event title (corresponds to MongoDB subject field)
    - episode: Episodic description (core content)
    - search_content: BM25 search field (supports multi-value storage, used for exact word matching)
    - summary: Event summary
    - group_id: Group ID (optional)
    - participants: List of participants
    - type: Event type (Conversation, etc.)
    - keywords: List of keywords
    - linked_entities: List of linked entity IDs
    - extend: Extension field (flexible storage)

    Tokenization notes:
    - Application layer is responsible for Chinese tokenization (jieba recommended)
    - title, episode, and summary fields store pre-tokenized results (space-separated)
    - search_content field supports multi-value storage, each value being a search term
    - ES uses standard analyzer for search_content, original sub-field for exact matching
    - During search, use terms query to match multiple terms in search_content.original field

    Sub-fields description:
    - original: Exact match, lowercase processed
    - ik: IK intelligent tokenization (requires IK plugin installed in ES)
    - edge_completion: Prefix matching and autocomplete
    """

    class CustomMeta:
        # Specify the field name used to automatically populate meta.id
        id_source_field = "event_id"

    # Basic identifier fields
    event_id = e_field.Keyword(required=True)
    user_id = e_field.Keyword()  # None for group memory
    user_name = e_field.Keyword()

    # Timestamp field
    timestamp = e_field.Date(required=True)

    # Core content fields - primary target for BM25 retrieval
    title = e_field.Text(
        required=False,
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
        fields={
            "keyword": e_field.Keyword(),  # Exact match
            # "completion": e_field.Completion(analyzer=completion_analyzer),  # Autocomplete
        },
    )

    episode = e_field.Text(
        required=True,
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
        fields={"keyword": e_field.Keyword()},  # Exact match
    )

    # Core BM25 retrieval field - search content supporting multi-value storage
    # Application layer can store multiple related search terms or phrases
    search_content = e_field.Text(
        multi=True,
        required=True,
        # star
        analyzer="standard",
        fields={
            # Original content field - for exact matching, lowercase processed
            "original": e_field.Text(
                analyzer=lower_keyword_analyzer, search_analyzer=lower_keyword_analyzer
            ),
            # # IK intelligent tokenization field - requires IK plugin installed
            # "ik": e_field.Text(
            #     analyzer="ik_smart",
            #     search_analyzer="ik_smart"
            # ),
            # Edge N-gram field - for prefix matching and autocomplete
            # "edge_completion": e_field.Text(
            #     analyzer=edge_analyzer,
            #     search_analyzer=lower_keyword_analyzer
            # ),
        },
    )

    # Summary field
    summary = e_field.Text(
        analyzer=whitespace_lowercase_trim_stop_analyzer,
        search_analyzer=whitespace_lowercase_trim_stop_analyzer,
    )

    # Classification and tagging fields
    group_id = e_field.Keyword()  # Group ID
    participants = e_field.Keyword(multi=True)

    type = e_field.Keyword()  # Conversation/Email/Notion, etc.
    keywords = e_field.Keyword(multi=True)  # List of keywords
    linked_entities = e_field.Keyword(multi=True)  # List of linked entity IDs

    subject = e_field.Text()  # Event title

    # todo: will abandon this field
    memcell_event_id_list = e_field.Keyword(multi=True)  # List of memory cell event IDs

    # Parent info
    parent_type = e_field.Keyword()  # Parent memory type (e.g., memcell)
    parent_id = e_field.Keyword()  # Parent memory ID

    # Extension field
    extend = e_field.Object(dynamic=True)  # Flexible extension field

    # Audit fields
    created_at = e_field.Date()
    updated_at = e_field.Date()
