"""
Qdrant collection converter base class — Analog zu ``BaseMilvusConverter``.

Provides a unified conversion interface from arbitrary data sources to Qdrant
collection entities (typically ``qdrant_client.http.models.PointStruct``
instances or compatible payload dicts).

All Qdrant collection converters should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar, get_args, get_origin

from core.observation.logger import get_logger
from core.oxm.qdrant.qdrant_collection_base import QdrantCollectionBase

logger = get_logger(__name__)

# Generic type variable — bound to QdrantCollectionBase so subclasses are
# explicit about which collection they convert into.
QdrantCollectionType = TypeVar("QdrantCollectionType", bound=QdrantCollectionBase)


class BaseQdrantConverter(ABC, Generic[QdrantCollectionType]):
    """
    Qdrant collection converter base class.

    Provides basic functionality for converting arbitrary data sources to
    Qdrant collection entities (point payloads). All Qdrant converters should
    inherit from this class.

    Features:
    - Unified conversion interface (class methods).
    - Type-safe Qdrant collection generic support.
    - Automatically retrieves the bound Qdrant collection type from generics.
    - Flexible data source support (Mongo docs are the typical source in
      EverOS, see ``from_mongo`` below).
    """

    @classmethod
    def get_qdrant_model(cls) -> Type[QdrantCollectionType]:
        """
        Retrieve the Qdrant collection model type from generic information.

        Returns:
            Type[QdrantCollectionType]: Qdrant collection model class.

        Raises:
            ValueError: When the subclass did not bind a generic argument.
        """
        # Get the generic base class of the current class.
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:
                if get_origin(base) is BaseQdrantConverter:
                    args = get_args(base)
                    if args:
                        return args[0]

        raise ValueError(
            "Unable to retrieve Qdrant collection type from generic information "
            f"of {cls.__name__}"
        )

    @classmethod
    @abstractmethod
    def from_mongo(cls, source_doc: Any) -> Any:
        """
        Convert from a data source (typically a Mongo doc) to a Qdrant entity.

        Subclasses must implement specific conversion logic. The concrete
        return type is typically a ``qdrant_client.http.models.PointStruct``
        (``id``, ``vector``, ``payload``) or — for converters that split one
        source doc into many points (e.g. ``UserProfile``) — a
        ``List[Dict[str, Any]]`` whose items the indexer wraps into
        ``PointStruct`` after embedding.

        The annotation is ``Any`` because the generic ``QdrantCollectionType``
        parameter documents the *target collection class*, not the wire-
        format the converter emits. Subclasses tighten the annotation
        ("-> PointStruct" or "-> List[Dict[str, Any]]") as they implement.

        Args:
            source_doc: Source data (any type — Mongo doc, dict, etc.).

        Returns:
            Either a ``PointStruct`` or a ``List[Dict[str, Any]]`` of
            per-item payload dicts, depending on the subclass.

        Raises:
            Exception: When an error occurs during conversion.
        """
        raise NotImplementedError("Subclasses must implement the from_mongo method")
