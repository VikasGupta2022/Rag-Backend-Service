import uuid
import logging
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Document, DocumentChunk
from src.services.chunking import TextChunker
from src.services.embedding import BaseEmbeddingService
from src.core.exceptions import IngestionError

logger = logging.getLogger(__name__)


class IngestionService:
    """Handles parsing, chunking, embedding, and storing documents."""

    def __init__(
        self,
        embedding_service: BaseEmbeddingService,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.embedding_service = embedding_service
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def ingest_text(
        self,
        session: AsyncSession,
        title: str,
        text_content: str,
        source: Optional[str] = None,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Processes and stores a complete text document."""
        if not text_content.strip():
            raise IngestionError("Document content cannot be empty.")

        try:
            # 1. Create parent Document record
            doc = Document(
                title=title,
                source=source,
                content_type=content_type,
            )
            session.add(doc)
            await session.flush()  # Flush to populate doc.id

            # 2. Chunk text
            chunk_records = self.chunker.create_chunks_with_metadata(
                text=text_content,
                base_metadata={"title": title, "source": source, **(metadata or {})},
            )

            if not chunk_records:
                raise IngestionError("Failed to extract any text chunks.")

            chunk_texts = [c["content"] for c in chunk_records]

            # 3. Batch generate embeddings
            embeddings = await self.embedding_service.embed_documents(chunk_texts)

            # 4. Create and attach chunk models
            for chunk_data, emb in zip(chunk_records, embeddings):
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    embedding=emb,
                    metadata_json=chunk_data["metadata"],
                )
                session.add(db_chunk)

            await session.commit()
            await session.refresh(doc)
            logger.info(
                f"Successfully ingested document '{title}' (ID: {doc.id}) with {len(chunk_records)} chunks."
            )
            return doc

        except Exception as e:
            await session.rollback()
            logger.error(f"Error during document ingestion: {e}")
            raise IngestionError(f"Ingestion failed: {e}") from e
