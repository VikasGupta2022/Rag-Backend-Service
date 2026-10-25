import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.session import get_db, is_db_connected
from src.db.models import Document, DocumentChunk
from src.services.embedding import get_embedding_service, BaseEmbeddingService
from src.services.ingestion import IngestionService
from src.services.chunking import TextChunker
from src.services.in_memory_store import get_in_memory_store
from src.core.security import verify_api_key

router = APIRouter(prefix="/documents", tags=["Document Management"])


# --- Schemas ---
class DocumentCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    content: str = Field(..., min_length=1, description="Raw text or markdown content")
    source: Optional[str] = Field(None, max_length=512, description="Source URL or file path")


class ChunkResponse(BaseModel):
    id: uuid.UUID
    chunk_index: int
    content: str
    metadata_json: Optional[dict] = None


class DocumentResponse(BaseModel):
    id: uuid.UUID
    title: str
    source: Optional[str]
    content_type: str
    chunks_count: int


class DocumentDetailResponse(DocumentResponse):
    chunks: List[ChunkResponse]


# --- Endpoints ---
@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    payload: DocumentCreateRequest,
    db: AsyncSession = Depends(get_db),
    embedder: BaseEmbeddingService = Depends(get_embedding_service),
    _: str = Depends(verify_api_key),
):
    """Ingests a text document: chunks it, generates embeddings, and saves to database or in-memory store."""
    from src.db import session as db_session

    if not db_session.is_db_connected:
        store = get_in_memory_store()
        doc = store.add_document(
            title=payload.title,
            source=payload.source,
            content_type="text/plain",
        )
        chunker = TextChunker()
        chunk_records = chunker.create_chunks_with_metadata(
            text=payload.content,
            base_metadata={"title": payload.title, "source": payload.source},
        )
        texts = [c["content"] for c in chunk_records]
        embeddings = await embedder.embed_documents(texts)
        store.add_chunks(doc["id"], payload.title, chunk_records, embeddings)
        return DocumentResponse(
            id=uuid.UUID(doc["id"]),
            title=doc["title"],
            source=doc["source"],
            content_type=doc["content_type"],
            chunks_count=len(chunk_records),
        )

    ingestion = IngestionService(embedding_service=embedder)
    doc = await ingestion.ingest_text(
        session=db,
        title=payload.title,
        text_content=payload.content,
        source=payload.source,
    )
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    chunks_res = await db.execute(stmt)
    chunks = chunks_res.scalars().all()

    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        source=doc.source,
        content_type=doc.content_type,
        chunks_count=len(chunks),
    )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    embedder: BaseEmbeddingService = Depends(get_embedding_service),
    _: str = Depends(verify_api_key),
):
    """Ingests a file (.txt, .md) directly via multipart form upload."""
    content_bytes = await file.read()
    try:
        text_content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be valid UTF-8 text.",
        )

    doc_title = title or file.filename or "Uploaded Document"
    from src.db import session as db_session

    if not db_session.is_db_connected:
        store = get_in_memory_store()
        doc = store.add_document(
            title=doc_title,
            source=file.filename,
            content_type=file.content_type or "text/plain",
        )
        chunker = TextChunker()
        chunk_records = chunker.create_chunks_with_metadata(
            text=text_content,
            base_metadata={"title": doc_title, "source": file.filename},
        )
        texts = [c["content"] for c in chunk_records]
        embeddings = await embedder.embed_documents(texts)
        store.add_chunks(doc["id"], doc_title, chunk_records, embeddings)
        return DocumentResponse(
            id=uuid.UUID(doc["id"]),
            title=doc["title"],
            source=doc["source"],
            content_type=doc["content_type"],
            chunks_count=len(chunk_records),
        )

    ingestion = IngestionService(embedding_service=embedder)
    doc = await ingestion.ingest_text(
        session=db,
        title=doc_title,
        text_content=text_content,
        source=file.filename,
        content_type=file.content_type or "text/plain",
    )

    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    chunks_res = await db.execute(stmt)
    chunks = chunks_res.scalars().all()

    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        source=doc.source,
        content_type=doc.content_type,
        chunks_count=len(chunks),
    )


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Lists all ingested documents with chunk counts."""
    from src.db import session as db_session

    if not db_session.is_db_connected:
        store = get_in_memory_store()
        docs = store.get_documents()
        return [
            DocumentResponse(
                id=uuid.UUID(d["id"]),
                title=d["title"],
                source=d.get("source"),
                content_type=d.get("content_type", "text/plain"),
                chunks_count=d.get("chunks_count", 0),
            )
            for d in docs[offset : offset + limit]
        ]

    stmt = (
        select(Document)
        .options(selectinload(Document.chunks))
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    docs = result.scalars().all()

    return [
        DocumentResponse(
            id=d.id,
            title=d.title,
            source=d.source,
            content_type=d.content_type,
            chunks_count=len(d.chunks),
        )
        for d in docs
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Deletes a document and its cascading chunks."""
    from src.db import session as db_session

    if not db_session.is_db_connected:
        store = get_in_memory_store()
        deleted = store.delete_document(str(document_id))
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
        return None

    stmt = select(Document).where(Document.id == document_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    await db.delete(doc)
    await db.commit()
    return None
