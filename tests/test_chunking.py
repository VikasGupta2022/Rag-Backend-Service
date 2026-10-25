import pytest
from src.services.chunking import TextChunker


def test_chunker_empty_text():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.split_text("") == []
    assert chunker.split_text("   ") == []


def test_chunker_short_text():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "FastAPI is a modern, fast web framework for building APIs with Python."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunker_splits_paragraphs():
    chunker = TextChunker(chunk_size=80, chunk_overlap=15)
    text = (
        "Paragraph one discusses pgvector and PostgreSQL extensions.\n\n"
        "Paragraph two focuses on Reciprocal Rank Fusion algorithms.\n\n"
        "Paragraph three covers Server-Sent Events token streaming."
    )
    chunks = chunker.split_text(text)
    assert len(chunks) >= 3
    assert any("pgvector" in c for c in chunks)
    assert any("Reciprocal Rank Fusion" in c for c in chunks)


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError, match="strictly smaller"):
        TextChunker(chunk_size=50, chunk_overlap=50)


def test_chunk_metadata_attachment():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    results = chunker.create_chunks_with_metadata(text, base_metadata={"source": "test.txt"})
    assert len(results) > 0
    first = results[0]
    assert "chunk_index" in first
    assert "content" in first
    assert first["metadata"]["source"] == "test.txt"
    assert first["metadata"]["chunk_index"] == 0
