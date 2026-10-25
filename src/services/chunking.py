import re
from typing import List, Dict, Any


class TextChunker:
    """
    Splits text recursively by paragraphs, sentences, and words to maintain semantic context.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Splits input string into overlapping chunks."""
        text = text.strip()
        if not text:
            return []

        # If text is already smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Split into semantic segments (double newlines -> paragraphs, single newlines, or sentence periods)
        paragraphs = re.split(r"(\n\s*\n|\.\s+)", text)
        
        segments: List[str] = []
        for i in range(0, len(paragraphs), 2):
            seg = paragraphs[i]
            delim = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
            full_seg = (seg + delim).strip()
            if full_seg:
                segments.append(full_seg)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for segment in segments:
            seg_len = len(segment)

            # If a single segment is itself longer than chunk_size, split by words
            if seg_len > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                words = segment.split()
                w_chunk: List[str] = []
                w_len = 0
                for w in words:
                    if w_len + len(w) + 1 > self.chunk_size and w_chunk:
                        chunks.append(" ".join(w_chunk))
                        # retain overlap words
                        overlap_words = []
                        ov_len = 0
                        for ow in reversed(w_chunk):
                            if ov_len + len(ow) + 1 <= self.chunk_overlap:
                                overlap_words.insert(0, ow)
                                ov_len += len(ow) + 1
                            else:
                                break
                        w_chunk = overlap_words
                        w_len = ov_len
                    w_chunk.append(w)
                    w_len += len(w) + 1
                if w_chunk:
                    chunks.append(" ".join(w_chunk))
                continue

            # Standard accumulation
            if current_length + seg_len + 1 > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Retain overlap from end of current chunk
                overlap_text = ""
                for prev in reversed(current_chunk):
                    if len(overlap_text) + len(prev) + 1 <= self.chunk_overlap:
                        overlap_text = prev + " " + overlap_text
                    else:
                        break
                current_chunk = [overlap_text.strip()] if overlap_text.strip() else []
                current_length = len(current_chunk[0]) if current_chunk else 0

            current_chunk.append(segment)
            current_length += seg_len + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return [c for c in chunks if c]

    def create_chunks_with_metadata(
        self, text: str, base_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Splits text and associates chunk index, start char, and metadata."""
        raw_chunks = self.split_text(text)
        metadata = base_metadata or {}
        results = []

        for idx, content in enumerate(raw_chunks):
            results.append({
                "chunk_index": idx,
                "content": content,
                "metadata": {
                    **metadata,
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                    "char_count": len(content),
                }
            })
        return results
