from fastapi import HTTPException, status


class RAGException(Exception):
    """Base exception for RAG service."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class IngestionError(RAGException):
    """Raised when document ingestion or chunking fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class LLMGenerationError(RAGException):
    """Raised when LLM completion fails."""
    pass


class DocumentNotFoundError(HTTPException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' was not found.",
        )


class UnauthorizedError(HTTPException):
    def __init__(self, detail: str = "Invalid or missing API key."):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "ApiKey"},
        )
