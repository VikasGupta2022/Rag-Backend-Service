from typing import Optional
from fastapi import Security
from fastapi.security.api_key import APIKeyHeader

from src.config import get_settings
from src.core.exceptions import UnauthorizedError

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Validates the X-API-Key header against settings.API_KEY.
    If settings.API_KEY is unset or empty, auth is bypassed (development mode).
    """
    settings = get_settings()
    if not settings.API_KEY:
        return "development-bypass"

    if not api_key or api_key != settings.API_KEY:
        raise UnauthorizedError()

    return api_key
