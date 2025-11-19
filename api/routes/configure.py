from fastapi import APIRouter
from models import ConfigureRequest, ConfigureResponse

router = APIRouter(prefix="", tags=["configure"])

@router.post("/configure", response_model=ConfigureResponse, summary="Determine appropriate language and domain")
async def configure_endpoint(request: ConfigureRequest):
    """
    Analyzes a user prompt (and optional language/domain hints) to determine the most suitable `(language, domain)` pair.
    """
    # Lógica real por implementar por HITZ
    return {"language": "unknown", "domain": "unknown"}