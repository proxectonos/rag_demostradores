import logging
from fastapi import APIRouter
from api.models import PredictRequest, PredictResponse, ContextItem
from demostrador.backend.rag import RAG

logger = logging.getLogger("RAG-API")

router = APIRouter(prefix="", tags=["predict"])

# Initialize RAG system once (adjust config path as needed)
rag_system = RAG(config_file="/home/compartido/pabloF/rag_demostradores/demostrador/backend/configs/general_config.json")

@router.post("/predict", response_model=PredictResponse, summary="Run RAG prediction")
async def predict_endpoint(request: PredictRequest):
    logger.info(f"Received /predict request: prompt='{request.prompt}', domain='{request.domain}', language='{request.language}', history_length={len(request.history)}")
    try:
        # Prepare chat history for RAG
        chat_history = [{"role": item.role, "content": item.content} for item in request.history]
        chat_history.append({"role": "user", "content": request.prompt})

        # Extract parameters from request
        model = None  # Use default or extract from request if needed
        domain = request.domain
        retrieval_method = None  # Use default or extract from request if needed
        top_k = 5  # You can adjust or extract from request

        # Call RAG backend
        chat_history, source_text, context_text = rag_system.generate_response(
            chat_history, model, domain, retrieval_method, top_k
        )

        # Parse context_text into ContextItem list
        contexts = []
        import re
        pattern = r"- \[(\d+)] \*\*(.*?)\*\* \(Source=(.*?), Pos=(.*?)\): (.*?)(?=\n- |\Z)"
        matches = re.findall(pattern, context_text, re.DOTALL)
        for num, title, source_id, pos, content in matches:
            contexts.append(ContextItem(
                id=num.strip(),
                title=title.strip(),
                passage=content.strip(),
                metadata={"source_id": source_id.strip(), "position": pos.strip()}
            ))

        # Get the assistant's response (last message in chat_history)
        response = chat_history[-1]["content"] if chat_history else ""

        logger.info(f"RAG response generated for prompt='{request.prompt}' with {len(contexts)} contexts.")
        return PredictResponse(response=response, contexts=contexts)
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}", exc_info=True)
        return PredictResponse(response="Error processing request.", contexts=[])