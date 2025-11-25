import logging
import sys
from fastapi import FastAPI
from api.routes import config, configure, predict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAG-API")

app = FastAPI(
    title="RAG Backend API",
    description="API para configurar y utilizar los demostradores de RAG",
    version="1.0.0",
)

# Registrar routers
app.include_router(config.router)
app.include_router(configure.router)
app.include_router(predict.router)