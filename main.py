"""
Modular FastAPI Application - Railway Production Ready
Skutečně modulární design s izolovanými službami
"""

import os
import logging
from typing import Optional, List
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import izolovaných služeb
from services.whisper_service import whisper_service
from services.openai_service import openai_service
from services.health_service import health_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Modular Speech-to-Text + Chat API",
    description="Skutečně modulární API s izolovanými službami",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    goals: Optional[List[str]] = []

@app.on_event("startup")
async def startup_event():
    """Načte všechny služby při startu"""
    logger.info("🚀 Starting modular application...")
    
    # Načíst Whisper službu (OpenAI API)
    whisper_service.load_model("api")
    
    # Načíst OpenAI službu
    openai_service.load_client()
    
    logger.info("✅ All services initialized")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Modular Speech-to-Text + Chat API",
        "version": "4.0.0",
        "status": "running",
        "architecture": "modular",
        "services": {
            "whisper": whisper_service.get_status(),
            "openai": openai_service.get_status()
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - agreguje stav všech služeb"""
    services = {
        "whisper": whisper_service.get_status(),
        "openai": openai_service.get_status()
    }
    
    return health_service.get_status(services)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Speech-to-text endpoint - používá izolovanou Whisper službu
    """
    try:
        # Kontrola souboru
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Přečíst obsah souboru
        content = await file.read()
        
        # Delegovat na Whisper službu
        result = whisper_service.transcribe(content, file.filename, language="cs")
        
        logger.info(f"✅ Transcription completed: {file.filename}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint - používá izolovanou OpenAI službu
    """
    try:
        # Delegovat na OpenAI službu
        result = openai_service.chat(request.message, request.goals)
        
        logger.info(f"✅ Chat completed: {len(request.message)} characters")
        return result
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/services")
async def services_status():
    """Endpoint pro monitoring jednotlivých služeb"""
    return {
        "whisper": whisper_service.get_status(),
        "openai": openai_service.get_status(),
        "health": health_service.get_status({
            "whisper": whisper_service.get_status(),
            "openai": openai_service.get_status()
        })
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)