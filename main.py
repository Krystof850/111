"""
OpenAI Whisper FastAPI Backend - Railway Ready
"""

import os
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="OpenAI Whisper API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the model
whisper_model = None

@app.on_event("startup")
async def startup_event():
    """Load Whisper model on startup"""
    global whisper_model
    print("🔄 Loading Whisper model...")
    whisper_model = whisper.load_model("tiny")
    print("✅ Whisper model loaded successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenAI Whisper API Server",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper": whisper_model is not None,
        "timestamp": "2025-07-09"
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file using OpenAI Whisper"""
    try:
        if not whisper_model:
            raise HTTPException(status_code=500, detail="Whisper model not loaded")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Transcribe
        result = whisper_model.transcribe(tmp_file_path, language="cs")
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return {
            "transcript": result["text"],
            "language": "cs",
            "timestamp": "2025-07-09"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)