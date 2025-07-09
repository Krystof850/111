"""
OpenAI Whisper FastAPI Backend - Railway Production
"""

import os
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenAI Whisper API",
    description="Speech-to-text API using OpenAI Whisper",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global whisper model
whisper_model = None

@app.on_event("startup")
async def startup_event():
    """Load Whisper model on startup"""
    global whisper_model
    try:
        import whisper
        logger.info("🚀 Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        logger.info("✅ Whisper model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
        whisper_model = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenAI Whisper API - Railway Production",
        "version": "3.0.0",
        "status": "running",
        "environment": "Railway",
        "whisper_loaded": whisper_model is not None,
        "model_type": "base",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if whisper_model else "loading",
        "service": "whisper-api",
        "version": "3.0.0",
        "environment": "Railway",
        "whisper_model": "loaded" if whisper_model else "loading...",
        "ready_for_transcription": whisper_model is not None
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file using OpenAI Whisper"""
    if whisper_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Whisper model not loaded yet - please wait and try again"
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file extension
    allowed_extensions = {'.m4a', '.mp3', '.wav', '.webm', '.mp4'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_extension}. Supported: {list(allowed_extensions)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # File size limit (25MB)
        if file_size > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail="File too large. Maximum size: 25MB"
            )
        
        logger.info(f"🎙️ Processing audio: {file.filename} ({file_size} bytes)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Transcribe using Whisper
            result = whisper_model.transcribe(tmp_file_path)
            transcript = result["text"].strip()
            
            logger.info(f"✅ Transcription successful: {len(transcript)} characters")
            
            return {
                "transcript": transcript,
                "success": True,
                "file_size": file_size,
                "language": result.get("language", "unknown"),
                "duration": result.get("duration", 0)
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Transcription failed: {str(e)}"
        )

if __name__ == "__main__":
    # Railway poskytuje PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Whisper API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
