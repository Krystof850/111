"""
OpenAI Whisper FastAPI Backend - Railway Production v2.1
"""

import os
import tempfile
import logging
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenAI Whisper API", 
    version="2.1.0",
    description="Production-ready OpenAI Whisper API for speech-to-text conversion"
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
model_loading_error = None

@app.on_event("startup")
async def startup_event():
    """Load Whisper model on startup"""
    global whisper_model, model_loading_error
    try:
        logger.info("Starting OpenAI Whisper model loading...")
        import whisper
        
        # Load base model (good balance of speed and accuracy)
        whisper_model = whisper.load_model("base")
        logger.info("✅ Whisper model loaded successfully!")
        
    except Exception as e:
        model_loading_error = str(e)
        logger.error(f"❌ Failed to load Whisper model: {e}")
        whisper_model = None

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "OpenAI Whisper API Server",
        "version": "2.1.0",
        "environment": "Railway Production", 
        "status": "running",
        "whisper_status": {
            "loaded": whisper_model is not None,
            "loading": False,
            "error": model_loading_error
        },
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)"
        },
        "supported_formats": [".m4a", ".mp3", ".wav", ".webm", ".mp4"],
        "max_file_size": "25MB"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if whisper_model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "whisper-api",
                "whisper_model": "not_loaded",
                "error": model_loading_error,
                "ready_for_transcription": False
            }
        )
    
    return {
        "status": "healthy",
        "service": "whisper-api",
        "environment": "Railway",
        "whisper_model": "loaded",
        "model_error": None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "ready_for_transcription": True
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text using OpenAI Whisper
    
    Accepts: .m4a, .mp3, .wav, .webm, .mp4 files
    Returns: {"transcript": "transcribed text"}
    """
    if whisper_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Whisper model not loaded. Please check /health endpoint for details."
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file extension
    allowed_extensions = {'.m4a', '.mp3', '.wav', '.webm', '.mp4', '.ogg'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_extension}. Supported: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # File size limit (25MB)
        if file_size > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size is 25MB"
            )
        
        logger.info(f"Processing audio: {file.filename} ({file_size/1024/1024:.1f}MB)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Transcribe using Whisper
        logger.info("Starting transcription...")
        result = whisper_model.transcribe(tmp_file_path, language="auto")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        transcript = result["text"].strip()
        detected_language = result.get("language", "unknown")
        
        logger.info(f"Transcription completed: {len(transcript)} characters, language: {detected_language}")
        
        return {
            "transcript": transcript,
            "success": True,
            "file_size": file_size,
            "duration": result.get("duration", 0),
            "language": detected_language
        }
        
    except Exception as e:
        # Clean up temporary file on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@app.get("/status")
def get_status():
    """Detailed status endpoint"""
    return {
        "server": "running",
        "whisper_model": {
            "loaded": whisper_model is not None,
            "type": "base",
            "error": model_loading_error
        },
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "environment": "Railway"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        access_log=True
    )