"""
OpenAI Whisper FastAPI Backend - Railway Production
"""

import os
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenAI Whisper API", version="2.0.0")

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
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        whisper_model = None

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "OpenAI Whisper API Server",
        "version": "2.0.0",
        "environment": "Railway Production",
        "status": "running",
        "whisper_status": {
            "loaded": whisper_model is not None,
            "loading": False,
            "error": None if whisper_model else "Model loading failed"
        },
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "whisper-api",
        "environment": "Railway",
        "whisper_model": "loaded" if whisper_model is not None else "loading...",
        "model_error": None,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "ready_for_transcription": whisper_model is not None
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text using OpenAI Whisper
    
    Accepts: .m4a, .mp3, .wav, .webm files
    Returns: {"transcript": "transcribed text"}
    """
    global whisper_model
    
    if whisper_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Whisper model is still loading. Please try again in a moment."
        )
    
    # Check file type
    allowed_types = ['.m4a', '.mp3', '.wav', '.webm', '.mp4']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Supported: {allowed_types}"
        )
    
    # Process with OpenAI Whisper
    try:
        content = await file.read()
        file_size = len(content)
        
        # File size limit (25MB)
        if file_size > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size is 25MB"
            )
        
        logger.info(f"Transcribing audio: {file.filename}, size: {file_size} bytes")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Transcribe using Whisper
        result = whisper_model.transcribe(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        transcript = result["text"].strip()
        logger.info(f"Transcription successful: {len(transcript)} characters")
        
        return {
            "transcript": transcript,
            "success": True,
            "file_size": file_size,
            "duration": result.get("duration", 0)
        }
        
    except Exception as e:
        # Clean up on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")