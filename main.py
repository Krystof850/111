"""
OpenAI Whisper + GPT-4o-mini FastAPI Backend - Railway Production Ready
"""

import os
import tempfile
import traceback
from datetime import datetime
from typing import Optional

import openai
import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# the newest OpenAI model is "gpt-4o-mini" which was released after knowledge cutoff
# do not change this unless explicitly requested by the user
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = FastAPI(title="Whisper + GPT-4o-mini API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
whisper_model = None
openai_client = None

class ChatRequest(BaseModel):
    message: str
    goals: Optional[list] = []

@app.on_event("startup")
async def startup_event():
    """Load Whisper model and OpenAI client on startup"""
    global whisper_model, openai_client
    
    try:
        # Load Whisper model
        whisper_model = whisper.load_model("tiny")
        print("✅ Whisper model loaded successfully")
        
        # Initialize OpenAI client
        if os.environ.get("OPENAI_API_KEY"):
            openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            print("✅ OpenAI client initialized successfully")
        else:
            print("⚠️ OpenAI API key not found")
            
    except Exception as e:
        print(f"❌ Startup error: {e}")
        traceback.print_exc()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Whisper + GPT-4o-mini API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper": whisper_model is not None,
        "openai": openai_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file using OpenAI Whisper"""
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Transcribe with Czech language
        result = whisper_model.transcribe(tmp_file_path, language="cs")
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return {
            "transcription": result["text"],
            "language": result.get("language", "cs"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Transcription error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat using OpenAI GPT-4o-mini"""
    if not openai_client:
        # Fallback response if OpenAI not available
        return {
            "response": "OpenAI není dostupný. Nastavte OPENAI_API_KEY v Railway environment.",
            "timestamp": datetime.now().isoformat(),
            "model": "fallback",
            "tokens_used": 0,
            "source": "fallback"
        }
    
    try:
        # Create chat completion with GPT-4o-mini
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Jsi užitečný AI asistent. Odpovídej v češtině a buď přátelský a nápomocný."
                },
                {
                    "role": "user", 
                    "content": request.message
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "tokens_used": response.usage.total_tokens,
            "source": "openai"
        }
        
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        
        # Fallback response on error
        return {
            "response": f"Omlouváme se, došlo k chybě při komunikaci s OpenAI: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "model": "error",
            "tokens_used": 0,
            "source": "error"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))