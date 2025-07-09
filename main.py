"""
OpenAI Whisper + GPT-4o-mini Combined Server
"""

import os
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import uvicorn
from openai import OpenAI

app = FastAPI(title="Whisper + GPT-4o-mini API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model = None
openai_client = None

class ChatRequest(BaseModel):
    message: str
    goals: list = []

@app.on_event("startup")
async def startup_event():
    global whisper_model, openai_client
    
    print("🔄 Loading Whisper model...")
    whisper_model = whisper.load_model("tiny")
    print("✅ Whisper model loaded")
    
    if os.environ.get("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print("✅ OpenAI client initialized")
    else:
        print("⚠️ OpenAI API key not found")

@app.get("/")
async def root():
    return {
        "message": "Whisper + GPT-4o-mini API",
        "status": "running",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "chat": "POST /chat",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper": whisper_model is not None,
        "openai": openai_client is not None
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        if not whisper_model:
            raise HTTPException(status_code=500, detail="Whisper model not loaded")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        result = whisper_model.transcribe(tmp_file_path, language="cs")
        os.unlink(tmp_file_path)
        
        return {
            "transcript": result["text"],
            "language": "cs",
            "timestamp": "2025-07-09"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        message = request.message
        
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Jsi AI asistent pro českou konverzaci. Odpovídej v češtině přirozeně a užitečně."
                        },
                        {
                            "role": "user",
                            "content": message
                        }
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "timestamp": "2025-07-09",
                    "model": "gpt-4o-mini",
                    "tokens_used": response.usage.total_tokens,
                    "source": "openai"
                }
                
            except Exception as e:
                print(f"OpenAI error: {e}")
        
        # Czech fallback
        message_lower = message.lower()
        
        if "ahoj" in message_lower:
            response = "Ahoj! Jak se máš? Jsem AI asistent."
        elif "jak se máš" in message_lower:
            response = "Mám se skvěle! Jak můžu pomoci?"
        elif "pomoc" in message_lower:
            response = "Rád ti pomohu! Co potřebuješ?"
        else:
            response = f"Rozumím: '{message}'. Jak ti mohu pomoci?"
        
        return {
            "response": response,
            "timestamp": "2025-07-09",
            "model": "basic-czech",
            "tokens_used": 0,
            "source": "fallback"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)