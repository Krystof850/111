"""
Combined OpenAI Whisper + GPT-4o Server
Speech-to-text + Real AI Chat in one server
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

# Initialize FastAPI app
app = FastAPI(title="Combined Whisper + GPT-4o API", version="1.0.0")

# Add CORS middleware
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
    goals: list = []

@app.on_event("startup")
async def startup():
    global whisper_model, openai_client
    
    # Load Whisper model
    print("🔄 Loading Whisper model...")
    whisper_model = whisper.load_model("tiny")
    print("✅ Whisper model loaded successfully")
    
    # Initialize OpenAI client
    if os.environ.get("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print("✅ OpenAI client initialized")
    else:
        print("⚠️ OpenAI API key not found - using fallback responses")

@app.get("/")
def root():
    return {
        "message": "Combined Whisper + GPT-4o API",
        "status": "running",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "chat": "POST /chat",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "whisper": whisper_model is not None,
        "openai": openai_client is not None,
        "timestamp": "2025-07-09"
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Speech-to-text using OpenAI Whisper"""
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

@app.post("/chat")
async def chat(request: ChatRequest):
    """Real AI chat using GPT-4o"""
    try:
        message = request.message
        
        # Try OpenAI GPT-4o first
        if openai_client:
            try:
                # the newest OpenAI model is "gpt-4o-mini" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Jsi AI asistent pro českou konverzaci. Odpovídej v češtině přirozeně a užitečně. Buď přátelský a nápomocný."
                        },
                        {
                            "role": "user",
                            "content": message
                        }
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                
                return {
                    "response": ai_response,
                    "timestamp": "2025-07-09",
                    "model": "gpt-4o-mini",
                    "tokens_used": response.usage.total_tokens,
                    "source": "openai"
                }
                
            except Exception as e:
                print(f"OpenAI error: {e}")
                # Fall through to fallback
        
        # Fallback Czech responses
        message_lower = message.lower()
        
        if "ahoj" in message_lower or "zdravím" in message_lower:
            response = "Ahoj! Jak se máš? Jsem AI asistent a jsem tu, abych ti pomohl."
        elif "jak se máš" in message_lower:
            response = "Mám se skvěle! Děkuji za optání. Jak můžu pomoci?"
        elif "pomoc" in message_lower:
            response = "Rád ti pomohu! Jakou pomoc potřebuješ?"
        elif "děkuji" in message_lower or "díky" in message_lower:
            response = "Není za co! Pokud budeš potřebovat další pomoc, jen se zeptej."
        elif "co umíš" in message_lower:
            response = "Jsem AI asistent a mohu ti pomoci s různými otázkami a úkoly. Můžeme si také jen popovídat v češtině."
        else:
            response = f"Rozumím tvé zprávě: '{message}'. Jak ti mohu pomoci?"
        
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