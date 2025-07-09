"""
OpenAI Whisper + GPT-4o-mini Chat API - Railway Ready
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
app = FastAPI(title="Speech-to-Text + Chat API", version="1.0.0")

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
    conversation_history: list = []

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global whisper_model, openai_client
    
    print("🔄 Loading Whisper model...")
    whisper_model = whisper.load_model("tiny")
    print("✅ Whisper model loaded")
    
    # Initialize OpenAI client if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print("✅ OpenAI client initialized")
    else:
        print("⚠️ OpenAI API key not found - chat will use basic responses")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Speech-to-Text + Chat API Server",
        "status": "running",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "chat": "POST /chat",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "openai_configured": openai_client is not None,
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

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with OpenAI or basic responses"""
    try:
        message = request.message
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Use OpenAI if available
        if openai_client:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "Jsi AI asistent pro českou konverzaci. Odpovídej v češtině přirozeně a užitečně. Buď přátelský a nápomocný."
                    }
                ]
                
                # Add conversation history
                if request.conversation_history:
                    messages.extend(request.conversation_history)
                
                messages.append({
                    "role": "user",
                    "content": message
                })
                
                # Call OpenAI
                completion = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7
                )
                
                response = completion.choices[0].message.content
                
                return {
                    "response": response,
                    "timestamp": "2025-07-09",
                    "model": "gpt-4o-mini",
                    "tokens_used": completion.usage.total_tokens if completion.usage else 0
                }
                
            except Exception as openai_error:
                print(f"OpenAI error: {openai_error}")
                # Fall back to basic response
                pass
        
        # Basic response logic
        message_lower = message.lower()
        
        if "ahoj" in message_lower:
            response = "Ahoj! Jak se máš? Jsem AI asistent a jsem tu, abych ti pomohl."
        elif "jak se máš" in message_lower:
            response = "Mám se skvěle! Děkuji za optání. Jak můžu pomoci?"
        elif "pomoc" in message_lower:
            response = "Rád ti pomohu! Jakou pomoc potřebuješ?"
        elif "děkuji" in message_lower:
            response = "Není za co! Pokud budeš potřebovat další pomoc, jen se zeptej."
        elif "co umíš" in message_lower:
            response = "Umím převádět řeč na text a chatovat v češtině. Mohu ti pomoci s různými otázkami a úkoly."
        else:
            response = f'Rozumím tvé zprávě: "{message}". Jak ti mohu pomoci?'
        
        return {
            "response": response,
            "timestamp": "2025-07-09",
            "model": "basic-chat",
            "status": "working"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)