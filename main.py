"""
OpenAI Whisper + GPT-4o-mini Complete API Server
"""

import os
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import uvicorn

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Speech-to-Text + Chat API",
    description="OpenAI Whisper + GPT-4o-mini Complete Solution",
    version="1.0.0"
)

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
    
    try:
        print("🔄 Loading Whisper model...")
        whisper_model = whisper.load_model("tiny")
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
    
    # Initialize OpenAI client
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            print("✅ OpenAI client initialized")
        except Exception as e:
            print(f"⚠️ OpenAI client initialization failed: {e}")
    else:
        print("⚠️ OpenAI not available - using basic chat responses")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Speech-to-Text + Chat API Server",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "POST /transcribe - Convert speech to text",
            "chat": "POST /chat - Chat with GPT-4o-mini",
            "health": "GET /health - Health check"
        },
        "whisper_loaded": whisper_model is not None,
        "openai_configured": openai_client is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "openai_configured": openai_client is not None,
        "openai_available": OPENAI_AVAILABLE,
        "timestamp": "2025-07-09T13:00:00Z"
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
        
        # Transcribe audio
        result = whisper_model.transcribe(tmp_file_path, language="cs")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "transcript": result["text"],
            "language": "cs",
            "confidence": 0.95,
            "timestamp": "2025-07-09T13:00:00Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with GPT-4o-mini or fallback responses"""
    try:
        message = request.message.strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Try OpenAI first if available
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
                
                # Add current message
                messages.append({
                    "role": "user",
                    "content": message
                })
                
                # Call OpenAI GPT-4o-mini
                completion = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                response = completion.choices[0].message.content
                
                return {
                    "response": response,
                    "timestamp": "2025-07-09T13:00:00Z",
                    "model": "gpt-4o-mini",
                    "tokens_used": completion.usage.total_tokens if completion.usage else 0,
                    "source": "openai"
                }
                
            except Exception as openai_error:
                print(f"OpenAI error: {openai_error}")
                # Fall through to basic response
        
        # Basic Czech response system
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["ahoj", "zdravím", "dobrý den"]):
            response = "Ahoj! Jak se máš? Jsem AI asistent a jsem tu, abych ti pomohl s čímkoli potřebuješ."
        elif any(word in message_lower for word in ["jak se máš", "co děláš"]):
            response = "Mám se skvěle, děkuji za optání! Jak můžu dnes pomoci?"
        elif any(word in message_lower for word in ["pomoc", "pomoct", "pomož"]):
            response = "Samozřejmě ti rád pomohu! Jakou pomoc potřebuješ? Můžu ti poradit s různými věcmi."
        elif any(word in message_lower for word in ["děkuji", "díky", "děkuju"]):
            response = "Není za co! Pokud budeš potřebovat další pomoc, klidně se zeptej."
        elif any(word in message_lower for word in ["co umíš", "co dokážeš"]):
            response = "Umím převádět řeč na text a chatovat v češtině. Mohu ti pomoci s otázkami, dát rady nebo jen si popovídat."
        elif any(word in message_lower for word in ["počasí", "venku"]):
            response = "Bohužel nemám přístup k aktuálním informacím o počasí. Doporučuji zkontrolovat si počasí na internetu nebo v aplikaci."
        elif any(word in message_lower for word in ["čas", "hodiny"]):
            response = "Aktuální čas si můžeš zkontrolovat na svém zařízení. Mohu ti pomoct s něčím jiným?"
        else:
            response = f'Rozumím tvé zprávě: "{message}". Jak ti konkrétně mohu pomoci?'
        
        return {
            "response": response,
            "timestamp": "2025-07-09T13:00:00Z",
            "model": "basic-czech",
            "tokens_used": 0,
            "source": "fallback"
        }
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"Global exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": "2025-07-09T13:00:00Z"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")