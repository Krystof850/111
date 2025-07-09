import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global whisper model
model = None

@app.on_event("startup")
async def startup():
    global model
    try:
        import whisper
        model = whisper.load_model("tiny")
        print("✅ Whisper loaded")
    except Exception as e:
        print(f"❌ Whisper load error: {e}")

@app.get("/")
def root():
    return {"message": "Simple Whisper API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "whisper": model is not None}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    if not file.filename:
        raise HTTPException(400, "No file")
    
    try:
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        result = model.transcribe(tmp_path)
        
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return {"transcript": result["text"].strip()}
        
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)