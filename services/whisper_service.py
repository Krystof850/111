"""
Whisper Service - Izolovaná služba pro speech-to-text
"""

import os
import tempfile
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperService:
    """Izolovaná služba pro OpenAI Whisper"""
    
    def __init__(self):
        self.model = None
        self.model_type = "api"
        self.is_loaded = False
        
    def load_model(self, model_type: str = "api") -> bool:
        """Načte Whisper model - používá OpenAI API pro úsporu paměti"""
        try:
            # Nepoužívat lokální model - používat OpenAI API
            self.model = "openai-api"
            self.model_type = "api"
            self.is_loaded = True
            logger.info(f"✅ Whisper service ready - using OpenAI API")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper service: {e}")
            self.is_loaded = False
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Vrátí stav služby"""
        return {
            "service": "whisper",
            "loaded": self.is_loaded,
            "model_type": "openai-api" if self.is_loaded else None,
            "ready": self.is_loaded,
            "memory_optimized": True
        }
    
    def transcribe(self, file_content: bytes, filename: str, language: str = "cs") -> Dict[str, Any]:
        """Transkribuje audio soubor pomocí OpenAI API"""
        if not self.is_loaded:
            raise Exception("Whisper service not ready")
        
        # Kontrola file extension
        allowed_extensions = {'.m4a', '.mp3', '.wav', '.webm', '.mp4'}
        file_extension = Path(filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise Exception(f"Unsupported file format: {file_extension}")
        
        # File size limit (25MB)
        if len(file_content) > 25 * 1024 * 1024:
            raise Exception("File too large. Maximum size: 25MB")
        
        try:
            # Použít OpenAI API místo lokálního modelu
            import openai
            
            # Získat API klíč
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OpenAI API key not configured")
            
            client = openai.OpenAI(api_key=api_key)
            
            # Vytvořit dočasný soubor
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Transkribovat pomocí OpenAI API
                with open(tmp_file_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language if language != "cs" else "czech"
                    )
                
                transcript = response.text.strip()
                
                logger.info(f"✅ OpenAI API transcription successful: {len(transcript)} characters")
                
                return {
                    "transcript": transcript,
                    "transcription": transcript,  # Backward compatibility
                    "text": transcript,          # Alternative format
                    "language": language,
                    "duration": 0,  # OpenAI API doesn't provide duration
                    "file_size": len(file_content),
                    "success": True,
                    "source": "openai-api"
                }
                
            finally:
                # Vyčistit dočasný soubor
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ OpenAI API transcription failed: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")

# Singleton instance
whisper_service = WhisperService()