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
        self.model_type = "tiny"
        self.is_loaded = False
        
    def load_model(self, model_type: str = "tiny") -> bool:
        """Načte Whisper model"""
        try:
            import whisper
            logger.info(f"Loading Whisper model: {model_type}")
            self.model = whisper.load_model(model_type)
            self.model_type = model_type
            self.is_loaded = True
            logger.info(f"✅ Whisper model {model_type} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            self.is_loaded = False
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Vrátí stav služby"""
        return {
            "service": "whisper",
            "loaded": self.is_loaded,
            "model_type": self.model_type if self.is_loaded else None,
            "ready": self.is_loaded
        }
    
    def transcribe(self, file_content: bytes, filename: str, language: str = "cs") -> Dict[str, Any]:
        """Transkribuje audio soubor"""
        if not self.is_loaded:
            raise Exception("Whisper model not loaded")
        
        # Kontrola file extension
        allowed_extensions = {'.m4a', '.mp3', '.wav', '.webm', '.mp4'}
        file_extension = Path(filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise Exception(f"Unsupported file format: {file_extension}")
        
        # File size limit (25MB)
        if len(file_content) > 25 * 1024 * 1024:
            raise Exception("File too large. Maximum size: 25MB")
        
        try:
            # Vytvořit dočasný soubor
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Transkribovat pomocí Whisper
                result = self.model.transcribe(tmp_file_path, language=language)
                transcript = result["text"].strip()
                
                logger.info(f"✅ Transcription successful: {len(transcript)} characters")
                
                return {
                    "transcript": transcript,
                    "transcription": transcript,  # Backward compatibility
                    "text": transcript,          # Alternative format
                    "language": result.get("language", language),
                    "duration": result.get("duration", 0),
                    "file_size": len(file_content),
                    "success": True
                }
                
            finally:
                # Vyčistit dočasný soubor
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Transcription failed: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")

# Singleton instance
whisper_service = WhisperService()