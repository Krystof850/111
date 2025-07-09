"""
OpenAI Service - Izolovaná služba pro GPT chat
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIService:
    """Izolovaná služba pro OpenAI GPT"""
    
    def __init__(self):
        self.client = None
        self.is_loaded = False
        self.model_name = "gpt-4o-mini"
        
    def load_client(self) -> bool:
        """Načte OpenAI client"""
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                logger.warning("⚠️ OpenAI API key not found")
                return False
            
            self.client = openai.OpenAI(api_key=api_key)
            self.is_loaded = True
            logger.info("✅ OpenAI client loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load OpenAI client: {e}")
            self.is_loaded = False
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Vrátí stav služby"""
        return {
            "service": "openai",
            "loaded": self.is_loaded,
            "model": self.model_name if self.is_loaded else None,
            "ready": self.is_loaded
        }
    
    def chat(self, message: str, goals: Optional[List[str]] = None) -> Dict[str, Any]:
        """Zpracuje chat zprávu"""
        if not self.is_loaded:
            # Fallback response
            return {
                "response": "OpenAI není dostupný. Nastavte OPENAI_API_KEY v Railway environment.",
                "timestamp": datetime.now().isoformat(),
                "model": "fallback",
                "tokens_used": 0,
                "source": "fallback"
            }
        
        try:
            # Připravit system prompt
            system_prompt = "Jsi užitečný AI asistent. Odpovídej v češtině a buď přátelský a nápomocný."
            
            if goals:
                system_prompt += f" Uživatel má tyto cíle: {', '.join(goals)}"
            
            # Vytvořit chat completion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return {
                "response": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "tokens_used": response.usage.total_tokens,
                "source": "openai"
            }
            
        except Exception as e:
            logger.error(f"❌ Chat error: {str(e)}")
            
            # Fallback response při chybě
            return {
                "response": f"Omlouváme se, došlo k chybě při komunikaci s OpenAI: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "model": "error",
                "tokens_used": 0,
                "source": "error"
            }

# Singleton instance
openai_service = OpenAIService()