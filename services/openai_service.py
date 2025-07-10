"""
OpenAI Service - Izolovaná služba pro GPT chat
"""

import os
import logging
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIService:
    """Izolovaná služba pro OpenAI GPT"""
    
    def __init__(self):
        self.client = None
        self.model = "gpt-4o-mini"
        self.is_loaded = False
        # Automaticky načíst client při inicializaci
        self.load_client()
        
    def load_client(self) -> bool:
        """Načte OpenAI client"""
        try:
            import openai
            
            # Získat API klíč
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("❌ OpenAI API key not found in environment")
                logger.error(f"Available env vars: {list(os.environ.keys())}")
                return False
            
            # Debugging - logovat první 10 znaků API klíče
            logger.info(f"🔑 OpenAI API key found: {api_key[:10]}...")
            
            self.client = openai.OpenAI(api_key=api_key)
            self.is_loaded = True
            logger.info(f"✅ OpenAI client loaded successfully")
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
            "model": self.model if self.is_loaded else None,
            "ready": self.is_loaded
        }
    
    def chat(self, message: str, goals: Optional[List[str]] = None) -> Dict[str, Any]:
        """Zpracuje chat zprávu"""
        if not self.is_loaded:
            raise Exception("OpenAI service not ready")
        
        try:
            # Základní systémová zpráva
            system_message = """Jsi inteligentní asistent který pomáhá s konverzací v češtině. 
            Odpovídej stručně, přátelsky a užitečně."""
            
            # Přidat cíle pokud jsou poskytnuty
            if goals and len(goals) > 0:
                goals_text = ", ".join(goals)
                system_message += f"\n\nUživatelské cíle: {goals_text}"
            
            # Vytvoř zprávy pro API
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ]
            
            # Volej OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            logger.info(f"✅ OpenAI chat successful: {len(ai_response)} characters")
            
            return {
                "response": ai_response,
                "model": self.model,
                "success": True,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI chat failed: {str(e)}")
            raise Exception(f"Chat failed: {str(e)}")

# Singleton instance
openai_service = OpenAIService()