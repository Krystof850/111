"""
Health Service - Izolovaná služba pro monitoring
"""

import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthService:
    """Izolovaná služba pro health monitoring"""
    
    def __init__(self):
        self.start_time = datetime.now()
        
    def get_status(self, services: Dict[str, Any]) -> Dict[str, Any]:
        """Vrátí kompletní health status"""
        
        # Zkontroluj stav všech služeb
        all_ready = True
        service_statuses = {}
        
        for service_name, service_data in services.items():
            is_ready = service_data.get("ready", False)
            service_statuses[service_name] = self.get_service_health(service_name, is_ready)
            
            if not is_ready:
                all_ready = False
        
        # Určit celkový stav
        if all_ready:
            status = "healthy"
        else:
            status = "degraded"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "app": "Modular Speech-to-Text + Chat API",
            "version": "4.0.0",
            "environment": "Railway",
            "services": service_statuses,
            "endpoints": {
                "health": "/health",
                "transcribe": "/transcribe",
                "chat": "/chat",
                "docs": "/docs"
            }
        }
    
    def get_service_health(self, service_name: str, is_ready: bool) -> Dict[str, Any]:
        """Vrátí health status konkrétní služby"""
        return {
            "service": service_name,
            "ready": is_ready,
            "status": "healthy" if is_ready else "degraded",
            "last_check": datetime.now().isoformat()
        }

# Singleton instance
health_service = HealthService()