"""
Health Service - Izolovaná služba pro monitoring
"""

import logging
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthService:
    """Izolovaná služba pro health monitoring"""
    
    def __init__(self):
        self.app_name = "Modular Speech-to-Text + Chat API"
        self.version = "4.0.0"
        self.environment = "Railway"
        
    def get_status(self, services: Dict[str, Any]) -> Dict[str, Any]:
        """Vrátí kompletní health status"""
        
        # Zjistit celkový stav
        all_services_ready = all(
            service.get("ready", False) for service in services.values()
        )
        
        overall_status = "healthy" if all_services_ready else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "app": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "services": services,
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
            "status": "healthy" if is_ready else "unavailable",
            "ready": is_ready,
            "timestamp": datetime.now().isoformat()
        }

# Singleton instance
health_service = HealthService()