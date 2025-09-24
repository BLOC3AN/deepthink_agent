import asyncio
import json
import logging
from typing import Set
from fastapi import WebSocket
from datetime import datetime


class WebSocketLogHandler(logging.Handler):
    """Custom log handler that sends logs to WebSocket clients"""
    
    def __init__(self):
        super().__init__()
        self.clients: Set[WebSocket] = set()
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def add_client(self, websocket: WebSocket):
        """Add a WebSocket client"""
        self.clients.add(websocket)
    
    def remove_client(self, websocket: WebSocket):
        """Remove a WebSocket client"""
        self.clients.discard(websocket)
    
    def emit(self, record):
        """Send log record to all connected WebSocket clients"""
        if not self.clients:
            return
        
        try:
            # Format the log message
            message = self.format(record)
            
            # Create log data for WebSocket
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "formatted": message
            }
            
            # Send to all connected clients
            asyncio.create_task(self._broadcast_log(log_data))
            
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Error in WebSocket log handler: {e}")
    
    async def _broadcast_log(self, log_data: dict):
        """Broadcast log data to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps(log_data)
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                await client.send_text(message)
            except Exception:
                # Client disconnected
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)


# Global WebSocket log handler instance
websocket_log_handler = WebSocketLogHandler()


def setup_websocket_logging():
    """Setup WebSocket logging for all loggers"""
    # Get root logger
    root_logger = logging.getLogger()
    
    # Add WebSocket handler to root logger
    root_logger.addHandler(websocket_log_handler)
    
    # Set level to INFO to capture all relevant logs
    websocket_log_handler.setLevel(logging.INFO)
    
    print("WebSocket logging setup completed")
