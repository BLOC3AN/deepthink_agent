from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from .mock_tools import MCPToolsManager
from utils.logger import Logger

logger = Logger(__name__)

app = FastAPI(title="MCP Mock Server", version="1.0.0")
tools_manager = MCPToolsManager()


class ToolRequest(BaseModel):
    tool_name: str
    method: str
    parameters: Dict[str, Any] = {}


class ToolResponse(BaseModel):
    success: bool
    result: str
    error: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "MCP Mock Server is running", "status": "healthy"}


@app.get("/tools")
async def get_available_tools():
    """Get list of available tools"""
    return tools_manager.get_available_tools()


@app.post("/execute", response_model=ToolResponse)
async def execute_tool(request: ToolRequest):
    """Execute a tool with specified method and parameters"""
    try:
        logger.info(f"Executing tool: {request.tool_name}.{request.method}")
        
        result = await tools_manager.execute_tool(
            tool_name=request.tool_name,
            method=request.method,
            **request.parameters
        )
        
        return ToolResponse(success=True, result=result)
        
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "tools_available": len(tools_manager.get_available_tools()),
        "server": "MCP Mock Server v1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
