from typing import Dict, Any
from utils.logger import Logger

logger = Logger(__name__)


class MockSQLTool:
    """Mock SQL tool that returns text responses"""
    
    def __init__(self):
        self.tool_name = "SQL Tool"
    
    async def execute_query(self, query: str, params: Dict = None) -> str:
        """Mock SQL query execution"""
        params_str = f", params: {params}" if params else ""
        response = f"{self.tool_name} được gọi với query: '{query}'{params_str}"
        logger.info(f"SQL Tool executed: {query}")
        return response


class MockWebSearchTool:
    """Mock web search tool that returns text responses"""
    
    def __init__(self):
        self.tool_name = "Web Search Tool"
    
    async def search(self, query: str, limit: int = 10) -> str:
        """Mock web search"""
        response = f"{self.tool_name} được gọi với query: '{query}', limit: {limit}"
        logger.info(f"Web Search Tool executed: {query}")
        return response


class MockRAGTool:
    """Mock RAG tool that returns text responses"""
    
    def __init__(self):
        self.tool_name = "RAG Tool"
    
    async def retrieve_and_generate(self, query: str, top_k: int = 5) -> str:
        """Mock RAG retrieval and generation"""
        response = f"{self.tool_name} được gọi với query: '{query}', top_k: {top_k}"
        logger.info(f"RAG Tool executed: {query}")
        return response


class MCPToolsManager:
    """Manager for all MCP mock tools"""
    
    def __init__(self):
        self.sql_tool = MockSQLTool()
        self.websearch_tool = MockWebSearchTool()
        self.rag_tool = MockRAGTool()
        logger.info("MCP Tools Manager initialized")
    
    async def execute_tool(self, tool_name: str, method: str, **kwargs) -> str:
        """Execute specified tool with method and parameters"""
        try:
            if tool_name == "sql":
                if method == "execute_query":
                    return await self.sql_tool.execute_query(
                        kwargs.get("query", ""), 
                        kwargs.get("params")
                    )
            elif tool_name == "websearch":
                if method == "search":
                    return await self.websearch_tool.search(
                        kwargs.get("query", ""),
                        kwargs.get("limit", 10)
                    )
            elif tool_name == "rag":
                if method == "retrieve_and_generate":
                    return await self.rag_tool.retrieve_and_generate(
                        kwargs.get("query", ""),
                        kwargs.get("top_k", 5)
                    )
            
            # Default response for unknown tools/methods
            params_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
            return f"Unknown tool '{tool_name}' method '{method}' được gọi với params: {params_str}"
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}.{method}: {e}")
            return f"Error executing {tool_name}.{method}: {str(e)}"
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get list of available tools and their methods"""
        return {
            "sql": {
                "methods": ["execute_query"],
                "description": "Execute SQL queries"
            },
            "websearch": {
                "methods": ["search"],
                "description": "Search the web"
            },
            "rag": {
                "methods": ["retrieve_and_generate"],
                "description": "Retrieve and generate using RAG"
            }
        }
