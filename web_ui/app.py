from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import json
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import DeepModelSystem
from web_ui.websocket_logger import websocket_log_handler, setup_websocket_logging
from utils.logger import Logger

# Setup WebSocket logging
setup_websocket_logging()

logger = Logger(__name__)

app = FastAPI(title="DeepModel Multi-Agent System", version="1.0.0")

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize DeepModel System
deepmodel_system = DeepModelSystem()


class ProcessRequest(BaseModel):
    user_input: str
    context: Dict[str, Any] = {}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main UI page"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepModel Multi-Agent System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
            min-height: 600px;
        }
        
        .input-section {
            padding: 30px;
            border-right: 1px solid #eee;
        }
        
        .logs-section {
            padding: 30px;
            background: #f8f9fa;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        textarea, input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        textarea:focus, input:focus {
            outline: none;
            border-color: #4facfe;
        }
        
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        .btn {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .logs-container {
            background: #1e1e1e;
            color: #fff;
            border-radius: 8px;
            padding: 20px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
        }
        
        .log-entry {
            margin-bottom: 5px;
            padding: 2px 0;
        }
        
        .log-info { color: #4facfe; }
        .log-warning { color: #ffa726; }
        .log-error { color: #ef5350; }
        .log-debug { color: #66bb6a; }
        
        .result-section {
            grid-column: 1 / -1;
            padding: 30px;
            border-top: 1px solid #eee;
            background: white;
        }
        
        .result-container {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            min-height: 200px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .status-indicator {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        
        .status-success { background: #d4edda; color: #155724; }
        .status-error { background: #f8d7da; color: #721c24; }
        .status-processing { background: #fff3cd; color: #856404; }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4facfe;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Markdown styling */
        .markdown-content {
            line-height: 1.6;
        }

        .markdown-content h1, .markdown-content h2, .markdown-content h3, .markdown-content h4 {
            color: #333;
            margin: 20px 0 10px 0;
            font-weight: 600;
        }

        .markdown-content h1 { font-size: 1.8em; border-bottom: 2px solid #4facfe; padding-bottom: 5px; }
        .markdown-content h2 { font-size: 1.5em; color: #4facfe; }
        .markdown-content h3 { font-size: 1.3em; color: #666; }
        .markdown-content h4 { font-size: 1.1em; color: #666; }

        .markdown-content ul, .markdown-content ol {
            margin: 10px 0;
            padding-left: 25px;
        }

        .markdown-content li {
            margin: 5px 0;
        }

        .markdown-content p {
            margin: 10px 0;
        }

        .markdown-content code {
            background: #f1f3f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .markdown-content pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #4facfe;
        }

        .markdown-content blockquote {
            border-left: 4px solid #ddd;
            margin: 15px 0;
            padding-left: 15px;
            color: #666;
            font-style: italic;
        }

        .task-details {
            background: white;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4facfe;
        }

        .task-details h5 {
            margin: 0 0 10px 0;
            color: #4facfe;
            font-size: 1.1em;
        }

        .execution-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }

        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e1e5e9;
        }

        .stat-value {
            font-size: 1.5em;
            font-weight: 600;
            color: #4facfe;
            display: block;
        }

        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 DeepModel Multi-Agent System</h1>
            <p>Intelligent multi-agent processing with real-time monitoring</p>
        </div>
        
        <div class="main-content">
            <div class="input-section">
                <h3>📝 Request Input</h3>
                <form id="requestForm">
                    <div class="form-group">
                        <label for="userInput">User Request:</label>
                        <textarea id="userInput" placeholder="Enter your request here... (e.g., 'Analyze market trends for AI technology and provide actionable insights')" required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="contextInput">Context (Optional):</label>
                        <input type="text" id="contextInput" placeholder='{"domain": "technology", "timeframe": "2024"}'>
                    </div>
                    
                    <button type="submit" class="btn" id="submitBtn">
                        🚀 Process Request
                    </button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing your request...</p>
                </div>
            </div>
            
            <div class="logs-section">
                <h3>📊 Real-time Logs</h3>
                <div class="logs-container" id="logsContainer">
                    <div class="log-entry log-info">System ready. Waiting for requests...</div>
                </div>
            </div>
        </div>
        
        <div class="result-section">
            <h3>📋 Results</h3>
            <div class="result-container" id="resultContainer">
                <p style="color: #666; text-align: center;">Results will appear here after processing...</p>
            </div>
        </div>
    </div>

    <!-- Markdown parsing library -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

    <script>
        let ws = null;
        let isProcessing = false;
        
        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = function() {
                addLog('Connected to real-time logging', 'info');
            };
            
            ws.onmessage = function(event) {
                const logData = JSON.parse(event.data);
                addLog(logData.formatted, logData.level.toLowerCase());
            };
            
            ws.onclose = function() {
                addLog('Disconnected from logging. Attempting to reconnect...', 'warning');
                setTimeout(initWebSocket, 3000);
            };
            
            ws.onerror = function() {
                addLog('WebSocket connection error', 'error');
            };
        }
        
        // Add log entry to the logs container
        function addLog(message, level = 'info') {
            const logsContainer = document.getElementById('logsContainer');
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${level}`;
            logEntry.textContent = message;
            
            logsContainer.appendChild(logEntry);
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
        
        // Handle form submission
        document.getElementById('requestForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (isProcessing) return;
            
            const userInput = document.getElementById('userInput').value.trim();
            const contextInput = document.getElementById('contextInput').value.trim();
            
            if (!userInput) {
                alert('Please enter a request');
                return;
            }
            
            // Parse context
            let context = {};
            if (contextInput) {
                try {
                    context = JSON.parse(contextInput);
                } catch (e) {
                    alert('Invalid JSON in context field');
                    return;
                }
            }
            
            // Start processing
            isProcessing = true;
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultContainer').innerHTML = '<div class="status-indicator status-processing">Processing...</div>';
            
            // Clear logs
            document.getElementById('logsContainer').innerHTML = '';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        user_input: userInput,
                        context: context
                    })
                });
                
                const result = await response.json();
                displayResult(result);
                
            } catch (error) {
                displayResult({
                    status: 'error',
                    error: error.message
                });
            } finally {
                isProcessing = false;
                document.getElementById('submitBtn').disabled = false;
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        // Display result
        function displayResult(result) {
            const container = document.getElementById('resultContainer');

            if (result.status === 'success') {
                // Create execution statistics
                const executionStats = `
                    <div class="execution-stats">
                        <div class="stat-item">
                            <span class="stat-value">${result.execution_result.total_execution_time.toFixed(2)}s</span>
                            <div class="stat-label">Total Time</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">${result.execution_result.task_count}</span>
                            <div class="stat-label">Tasks Executed</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">${result.planning_result.detected_intents.length}</span>
                            <div class="stat-label">Intents Detected</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">${result.execution_result.overall_status}</span>
                            <div class="stat-label">Status</div>
                        </div>
                    </div>
                `;

                // Create task details if available
                let taskDetails = '';
                if (result.execution_result && result.execution_result.task_results) {
                    taskDetails = '<h4>📋 Task Execution Details:</h4>';
                    result.execution_result.task_results.forEach((task, index) => {
                        taskDetails += `
                            <div class="task-details">
                                <h5>Task ${index + 1}: ${task.agent_type.charAt(0).toUpperCase() + task.agent_type.slice(1)} Agent</h5>
                                <p><strong>Agent:</strong> ${task.agent_name}</p>
                                <p><strong>Status:</strong> ${task.status}</p>
                                <p><strong>Execution Time:</strong> ${task.execution_time.toFixed(2)}s</p>
                                <p><strong>Result:</strong> ${task.result_summary}</p>
                                ${task.error_message ? `<p style="color: #ef5350;"><strong>Error:</strong> ${task.error_message}</p>` : ''}
                            </div>
                        `;
                    });
                }

                // Parse markdown for final output
                const markdownOutput = marked.parse(result.final_output || 'No output generated');

                container.innerHTML = `
                    <div class="status-indicator status-success">✅ Success</div>

                    <div style="margin: 20px 0;">
                        <h4>📊 Execution Summary</h4>
                        <p><strong>Session ID:</strong> ${result.session_id}</p>
                        <p><strong>Detected Intents:</strong> ${result.planning_result.detected_intents.join(', ')}</p>
                        <p><strong>Execution Strategy:</strong> ${result.planning_result.execution_strategy}</p>
                        <p><strong>Estimated Time:</strong> ${result.planning_result.estimated_time}s</p>
                    </div>

                    ${executionStats}

                    ${taskDetails}

                    <hr style="margin: 20px 0;">
                    <h4>📄 Final Output:</h4>
                    <div class="markdown-content" style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #e1e5e9;">
                        ${markdownOutput}
                    </div>
                `;
            } else {
                container.innerHTML = `
                    <div class="status-indicator status-error">❌ Error</div>
                    <p><strong>Error:</strong> ${result.error}</p>
                    <p><strong>Phase:</strong> ${result.phase || 'Unknown'}</p>
                `;
            }
        }
        
        // Initialize WebSocket when page loads
        window.addEventListener('load', initWebSocket);
    </script>
</body>
</html>
    """)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time logging"""
    await websocket.accept()
    websocket_log_handler.add_client(websocket)
    
    try:
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_log_handler.remove_client(websocket)


@app.post("/process")
async def process_request(request: ProcessRequest):
    """Process user request through the multi-agent system"""
    logger.info(f"🚀 New request received: {request.user_input[:100]}...")
    
    try:
        # Initialize system if needed
        await deepmodel_system.initialize()
        
        # Process the request
        result = await deepmodel_system.process_request(
            request.user_input, 
            request.context
        )
        
        logger.info(f"✅ Request completed with status: {result['status']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "phase": "system_error"
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "DeepModel Multi-Agent System"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
