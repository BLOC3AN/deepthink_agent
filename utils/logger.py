import logging
import json
from typing import Dict, Any, Optional

class Logger:
    def __init__(self, name, log_file="app.log"): # Thêm tham số log_file để chỉ định tên file log
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Xóa các handler hiện có để tránh log bị lặp lại nếu bạn khởi tạo lại Logger
        if not self.logger.handlers:
            # Ghi log ra console
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            # Ghi log ra file
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info("✅ "+message)
    def error(self, message):
        self.logger.error("❌ "+message)
    def debug(self, message):
        self.logger.debug("🔥 "+message)
    def warning(self, message):
        self.logger.warning("⚠️ "+message)

    # Enhanced logging methods for API calls and operations
    def log_api_call(self, url: str, method: str, payload: Optional[Dict[str, Any]] = None):
        """Log API call details"""
        self.info(f"🚀 API Call - {method} {url}")
        if payload:
            payload_str = json.dumps(payload, ensure_ascii=False)
            self.debug(f"📦 Request payload: {payload_str}")

    def log_api_response(self, status_code: int, response_body: str, success: bool = True):
        """Log API response details"""
        if success and status_code == 200:
            self.info(f"📡 API Response - Status: {status_code}")
        else:
            self.error(f"📡 API Response - Status: {status_code}")

        if response_body:
            body_str = response_body[:500] + "..." if len(response_body) > 500 else response_body
            self.debug(f"📄 Response body: {body_str}")

    def log_operation_result(self, operation: str, success: bool, details: Any = None, level: str = "info"):
        """Log operation result with structured format - simplified"""
        if level == "debug":
            # Only log in debug mode to reduce verbosity
            if success:
                self.debug(f"✅ {operation} OK")
                if details:
                    details_str = str(details)[:150] + "..." if len(str(details)) > 150 else str(details)
                    self.debug(f"Details: {details_str}")
            else:
                self.debug(f"❌ {operation} failed")
                if details:
                    details_str = str(details)[:300] + "..." if len(str(details)) > 300 else str(details)
                    self.debug(f"Error: {details_str}")
        elif not success:
            # Always log failures in error level
            self.error(f"❌ {operation} failed")
            if details:
                details_str = str(details)[:300] + "..." if len(str(details)) > 300 else str(details)
                self.error(f"Error: {details_str}")

    def log_tool_execution(self, tool_name: str, input_data: Any, result: Any = None, level: str = "info"):
        """Log tool execution with input and output - simplified"""
        if level == "debug":
            self.debug(f"🔧 Tool '{tool_name}' called")

            # Only log input data in debug mode and if it's meaningful
            if input_data and len(str(input_data)) > 10:
                input_str = json.dumps(input_data, ensure_ascii=False) if isinstance(input_data, dict) else str(input_data)
                if len(input_str) > 150:
                    input_str = input_str[:150] + "..."
                self.debug(f"📥 Input: {input_str}")
        else:
            # Only log tool name in info mode
            self.info(f"🔧 {tool_name}")

        # Only log result if explicitly provided and in debug mode
        if result and level == "debug":
            result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            self.debug(f"📤 Result: {result_str}")

    def log_exception(self, operation: str, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """Log exception with context"""
        self.error(f"💥 Exception in {operation}: {str(exception)}")
        self.debug(f"🔍 Exception type: {type(exception).__name__}")
        if context:
            context_str = json.dumps(context, ensure_ascii=False)[:300]
            self.debug(f"🔍 Exception context: {context_str}")

        # Log stack trace in debug mode
        import traceback
        self.debug(f"📚 Stack trace: {traceback.format_exc()}")
