from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from typing import List, Dict, Optional
import os
from utils.logger import Logger

logger = Logger(__name__)


class MongoDBClient:
    def __init__(self):
        self.client = None
        self.db = None
        self.connection_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.database_name = os.getenv("DATABASE_NAME", "deepmodel_agents")
    
    async def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            # Test connection
            await self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"Connected to MongoDB: {self.database_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    async def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def get_all_agent_cards(self) -> List[Dict]:
        """Get all active agent cards from database"""
        try:
            collection = self.db.agent_cards
            cursor = collection.find({"status": "active"})
            agents = await cursor.to_list(length=None)
            
            # Convert ObjectId to string for JSON serialization
            for agent in agents:
                agent["_id"] = str(agent["_id"])
            
            logger.info(f"Retrieved {len(agents)} active agent cards")
            return agents
        except Exception as e:
            logger.error(f"Error retrieving agent cards: {e}")
            return []
    
    async def create_session(self, session_data: Dict) -> str:
        """Create new session record"""
        try:
            collection = self.db.sessions
            result = await collection.insert_one(session_data)
            session_id = str(result.inserted_id)
            logger.info(f"Created session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return ""
    
    async def create_task(self, task_data: Dict) -> str:
        """Create new task record"""
        try:
            collection = self.db.tasks
            result = await collection.insert_one(task_data)
            task_id = str(result.inserted_id)
            logger.info(f"Created task: {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return ""
    
    async def update_task_status(self, task_id: str, status: str, result_data: Optional[Dict] = None):
        """Update task status and result"""
        try:
            collection = self.db.tasks
            update_data = {"status": status}
            if result_data:
                update_data["output_data"] = result_data
                update_data["completed_at"] = {"$currentDate": True}
            
            await collection.update_one(
                {"task_id": task_id},
                {"$set": update_data}
            )
            logger.info(f"Updated task {task_id} status to {status}")
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
