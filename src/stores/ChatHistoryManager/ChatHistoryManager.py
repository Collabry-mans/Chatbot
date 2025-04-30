import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from threading import Lock
from cryptography.fernet import Fernet
import uuid

class ChatHistoryManager:
    def __init__(
        self,
        storage_path: str = "chat_history.json",
        max_history_per_user: int = 100,
        encryption_key: Optional[bytes] = None,
        retention_days: int = 30
    ):
        """
        Args:
            storage_path: Path to JSON storage file
            max_history_per_user: Maximum messages stored per user
            encryption_key: Fernet key for encryption (None for plaintext)
            retention_days: Days to keep inactive conversations
        """
        self.storage_path = storage_path
        self.max_history = max_history_per_user
        self.retention_days = retention_days
        self.lock = Lock()
        
        # Encryption setup
        self.cipher = Fernet(encryption_key) if encryption_key else None
        self._initialize_storage()

    def _initialize_storage(self):
        """Create empty storage file if missing"""
        with self.lock:
            if not os.path.exists(self.storage_path):
                with open(self.storage_path, 'w') as f:
                    json.dump({"conversations": {}, "metadata": {"version": 1}}, f)

    def _encrypt(self, text: str) -> str:
        """Encrypt message content if cipher exists"""
        if not self.cipher:
            return text
        return self.cipher.encrypt(text.encode()).decode()

    def _decrypt(self, text: str) -> str:
        """Decrypt message content if cipher exists"""
        if not self.cipher:
            return text
        return self.cipher.decrypt(text.encode()).decode()

    def _read_data(self) -> Dict:
        """Thread-safe file read"""
        with self.lock:
            with open(self.storage_path, 'r') as f:
                return json.load(f)

    def _write_data(self, data: Dict):
        """Thread-safe file write"""
        with self.lock:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

    def add_message(
        self,
        user_id: str,
        message
    ) -> None:
        """Add a new message to history
        
        Args:
            user_id: Unique user/conversation identifier
            role: 'user' or 'bot'
            content: Message text
            metadata: Additional tracking data
        """
        data = self._read_data()
        
        if user_id not in data["conversations"]:
            data["conversations"][user_id] = {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "messages": []
            }

        data["conversations"][user_id]["messages"].append(message)
        data["conversations"][user_id]["updated_at"] = datetime.now().isoformat()
        
        # Enforce max history limit
        if len(data["conversations"][user_id]["messages"]) > self.max_history:
            data["conversations"][user_id]["messages"] = data["conversations"][user_id]["messages"][-self.max_history:]
        
        self._write_data(data)

    def get_conversation(
        self,
        user_id: str,
        max_messages: Optional[int] = 3,
        include_metadata: bool = False
    ) -> List[Dict]:
        """Retrieve conversation history
        
        Args:
            user_id: User identifier
            max_messages: None for all messages
            include_metadata: Whether to include message metadata
        """
        data = self._read_data()
        if user_id not in data["conversations"]:
            return []
        messages = data["conversations"][user_id]["messages"]
        if max_messages:
            messages = messages[-max_messages:]
 
        ans=[]
        for thread in messages:
            for msg in thread:
                x={
                    "role":msg["role"],
                    "content":msg["content"]
                }
                ans.append(x)
        return ans


    def cleanup_old_conversations(self):
        """Remove conversations older than retention_days"""
        threshold = datetime.now() - timedelta(days=self.retention_days)
        data = self._read_data()
        
        updated = False
        for user_id, conv in list(data["conversations"].items()):
            last_update = datetime.fromisoformat(conv["updated_at"])
            if last_update < threshold:
                del data["conversations"][user_id]
                updated = True
        
        if updated:
            self._write_data(data)

    def delete_conversation(self, user_id: str) -> bool:
        """Remove specific conversation"""
        data = self._read_data()
        if user_id in data["conversations"]:
            del data["conversations"][user_id]
            self._write_data(data)
            return True
        return False
    
