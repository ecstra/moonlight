import unittest, sys
import importlib.util
from pathlib import Path

# Get the parent directory (src/)
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

# Load modules directly without triggering __init__.py
# This avoids the relative import issues in main.py

# Load base.py module
base_spec = importlib.util.spec_from_file_location("agent.base", parent_path / "agent" / "base.py")
base_module = importlib.util.module_from_spec(base_spec)
sys.modules['agent.base'] = base_module
base_spec.loader.exec_module(base_module)
Content = base_module.Content

# Load history.py module
history_spec = importlib.util.spec_from_file_location("agent.history", parent_path / "agent" / "history.py")
history_module = importlib.util.module_from_spec(history_spec)
sys.modules['agent.history'] = history_module
history_spec.loader.exec_module(history_module)
AgentHistory = history_module.AgentHistory


class TestAgentHistory(unittest.IsolatedAsyncioTestCase):
    
    def test_initialization(self):
        """Test that AgentHistory initializes correctly with system role"""
        system_role = "You are a helpful assistant."
        history = AgentHistory(system_role=system_role)
        
        # Should have one message (system message)
        self.assertEqual(len(history.get_history()), 1)
        self.assertEqual(history.get_history()[0]["role"], "system")
        self.assertEqual(history.get_history()[0]["content"], system_role)
    
    async def test_add_text_only_message(self):
        """Test adding a simple text message without images"""
        history = AgentHistory(system_role="System")
        content = Content(text="Hello, world!")
        
        await history.add(role="user", content=content)
        
        # Should have 2 messages: system + user
        self.assertEqual(len(history.get_history()), 2)
        self.assertEqual(history.get_history()[1]["role"], "user")
        self.assertEqual(history.get_history()[1]["content"], "Hello, world!")
    
    async def test_add_multiple_messages(self):
        """Test adding multiple messages in sequence"""
        history = AgentHistory(system_role="System")
        
        await history.add(role="user", content=Content(text="Question"))
        await history.add(role="assistant", content=Content(text="Answer"))
        await history.add(role="user", content=Content(text="Follow-up"))
        
        # Should have 4 messages: system + 3 added
        self.assertEqual(len(history.get_history()), 4)
        self.assertEqual(history.get_history()[1]["content"], "Question")
        self.assertEqual(history.get_history()[2]["content"], "Answer")
        self.assertEqual(history.get_history()[3]["content"], "Follow-up")
    
    async def test_add_message_with_images(self):
        """Test adding a message with base64 images"""
        history = AgentHistory(system_role="System")
        
        # Use valid base64 data URL format
        image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        content = Content(text="What's in this image?", images=[image_url])
        
        await history.add(role="user", content=content)
        
        # Should have structured content with text and image_url
        message = history.get_history()[1]
        self.assertEqual(message["role"], "user")
        self.assertIsInstance(message["content"], list)
        self.assertEqual(len(message["content"]), 2)
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][0]["text"], "What's in this image?")
        self.assertEqual(message["content"][1]["type"], "image_url")
        self.assertEqual(message["content"][1]["image_url"]["url"], image_url)
    
    async def test_add_message_with_multiple_images(self):
        """Test adding a message with multiple base64 images"""
        history = AgentHistory(system_role="System")
        
        image1 = "data:image/png;base64,ABC123"
        image2 = "data:image/jpeg;base64,DEF456"
        content = Content(text="Compare these images", images=[image1, image2])
        
        await history.add(role="user", content=content)
        
        message = history.get_history()[1]
        self.assertEqual(len(message["content"]), 3)  # 1 text + 2 images
        self.assertEqual(message["content"][1]["image_url"]["url"], image1)
        self.assertEqual(message["content"][2]["image_url"]["url"], image2)
    
    async def test_update_system_role(self):
        """Test updating the system role"""
        history = AgentHistory(system_role="Old system role")
        await history.add(role="user", content=Content(text="User message"))
        
        new_system_role = "New system role"
        history.update_system_role(system_role=new_system_role)
        
        # System message should be updated, user message should remain
        self.assertEqual(len(history.get_history()), 2)
        self.assertEqual(history.get_history()[0]["role"], "system")
        self.assertEqual(history.get_history()[0]["content"], new_system_role)
        self.assertEqual(history.get_history()[1]["content"], "User message")
    
    async def test_clear_history(self):
        """Test clearing history"""
        history = AgentHistory(system_role="System")
        await history.add(role="user", content=Content(text="Message 1"))
        await history.add(role="assistant", content=Content(text="Message 2"))
        await history.add(role="user", content=Content(text="Message 3"))
        
        history.clear_history()
        
        # Should only have the system message
        self.assertEqual(len(history.get_history()), 1)
        self.assertEqual(history.get_history()[0]["role"], "system")
        self.assertEqual(history.get_history()[0]["content"], "System")
    
    async def test_repr(self):
        """Test __repr__ method"""
        history = AgentHistory(system_role="Test system")
        await history.add(role="user", content=Content(text="Test"))
        
        repr_str = repr(history)
        self.assertIn("AgentHistory", repr_str)
        self.assertIn("system_role='Test system'", repr_str)
        self.assertIn("messages=2", repr_str)
    
    async def test_str(self):
        """Test __str__ method returns history as string"""
        history = AgentHistory(system_role="System")
        await history.add(role="user", content=Content(text="Hello"))
        
        str_output = str(history)
        self.assertIsInstance(str_output, str)
        self.assertIn("system", str_output)
        self.assertIn("user", str_output)


if __name__ == "__main__":
    unittest.main()