from datetime import datetime
from typing import Any

import pytest

from src.deriver.queue_payload import DeriverQueuePayload


class TestDeriverQueuePayloadCreatePayload:
    """Test suite for DeriverQueuePayload.create_payload method"""

    def create_valid_message(self, **overrides: Any) -> dict[str, Any]:
        """Create a valid message dictionary for testing"""
        default_message = {
            "content": "Test message content",
            "workspace_name": "test_workspace",
            "session_name": "test_session",
            "message_id": 123,
            "created_at": datetime.now(),
        }
        default_message.update(overrides)
        return default_message

    def test_workspace_name_validation_error_for_none(self):
        """Test that workspace_name=None raises TypeError - covers line 55"""
        message = self.create_valid_message(workspace_name=None)
        
        with pytest.raises(TypeError, match="Workspace name must be a string"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target", 
                task_type="representation"
            )

    def test_workspace_name_validation_error_for_int(self):
        """Test that workspace_name as integer raises TypeError - covers line 55"""
        message = self.create_valid_message(workspace_name=123)
        
        with pytest.raises(TypeError, match="Workspace name must be a string"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_workspace_name_validation_error_for_bool(self):
        """Test that workspace_name as boolean raises TypeError - covers line 55"""
        message = self.create_valid_message(workspace_name=True)
        
        with pytest.raises(TypeError, match="Workspace name must be a string"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_workspace_name_validation_error_for_list(self):
        """Test that workspace_name as list raises TypeError - covers line 55"""
        message = self.create_valid_message(workspace_name=["workspace"])
        
        with pytest.raises(TypeError, match="Workspace name must be a string"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_workspace_name_validation_error_for_dict(self):
        """Test that workspace_name as dict raises TypeError - covers line 55"""
        message = self.create_valid_message(workspace_name={"name": "workspace"})
        
        with pytest.raises(TypeError, match="Workspace name must be a string"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_workspace_name_validation_error_for_float(self):
        """Test that workspace_name as float raises TypeError - covers line 55"""
        message = self.create_valid_message(workspace_name=12.34)
        
        with pytest.raises(TypeError, match="Workspace name must be a string"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_workspace_name_validation_error_missing_key(self):
        """Test that missing workspace_name key raises TypeError - covers line 55"""
        message = self.create_valid_message()
        del message["workspace_name"]  # Remove the key entirely
        
        with pytest.raises(TypeError, match="Workspace name must be a string"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_valid_workspace_name_string_passes(self):
        """Test that valid string workspace_name passes validation"""
        message = self.create_valid_message(workspace_name="valid_workspace")
        
        # Should not raise TypeError
        result = DeriverQueuePayload.create_payload(
            message=message,
            sender_name="sender",
            target_name="target",
            task_type="representation"
        )
        
        assert result["workspace_name"] == "valid_workspace"
        assert isinstance(result, dict)

    def test_empty_string_workspace_name_passes(self):
        """Test that empty string workspace_name passes isinstance check but may fail later validation"""
        message = self.create_valid_message(workspace_name="")
        
        # Should not raise TypeError on line 55 (passes isinstance check)
        # May fail later in Pydantic validation, but that's not line 55
        result = DeriverQueuePayload.create_payload(
            message=message,
            sender_name="sender", 
            target_name="target",
            task_type="representation"
        )
        
        assert result["workspace_name"] == ""
        assert isinstance(result, dict)

    def test_message_id_validation_error_for_none(self):
        """Test that message_id=None raises TypeError - covers line 60"""
        message = self.create_valid_message(message_id=None)
        
        with pytest.raises(TypeError, match="Message ID must be an integer"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target", 
                task_type="representation"
            )

    def test_message_id_validation_error_for_string(self):
        """Test that message_id as string raises TypeError - covers line 60"""
        message = self.create_valid_message(message_id="123")
        
        with pytest.raises(TypeError, match="Message ID must be an integer"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_message_id_validation_error_for_float(self):
        """Test that message_id as float raises TypeError - covers line 60"""
        message = self.create_valid_message(message_id=123.45)
        
        with pytest.raises(TypeError, match="Message ID must be an integer"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )


    def test_message_id_validation_error_for_list(self):
        """Test that message_id as list raises TypeError - covers line 60"""
        message = self.create_valid_message(message_id=[123])
        
        with pytest.raises(TypeError, match="Message ID must be an integer"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_message_id_validation_error_for_dict(self):
        """Test that message_id as dict raises TypeError - covers line 60"""
        message = self.create_valid_message(message_id={"id": 123})
        
        with pytest.raises(TypeError, match="Message ID must be an integer"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_message_id_validation_error_missing_key(self):
        """Test that missing message_id key raises TypeError - covers line 60"""
        message = self.create_valid_message()
        del message["message_id"]  # Remove the key entirely
        
        with pytest.raises(TypeError, match="Message ID must be an integer"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_created_at_validation_error_missing_key(self):
        """Test that missing created_at key raises TypeError - covers line 64"""
        message = self.create_valid_message()
        del message["created_at"]  # Remove the key entirely
        
        with pytest.raises(TypeError, match="created_at is required"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_created_at_validation_error_for_string(self):
        """Test that created_at as string raises TypeError - covers line 66"""
        message = self.create_valid_message(created_at="2023-01-01T00:00:00")
        
        with pytest.raises(TypeError, match="created_at must be a datetime object"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_created_at_validation_error_for_int(self):
        """Test that created_at as integer raises TypeError - covers line 66"""
        message = self.create_valid_message(created_at=1672531200)  # timestamp
        
        with pytest.raises(TypeError, match="created_at must be a datetime object"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_created_at_validation_error_for_float(self):
        """Test that created_at as float raises TypeError - covers line 66"""
        message = self.create_valid_message(created_at=1672531200.123)
        
        with pytest.raises(TypeError, match="created_at must be a datetime object"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_created_at_validation_error_for_none(self):
        """Test that created_at=None raises TypeError - covers line 66"""
        message = self.create_valid_message(created_at=None)
        
        with pytest.raises(TypeError, match="created_at must be a datetime object"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_created_at_validation_error_for_list(self):
        """Test that created_at as list raises TypeError - covers line 66"""
        message = self.create_valid_message(created_at=[datetime.now()])
        
        with pytest.raises(TypeError, match="created_at must be a datetime object"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_created_at_validation_error_for_dict(self):
        """Test that created_at as dict raises TypeError - covers line 66"""
        message = self.create_valid_message(created_at={"datetime": datetime.now()})
        
        with pytest.raises(TypeError, match="created_at must be a datetime object"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_created_at_validation_error_for_bool(self):
        """Test that created_at as boolean raises TypeError - covers line 66"""
        message = self.create_valid_message(created_at=True)
        
        with pytest.raises(TypeError, match="created_at must be a datetime object"):
            DeriverQueuePayload.create_payload(
                message=message,
                sender_name="sender",
                target_name="target",
                task_type="representation"
            )

    def test_model_dump_exception_handling(self):
        """Test that exceptions during model_dump are caught and wrapped in ValueError - covers lines 89-90"""
        from unittest.mock import patch
        
        message = self.create_valid_message()
        
        # Mock model_dump to raise an exception after successful model creation
        with patch.object(DeriverQueuePayload, 'model_dump', side_effect=RuntimeError("JSON serialization failed")):
            with pytest.raises(ValueError, match="Failed to create valid payload: JSON serialization failed"):
                DeriverQueuePayload.create_payload(
                    message=message,
                    sender_name="sender",
                    target_name="target",
                    task_type="representation"
                )