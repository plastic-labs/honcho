import os
import sys

# Set environment variables for testing
os.environ['LLM_EMBEDDING_PROVIDER'] = 'custom'
os.environ['LLM_CUSTOM_EMBEDDING_API_KEY'] = 'test-api-key'
os.environ['LLM_CUSTOM_EMBEDDING_BASE_URL'] = 'https://api.example.com/v1'
os.environ['LLM_CUSTOM_EMBEDDING_MODEL'] = 'custom-embedding-model'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Only import settings to avoid google genai import error
from src.config import settings

print("Testing custom embedding configuration...")
print(f"Embedding provider: {settings.LLM.EMBEDDING_PROVIDER}")
print(f"Custom API key: {settings.LLM.CUSTOM_EMBEDDING_API_KEY}")
print(f"Custom base URL: {settings.LLM.CUSTOM_EMBEDDING_BASE_URL}")
print(f"Custom model: {settings.LLM.CUSTOM_EMBEDDING_MODEL}")

# Verify that the configuration is correctly loaded
if settings.LLM.EMBEDDING_PROVIDER == 'custom':
    print("✓ EMBEDDING_PROVIDER correctly set to 'custom'")
else:
    print("✗ EMBEDDING_PROVIDER not set correctly")

if settings.LLM.CUSTOM_EMBEDDING_API_KEY == 'test-api-key':
    print("✓ CUSTOM_EMBEDDING_API_KEY correctly loaded")
else:
    print("✗ CUSTOM_EMBEDDING_API_KEY not loaded correctly")

if settings.LLM.CUSTOM_EMBEDDING_BASE_URL == 'https://api.example.com/v1':
    print("✓ CUSTOM_EMBEDDING_BASE_URL correctly loaded")
else:
    print("✗ CUSTOM_EMBEDDING_BASE_URL not loaded correctly")

if settings.LLM.CUSTOM_EMBEDDING_MODEL == 'custom-embedding-model':
    print("✓ CUSTOM_EMBEDDING_MODEL correctly loaded")
else:
    print("✗ CUSTOM_EMBEDDING_MODEL not loaded correctly")

print("Test completed!")
