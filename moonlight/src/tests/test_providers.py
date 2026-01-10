import unittest, sys

from pathlib import Path
print(str(Path(__file__).parent.parent))
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from provider.main import Provider, ProviderError

class TestProvider(unittest.TestCase):
        def test_builtin_provider(self):
            """ Checks if the ENDPOINT mapping is working properly or not """
            # Test that builtin provider names get mapped to correct endpoints
            provider = Provider(source="openai", api="test-key")
            self.assertEqual(provider._source, "https://api.openai.com/v1")
            
            provider = Provider(source="groq", api="test-key")
            self.assertEqual(provider._source, "https://api.groq.com/openai/v1")
            
            provider = Provider(source="together", api="test-key")
            self.assertEqual(provider._source, "https://api.together.xyz/v1")
        
        def test_empty_source(self):
            """ Test that empty source raises ProviderError """
            with self.assertRaises(ProviderError) as context:
                provider = Provider(source="", api="test-key")
            self.assertIn("Source is empty", str(context.exception))
        
        def test_none_source(self):
            """ Test that None source raises ProviderError """
            with self.assertRaises(ProviderError) as context:
                provider = Provider(source=None, api="test-key")
            self.assertIn("Source is empty", str(context.exception))
        
        def test_non_string_source(self):
            """ Test that non-string source raises ProviderError """
            with self.assertRaises(ProviderError) as context:
                provider = Provider(source=123, api="test-key")
            self.assertIn("Source must be a string", str(context.exception))
        
        def test_invalid_endpoint_url(self):
            """ Test that invalid endpoint URL raises ProviderError """
            with self.assertRaises(ProviderError) as context:
                provider = Provider(source="not-a-valid-url", api="test-key")
            self.assertIn("Invalid provider endpoint", str(context.exception))
        
        def test_empty_api_key(self):
            """ Test that empty API key raises ProviderError """
            with self.assertRaises(ProviderError) as context:
                provider = Provider(source="openai", api="")
            self.assertIn("API key is empty", str(context.exception))
        
        def test_none_api_key(self):
            """ Test that None API key raises ProviderError """
            with self.assertRaises(ProviderError) as context:
                provider = Provider(source="openai", api=None)
            self.assertIn("API key is empty", str(context.exception))
        
        def test_non_string_api_key(self):
            """ Test that non-string API key raises ProviderError """
            with self.assertRaises(ProviderError) as context:
                provider = Provider(source="openai", api=12345)
            self.assertIn("API must be a string", str(context.exception))
        
        def test_valid_custom_endpoint(self):
            """ Test that valid custom endpoint works correctly """
            provider = Provider(source="https://custom-api.example.com/v1", api="test-key")
            self.assertEqual(provider._source, "https://custom-api.example.com/v1")
        
        def test_valid_builtin_provider_with_api(self):
            """ Test that valid builtin provider with API key works correctly """
            provider = Provider(source="openai", api="sk-test123")
            self.assertEqual(provider._source, "https://api.openai.com/v1")

if __name__ == "__main__":    
    unittest.main()