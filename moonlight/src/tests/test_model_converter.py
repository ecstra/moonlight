import unittest, sys
from pathlib import Path
from typing import Optional, Dict, List
from pydantic import BaseModel
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.model_converter import ModelConverter


# Test models
@dataclass
class SimpleDataclass:
    name: str
    age: int


@dataclass
class NestedDataclass:
    id: int
    details: SimpleDataclass


class SimplePydantic(BaseModel):
    name: str
    age: int


class PydanticWithOptional(BaseModel):
    name: str
    description: Optional[str]
    count: int


class PydanticWithComplex(BaseModel):
    title: str
    tags: List[str]
    metadata: Dict[str, str]


@dataclass
class DataclassInPydantic:
    value: str


class PydanticWithNested(BaseModel):
    id: int
    data: DataclassInPydantic


class TestModelConverter(unittest.TestCase):
    
    def test_pydantic_basic_schema(self):
        """Test schema generation for basic Pydantic model"""
        schema = ModelConverter.model_to_schema(SimplePydantic)
        
        self.assertEqual(schema["type"], "object")
        self.assertIn("name", schema["properties"])
        self.assertIn("age", schema["properties"])
        self.assertEqual(schema["properties"]["name"]["type"], "string")
        self.assertEqual(schema["properties"]["age"]["type"], "integer")
        self.assertEqual(set(schema["required"]), {"name", "age"})
    
    def test_pydantic_optional_schema(self):
        """Test schema generation with Optional fields"""
        schema = ModelConverter.model_to_schema(PydanticWithOptional)
        
        self.assertIn("name", schema["properties"])
        self.assertIn("description", schema["properties"])
        self.assertIn("count", schema["properties"])
        # Optional field should not be required
        self.assertNotIn("description", schema["required"])
        self.assertIn("name", schema["required"])
        self.assertIn("count", schema["required"])
    
    def test_pydantic_complex_types_schema(self):
        """Test schema generation with List and Dict types"""
        schema = ModelConverter.model_to_schema(PydanticWithComplex)
        
        # Check List[str]
        self.assertEqual(schema["properties"]["tags"]["type"], "array")
        self.assertEqual(schema["properties"]["tags"]["items"]["type"], "string")
        
        # Check Dict[str, str]
        self.assertEqual(schema["properties"]["metadata"]["type"], "object")
        self.assertEqual(schema["properties"]["metadata"]["additionalProperties"]["type"], "string")
    
    def test_dataclass_basic_schema(self):
        """Test schema generation for basic dataclass"""
        schema = ModelConverter.model_to_schema(SimpleDataclass)
        
        self.assertEqual(schema["type"], "object")
        self.assertIn("name", schema["properties"])
        self.assertIn("age", schema["properties"])
        self.assertEqual(schema["properties"]["name"]["type"], "string")
        self.assertEqual(schema["properties"]["age"]["type"], "integer")
        self.assertEqual(set(schema["required"]), {"name", "age"})
    
    def test_nested_schema(self):
        """Test schema generation with nested classes"""
        schema = ModelConverter.model_to_schema(PydanticWithNested)
        
        self.assertIn("data", schema["properties"])
        nested_schema = schema["properties"]["data"]
        self.assertEqual(nested_schema["type"], "object")
        self.assertIn("value", nested_schema["properties"])
        self.assertEqual(nested_schema["properties"]["value"]["type"], "string")
    
    def test_json_string_to_pydantic(self):
        """Test converting JSON string to Pydantic model"""
        json_str = '{"name": "Alice", "age": 30}'
        result = ModelConverter.json_to_model(SimplePydantic, json_str)
        
        self.assertIsInstance(result, SimplePydantic)
        self.assertEqual(result.name, "Alice")
        self.assertEqual(result.age, 30)
    
    def test_dict_to_pydantic(self):
        """Test converting dict to Pydantic model"""
        data = {"name": "Bob", "age": 25}
        result = ModelConverter.json_to_model(SimplePydantic, data)
        
        self.assertIsInstance(result, SimplePydantic)
        self.assertEqual(result.name, "Bob")
        self.assertEqual(result.age, 25)
    
    def test_json_to_pydantic_with_optional(self):
        """Test converting JSON to Pydantic model with Optional fields"""
        # With optional field
        data1 = {"name": "Charlie", "description": "Test user", "count": 5}
        result1 = ModelConverter.json_to_model(PydanticWithOptional, data1)
        self.assertEqual(result1.description, "Test user")
        
        # Without optional field
        data2 = {"name": "Dave", "count": 10}
        result2 = ModelConverter.json_to_model(PydanticWithOptional, data2)
        self.assertIsNone(result2.description)
    
    def test_json_to_dataclass(self):
        """Test converting JSON to dataclass"""
        data = {"name": "Eve", "age": 28}
        result = ModelConverter.json_to_model(SimpleDataclass, data)
        
        self.assertIsInstance(result, SimpleDataclass)
        self.assertEqual(result.name, "Eve")
        self.assertEqual(result.age, 28)
    
    def test_json_to_nested_pydantic(self):
        """Test converting JSON with nested dataclass to Pydantic model"""
        data = {
            "id": 1,
            "data": {
                "value": "test"
            }
        }
        result = ModelConverter.json_to_model(PydanticWithNested, data)
        
        self.assertIsInstance(result, PydanticWithNested)
        self.assertEqual(result.id, 1)
        self.assertIsInstance(result.data, DataclassInPydantic)
        self.assertEqual(result.data.value, "test")
    
    def test_json_to_nested_dataclass(self):
        """Test converting JSON with nested dataclass"""
        data = {
            "id": 42,
            "details": {
                "name": "Frank",
                "age": 35
            }
        }
        result = ModelConverter.json_to_model(NestedDataclass, data)
        
        self.assertIsInstance(result, NestedDataclass)
        self.assertEqual(result.id, 42)
        self.assertIsInstance(result.details, SimpleDataclass)
        self.assertEqual(result.details.name, "Frank")
        self.assertEqual(result.details.age, 35)
    
    def test_json_with_complex_types(self):
        """Test converting JSON with List and Dict"""
        data = {
            "title": "Test",
            "tags": ["python", "testing", "unittest"],
            "metadata": {"author": "John", "version": "1.0"}
        }
        result = ModelConverter.json_to_model(PydanticWithComplex, data)
        
        self.assertEqual(result.title, "Test")
        self.assertEqual(result.tags, ["python", "testing", "unittest"])
        self.assertEqual(result.metadata, {"author": "John", "version": "1.0"})
    
    def test_unsupported_type_raises_error(self):
        """Test that unsupported types raise ValueError"""
        class UnsupportedClass:
            pass
        
        with self.assertRaises(ValueError) as context:
            ModelConverter.json_to_model(UnsupportedClass, {})
        self.assertIn("Unsupported type", str(context.exception))
    
    def test_schema_no_required_fields(self):
        """Test schema generation when all fields have defaults"""
        @dataclass
        class AllOptionalDataclass:
            name: str = "default"
            age: int = 0
        
        schema = ModelConverter.model_to_schema(AllOptionalDataclass)
        self.assertNotIn("required", schema)
    
    def test_bool_and_float_types(self):
        """Test schema generation for bool and float types"""
        class TypeVariety(BaseModel):
            is_active: bool
            score: float
        
        schema = ModelConverter.model_to_schema(TypeVariety)
        self.assertEqual(schema["properties"]["is_active"]["type"], "boolean")
        self.assertEqual(schema["properties"]["score"]["type"], "number")


if __name__ == "__main__":
    unittest.main()