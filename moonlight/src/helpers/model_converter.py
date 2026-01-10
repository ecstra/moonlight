import json
from typing import Union, get_origin, get_args
from dataclasses import fields, MISSING, is_dataclass

class ModelConverter:
    """
    Utility class for converting between Pydantic/dataclass models and JSON schemas
    """
    
    @staticmethod
    def _get_type_schema(field_type):
        """
        Convert a Python type to JSON schema type
        """
        origin = get_origin(field_type)
        
        # Handle Optional
        if origin is Union:
            args = get_args(field_type)
            # Check if it's Optional (Union with None)
            if type(None) in args:
                non_none_types = [t for t in args if t is not type(None)]
                if len(non_none_types) == 1:
                    return ModelConverter._get_type_schema(non_none_types[0])
        
        # Handle Dict
        if origin is dict or field_type is dict:
            args = get_args(field_type)
            if args:
                _, value_type = args
                return {
                    "type": "object",
                    "additionalProperties": ModelConverter._get_type_schema(value_type)
                }
            return {"type": "object"}
        
        # Handle List
        if origin is list or field_type is list:
            args = get_args(field_type)
            if args:
                return {
                    "type": "array",
                    "items": ModelConverter._get_type_schema(args[0])
                }
            return {"type": "array"}
        
        # Handle basic types
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        
        if field_type in type_mapping:
            return {"type": type_mapping[field_type]}
        
        # Handle classes (dataclass or BaseModel)
        if isinstance(field_type, type) and (hasattr(field_type, '__dataclass_fields__') or hasattr(field_type, 'model_fields')):
            return ModelConverter.model_to_schema(field_type)
        
        return {"type": "string"}  # Default fallback

    @staticmethod
    def model_to_schema(cls):
        """
        Convert a Pydantic model or dataclass to JSON schema
        """
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Handle Pydantic models
        if hasattr(cls, 'model_fields'):
            for field_name, field_info in cls.model_fields.items():
                field_type = field_info.annotation
                schema["properties"][field_name] = ModelConverter._get_type_schema(field_type)
                
                # Check if field is required (not Optional and required)
                origin = get_origin(field_type)
                is_optional = origin is Union and type(None) in get_args(field_type)
                
                if field_info.is_required() and not is_optional:
                    schema["required"].append(field_name)
        
        # Handle dataclasses
        elif hasattr(cls, '__dataclass_fields__'):
            for field in fields(cls):
                schema["properties"][field.name] = ModelConverter._get_type_schema(field.type)
                
                # Check if field has default value
                if field.default is field.default_factory is MISSING:
                    schema["required"].append(field.name)
        
        if not schema["required"]:
            del schema["required"]
        
        return schema

    @staticmethod
    def json_to_model(cls, data):
        """
        Convert JSON data back to typed class instance
        """
        # If data is a string, parse it first
        if isinstance(data, str):
            # Remove markdown code block markers if present
            data = data.strip()
            
            # Remove opening markdown code block (```json or ```)
            if data.startswith('```'):
                # Find the first newline after ```
                first_newline = data.find('\n')
                if first_newline != -1:
                    data = data[first_newline + 1:]
                else:
                    # No newline, just remove the ```
                    data = data[3:]
                    if data.startswith('json'):
                        data = data[4:]
                    data = data.strip()
            
            # Remove closing markdown code block (```)
            if data.endswith('```'):
                data = data[:-3].rstrip()
            
            data = json.loads(data)
        
        # Handle Pydantic models - they have built-in validation
        if hasattr(cls, 'model_validate'):
            # For Pydantic v2, handle Optional fields that are missing
            if hasattr(cls, 'model_fields'):
                processed_data = dict(data)
                for field_name, field_info in cls.model_fields.items():
                    # If field is missing and is Optional, provide None
                    if field_name not in processed_data:
                        origin = get_origin(field_info.annotation)
                        if origin is Union and type(None) in get_args(field_info.annotation):
                            processed_data[field_name] = None
                return cls.model_validate(processed_data)
            return cls.model_validate(data)
        
        # Handle dataclasses
        if is_dataclass(cls):
            field_values = {}
            for field in fields(cls):
                field_name = field.name
                if field_name not in data:
                    continue
                    
                field_type = field.type
                field_value = data[field_name]
                
                # Recursively handle nested classes
                if isinstance(field_type, type) and (is_dataclass(field_type) or hasattr(field_type, 'model_validate')):
                    field_values[field_name] = ModelConverter.json_to_model(field_type, field_value)
                else:
                    field_values[field_name] = field_value
            
            return cls(**field_values)
        
        raise ValueError(f"Unsupported type: {cls}")
    
    @staticmethod
    def model_to_string(cls):
        """
        Convert a model class to a Python class definition string
        
        Args:
            cls: A Pydantic model or dataclass class
            
        Returns:
            str: Python class definition(s) as a string, including nested classes
        """
        visited = set()
        definitions = []
        
        def collect_definitions(model_cls):
            """Recursively collect class definitions"""
            if model_cls in visited or not isinstance(model_cls, type):
                return
            
            # Skip built-in types
            if model_cls.__module__ in ('builtins', 'typing'):
                return
            
            visited.add(model_cls)
            
            # Collect nested classes first
            if hasattr(model_cls, 'model_fields'):
                for field_info in model_cls.model_fields.values():
                    nested_types = ModelConverter._extract_nested_types(field_info.annotation)
                    for nested_type in nested_types:
                        collect_definitions(nested_type)
            elif hasattr(model_cls, '__dataclass_fields__'):
                for field in fields(model_cls):
                    nested_types = ModelConverter._extract_nested_types(field.type)
                    for nested_type in nested_types:
                        collect_definitions(nested_type)
            
            # Generate class definition
            class_def = ModelConverter._generate_class_definition(model_cls)
            definitions.append(class_def)
        
        collect_definitions(cls)
        return "\n\n".join(definitions)
    
    @staticmethod
    def _extract_nested_types(field_type):
        """
        Extract nested class types from a type annotation
        """
        nested = []
        origin = get_origin(field_type)
        
        # Handle Union/Optional
        if origin is Union:
            for arg in get_args(field_type):
                if isinstance(arg, type) and arg is not type(None):
                    nested.extend(ModelConverter._extract_nested_types(arg))
        
        # Handle Dict, List
        elif origin in (dict, list):
            for arg in get_args(field_type):
                if isinstance(arg, type):
                    nested.extend(ModelConverter._extract_nested_types(arg))
        
        # Handle class types
        elif isinstance(field_type, type) and (hasattr(field_type, '__dataclass_fields__') or hasattr(field_type, 'model_fields')):
            nested.append(field_type)
        
        return nested
    
    @staticmethod
    def _generate_class_definition(cls):
        """
        Generate a single class definition string
        """
        lines = []
        
        # Determine base class
        if hasattr(cls, 'model_fields'):
            lines.append(f"class {cls.__name__}(BaseModel):")
        elif is_dataclass(cls):
            lines.append(f"@dataclass")
            lines.append(f"class {cls.__name__}:")
        else:
            lines.append(f"class {cls.__name__}:")
        
        # Generate field definitions
        if hasattr(cls, 'model_fields'):
            for field_name, field_info in cls.model_fields.items():
                type_str = ModelConverter._format_type_annotation(field_info.annotation)
                lines.append(f"    {field_name}: {type_str}")
        elif hasattr(cls, '__dataclass_fields__'):
            for field in fields(cls):
                type_str = ModelConverter._format_type_annotation(field.type)
                # Check if field has default
                if field.default is not MISSING:
                    lines.append(f"    {field.name}: {type_str} = {repr(field.default)}")
                elif field.default_factory is not MISSING:
                    lines.append(f"    {field.name}: {type_str} = field(default_factory=...)")
                else:
                    lines.append(f"    {field.name}: {type_str}")
        
        if len(lines) == 1 or (len(lines) == 2 and lines[0].startswith("@dataclass")):
            lines.append("    pass")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_type_annotation(field_type):
        """
        Format a type annotation as a string
        """
        origin = get_origin(field_type)
        
        # Handle Union/Optional
        if origin is Union:
            args = get_args(field_type)
            if type(None) in args and len(args) == 2:
                # It's Optional[X]
                non_none_type = [t for t in args if t is not type(None)][0]
                return f"Optional[{ModelConverter._format_type_annotation(non_none_type)}]"
            else:
                # It's Union[X, Y, ...]
                arg_strs = [ModelConverter._format_type_annotation(arg) for arg in args]
                return f"Union[{', '.join(arg_strs)}]"
        
        # Handle Dict
        if origin is dict:
            args = get_args(field_type)
            if args:
                key_type, val_type = args
                return f"Dict[{ModelConverter._format_type_annotation(key_type)}, {ModelConverter._format_type_annotation(val_type)}]"
            return "Dict"
        
        # Handle List
        if origin is list:
            args = get_args(field_type)
            if args:
                return f"List[{ModelConverter._format_type_annotation(args[0])}]"
            return "List"
        
        # Handle class types
        if isinstance(field_type, type):
            return field_type.__name__
        
        # Fallback to string representation
        return str(field_type)