import json, unittest
from .parser import parse_json, parse_json_with_fixes, JSONParseError
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

# Create a console instance
console = Console()

def test_parse(input_json, description, expected=None):
    # Create a visually distinct section for each test
    console.print("\n")
    console.rule(f"[bold white on blue] TEST: {description} [/]", characters="=")
    console.print()
    
    # Display the input with better formatting
    input_panel = Panel(
        render_input(input_json),
        title="[bold white]Input JSON[/]",
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(input_panel)
    
    # Parse the JSON
    try:
        result = parse_json(input_json)
        
        # Create success message
        result_text = Text()
        result_text.append("✓ ", style="bold green")
        result_text.append("Valid JSON", style="green")
        
        # Display the result with nice formatting
        result_panel = Panel(
            render_result(result),
            title=result_text,
            border_style="green",
            padding=(1, 2)
        )
        console.print(result_panel)
        
        # Check if result matches expected output
        if expected is not None:
            if result == expected:
                console.print(Panel(
                    "✓ Result matches expected outcome",
                    title="[bold green]Test PASSED[/]",
                    border_style="green",
                    padding=(1, 2)
                ))
            else:
                console.print(Panel(
                    f"✗ Result does not match expected outcome\nExpected: {render_result(expected)}",
                    title="[bold red]Test FAILED[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
        
        return True
    
    except JSONParseError as e:
        try:
            # Display error with clear formatting
            error_text = Text()
            error_text.append("✗ ", style="bold red")
            error_text.append("JSON Parse Error", style="red")
            
            error_panel = Panel(
                f"{str(e)}", 
                title=error_text,
                border_style="red",
                padding=(1, 2)
            )
            console.print(error_panel)
            
            # Attempt to fix
            console.print(Markdown("### Attempting to fix..."))
            result, fixes = parse_json_with_fixes(input_json)
            
            if fixes:
                # Create a table for fixes
                fix_table = Table(show_header=True, header_style="bold yellow")
                fix_table.add_column("Applied Fixes")
                for fix in fixes:
                    fix_table.add_row(f"• {fix}")
                console.print(fix_table)
            else:
                console.print(Panel(
                    "[italic]No fix was recorded or fix could not be applied to this input[/]",
                    title="[bold red]Fix Failed[/]",
                    border_style="red"
                ))
            
            # Show fixed result
            fixed_panel = Panel(
                render_result(result),
                title="[bold green]Fixed Result[/]",
                border_style="green",
                padding=(1, 2)
            )
            console.print(fixed_panel)
            
            # Check if fixed result matches expected output
            if expected is not None:
                if result == expected:
                    console.print(Panel(
                        "✓ Fixed result matches expected outcome",
                        title="[bold green]Test PASSED[/]",
                        border_style="green",
                        padding=(1, 2)
                    ))
                else:
                    console.print(Panel(
                        f"✗ Fixed result does not match expected outcome\nExpected: {render_result(expected)}",
                        title="[bold red]Test FAILED[/]",
                        border_style="red",
                        padding=(1, 2)
                    ))
                            
        except Exception as fix_error:
            # Display error message for failed auto-fix
            failed_panel = Panel(
                str(fix_error),  # Convert exception to string
                title="✗ Auto-Fix Failed",
                border_style="red"
            )
            console.print(failed_panel)
            
            # If expected is provided but fix failed, show test failed
            if expected is not None:
                console.print(Panel(
                    "✗ Could not auto-fix to match expected outcome",
                    title="[bold red]Test FAILED[/]",
                    border_style="red",
                    padding=(1, 2)
                ))
            
        return False

def render_input(input_json):
    """Format input JSON for display"""
    if not input_json.strip():
        return "[italic](empty string)[/]"
    
    return f"{repr(input_json)}"

def render_result(result):
    """Format result for display"""
    if isinstance(result, (dict, list)):
        try:
            # Pretty format complex structures without syntax highlighting
            formatted = json.dumps(result, indent=2)
            return formatted
        except:
            return repr(result)
    else:
        # Display scalar values with type information
        return f"{result} ({type(result).__name__})"

def run_test_cases(test_cases, section_title):
    """Run a group of test cases with consistent formatting"""
    console.print(f"\n=== {section_title} ===", style="bold yellow")
    
    for test_case in test_cases:
        test_parse(
            test_case["json"],
            test_case["description"],
            test_case.get("expected")  # Use get() to handle cases with no expected value
        )

# Define test cases with expected results
valid_json_tests = [
    {
        "json": '{"name": "John", "age": 30}',
        "description": "Simple object",
        "expected": {"name": "John", "age": 30}
    },
    {
        "json": '[1, 2, 3, 4]',
        "description": "Simple array",
        "expected": [1, 2, 3, 4]
    },
    {
        "json": '{"numbers": [1, 2, 3], "nested": {"a": true, "b": false}}',
        "description": "Nested structures",
        "expected": {"numbers": [1, 2, 3], "nested": {"a": True, "b": False}}
    },
    {
        "json": 'null',
        "description": "Null value",
        "expected": None
    },
    {
        "json": '{"key": true}',
        "description": "Boolean value",
        "expected": {"key": True}
    },
    {
        "json": '{"unicode": "こんにちは世界"}',
        "description": "Unicode characters",
        "expected": {"unicode": "こんにちは世界"}
    },
    {
        "json": '{"unicode_escape": "\\u3053\\u3093\\u306b\\u3061\\u306f"}',
        "description": "Escaped Unicode",
        "expected": {"unicode_escape": "こんにちは"}
    },
    {
        "json": '{"binary_data": "SGVsbG8gV29ybGQ="}',
        "description": "Base64 encoded binary data",
        "expected": {"binary_data": "SGVsbG8gV29ybGQ="}
    },
    {
        "json": '{"large_number": 1.234e+100, "small_number": 1.234e-100}',
        "description": "Scientific notation",
        "expected": {"large_number": 1.234e+100, "small_number": 1.234e-100}
    },
    {
        "description": "Positive and negative floats",
        "json": '{"positive_float": 3.14, "negative_float": -2.71}',
        "expected": {"positive_float": 3.14, "negative_float": -2.71}
    },
    {
        "description": "Positive and negative integers",
        "json": '{"positive_int": 42, "negative_int": -42}',
        "expected": {"positive_int": 42, "negative_int": -42}
    },
    {
        "json": '{"max_int": 9223372036854775807, "min_int": -9223372036854775808}',
        "description": "Large integers",
        "expected": {"max_int": 9223372036854775807, "min_int": -9223372036854775808}
    },
    {
        "json": '{}',
        "description": "Empty object",
        "expected": {}
    },
    {
        "json": '[]',
        "description": "Empty array",
        "expected": []
    },
    {
        "json": '{"a":{"b":{"c":{"d":{"e":{"f":{"g":1}}}}}}}',
        "description": "Deeply nested object",
        "expected": {"a":{"b":{"c":{"d":{"e":{"f":{"g":1}}}}}}}
    },
    {
        "json": '[[[[[1]]]]]',
        "description": "Deeply nested array",
        "expected": [[[[[1]]]]]
    },
    {
        "json": '{"emoji": "😀 🚀 🌍"}',
        "description": "Emoji characters",
        "expected": {"emoji": "😀 🚀 🌍"}
    },
    {
        "json": '{"mixed": [1, "text", true, null, {"nested": []}]}',
        "description": "Mixed types array",
        "expected": {"mixed": [1, "text", True, None, {"nested": []}]}
    },
    # New valid test cases
    {
        "json": '{"special@chars#in$key": "value"}',
        "description": "Special characters in key",
        "expected": {"special@chars#in$key": "value"}
    },
    {
        "json": '{"whitespace": "   trimmed   "}',
        "description": "Whitespace in string value",
        "expected": {"whitespace": "   trimmed   "}
    },
    {
        "json": '[\n1,\n2,\n3\n]',
        "description": "Array with newlines",
        "expected": [1, 2, 3]
    },
    {
        "json": '{"zero": 0, "one": 1}',
        "description": "Simple numeric values",
        "expected": {"zero": 0, "one": 1}
    },
    {
        "json": '{"complex": {"nested": {"array": [1, 2, [3, 4, {"deep": null}]]}}}',
        "description": "Complex nested structure",
        "expected": {"complex": {"nested": {"array": [1, 2, [3, 4, {"deep": None}]]}}}
    },
        {
        "json": '{"escaped_chars": "\\u0021\\u0040\\u0023"}',
        "description": "Unicode escapes for special characters",
        "expected": {"escaped_chars": "!@#"}
    },
    {
        "json": '{"nested_arrays": [[[[]]]], "nested_objects": {"a":{"b":{"c":{}}}}}',
        "description": "Deeply nested empty structures",
        "expected": {"nested_arrays": [[[[]]]], "nested_objects": {"a":{"b":{"c":{}}}}}
    },
    {
        "json": '{"zero_fractions": [0.0000001, 0.000000001, 1e-10]}',
        "description": "Very small decimal numbers",
        "expected": {"zero_fractions": [0.0000001, 0.000000001, 1e-10]}
    },
    {
        "json": '{"huge_exponents": [1e+37, 1e+100, 1e+308]}',
        "description": "Large exponential numbers",
        "expected": {"huge_exponents": [1e+37, 1e+100, 1e+308]}
    },
    {
        "json": '{"all_escapes": "\\b\\f\\n\\r\\t\\/\\\\\\\""}',
        "description": "All standard JSON escape sequences",
        "expected": {"all_escapes": "\b\f\n\r\t/\\\""}
    },
    {
        "json": ' { "whitespace" : [ 1 , 2 , 3 ] } ',
        "description": "Excessive whitespace",
        "expected": {"whitespace": [1, 2, 3]}
    },
    {
        "json": '{"empty_string":""}',
        "description": "Empty string value",
        "expected": {"empty_string": ""}
    },
    {
        "json": '[0, 1, -0, -1, 9223372036854775807, -9223372036854775808]',
        "description": "Array of various integers including max/min",
        "expected": [0, 1, -0, -1, 9223372036854775807, -9223372036854775808]
    },
    {
        "json": '{"mixed_number_types": [0, -0, 0.0, -0.0, 1e2, -1e2, 1.0e2, -1.0e2]}',
        "description": "Mixed numeric types and formats",
        "expected": {"mixed_number_types": [0, -0, 0.0, -0.0, 1e2, -1e2, 1.0e2, -1.0e2]}
    },
    {
        "json": '{"special_floats": [0.5, -0.5, 0.25, -0.25, 0.125, -0.125]}',
        "description": "Special fraction floats",
        "expected": {"special_floats": [0.5, -0.5, 0.25, -0.25, 0.125, -0.125]}
    },
    {
        "json": '{"unusual_but_valid_keys": {"$special": true, "_underscore": true, "123numeric": true}}',
        "description": "Unusual but valid property names",
        "expected": {"unusual_but_valid_keys": {"$special": True, "_underscore": True, "123numeric": True}}
    },
    {
        "json": '{"control_char_escapes": "Line1\\nLine2\\rCarriage\\tTab\\bBackspace\\fFormfeed"}',
        "description": "Control character escapes",
        "expected": {"control_char_escapes": "Line1\nLine2\rCarriage\tTab\bBackspace\fFormfeed"}
    },
    {
        "json": '[true, false, null, 42, "text", {"key":"value"}, [1, 2]]',
        "description": "Array with all JSON value types",
        "expected": [True, False, None, 42, "text", {"key": "value"}, [1, 2]]
    },
    {
        "json": '{"empty_array_in_array": [[]]}',
        "description": "Empty array within array",
        "expected": {"empty_array_in_array": [[]]}
    },
    {
        "json": '{"empty_object_in_array": [{}]}',
        "description": "Empty object within array",
        "expected": {"empty_object_in_array": [{}]}
    },
    {
        "json": '{"long_decimal": 0.123456789012345678901234567890}',
        "description": "Long decimal number",
        "expected": {"long_decimal": 0.123456789012345678901234567890}
    }
]

invalid_json_tests = [
    {
        "json": """{"key": "values'"}""",
        "description": "Apostrophe in string",
        "expected": {"key": "values'"}
    },
    {
        "json": '',
        "description": "Empty input",
        "expected": None
    },
    {
        "json": '[1, 2, 3,]',
        "description": "Trailing comma in array",
        "expected": [1, 2, 3]
    },
    {
        "json": '{"a": 1, "b": 2,}',
        "description": "Trailing comma in object",
        "expected": {"a": 1, "b": 2}
    },
    {
        "json": '{"first": "John" "last": "Doe"}',
        "description": "Missing comma in object",
        "expected": {"first": "John", "last": "Doe"}
    },
    {
        "json": '{name: "John"}',
        "description": "Unquoted key",
        "expected": {"name": "John"}
    },
    {
        "json": '{"a": TRUE}',
        "description": "Invalid case for literals",
        "expected": {"a": True}
    },
    {
        "description": "negative with trailing decimal",
        "json": '{"negative_float": -.71}',
        "expected": {"negative_float": -0.71}
    },
    {
        "json": '{"unclosed": "string}',
        "description": "Unterminated string",
        "expected": {"unclosed": "string"}
    },
    {
        "json": '{"array": [1, 2, 3}',
        "description": "Missing closing bracket",
        "expected": {"array": [1, 2, 3]}
    },
    {
        "json": '{"valid": tru}',
        "description": "Malformed keyword",
        "expected": {"valid": True}
    },
    {
        "json": '{1: "numeric key"}',
        "description": "Numeric key without quotes",
        "expected": {"1": "numeric key"}
    },
    {
        "json": '{"nested": {"a": 1,}',
        "description": "Unclosed nested object",
        "expected": {"nested": {"a": 1}}
    },
    {
        "json": '{"key": [1, 2, 3',
        "description": "Unclosed array",
        "expected": {"key": [1, 2, 3]}
    },
    {
        "json": '{"a": "value with \n newline"}',
        "description": "Unescaped control character",
        "expected": {"a": "value with \n newline"}
    },
    {
        "json": '{"a": 01}',
        "description": "Leading zero in number",
        "expected": {"a": 1}
    },
    {
        "json": '{"a": .5}',
        "description": "Missing leading digit in number",
        "expected": {"a": 0.5}
    },
    {
        "json": '{"a": 1e+}',
        "description": "Invalid exponent in number",
        "expected": {"a": 1}
    },
    {
        "json": '{"a": +1}',
        "description": "Plus sign in number",
        "expected": {"a": 1}
    },
    {
        "json": '{"a": "\\u12"}',
        "description": "Incomplete unicode escape",
        "expected": {"a": "\ufffd"}
    },
    {
        "json": '{]',
        "description": "Mismatched brackets",
        "expected": {}
    },
    {
        "json": "{key: [1, 2, 3,}",
        "description": "Unquoted key with trailing comma and unclosed array",
        "expected": {"key": [1, 2, 3]}
    },
    {
        "json": "{key: [1, 2, 3,",
        "description": "Unquoted key with trailing comma and unclosed array and missing closing brace",
        "expected": {"key": [1, 2, 3]}
    },
    {
        "json": '// This is a comment\n{"a": 1}',
        "description": "JavaScript-style comments",
        "expected": {"a": 1}
    },
    {
        "json": '{"a": fales}',
        "description": "Misspelled false",
        "expected": {"a": False}
    },
    {
        "json": '{"a": nall}',
        "description": "Misspelled null",
        "expected": {"a": None}
    },
    {
        "json": '{"a": truee}',
        "description": "Misspelled true",
        "expected": {"a": True}
    },
    {
        "json": "{'a': 'value'}",
        "description": "Single quotes instead of double quotes",
        "expected": {"a": "value"}
    },
    {
        "json": '{"a": undefined}',
        "description": "JavaScript undefined value",
        "expected": {"a": None}
    },
    {
        "json": '{"a": NaN}',
        "description": "JavaScript NaN value",
        "expected": {"a": 0}  # An approximation, as JSON doesn't have NaN
    },
    {
        "json": '{"a": Infinity}',
        "description": "JavaScript Infinity value",
        "expected": {"a": float('inf')}  # An approximation
    },
    {
        "json": '{"a": 0xFF}',
        "description": "Hexadecimal notation",
        "expected": {"a": 255}
    },
    {
        "json": '{"a": 1.}',
        "description": "Trailing decimal point",
        "expected": {"a": 1.0}
    },
    {
        "json": '{"a": 1.2.3}',
        "description": "Multiple decimal points",
        "expected": {"a": "1.2.3"}
    },
    {
        "json": '{"a": 0b1010}',
        "description": "Binary notation",
        "expected": {"a": 10}
    },
    {
        "json": '{"a": True}',
        "description": "Python-style boolean",
        "expected": {"a": True}
    },
    {
        "json": '{"a": None}',
        "description": "Python-style null",
        "expected": {"a": None}
    },
    {
        "json": '{"a": 1}}',
        "description": "Extra closing brace",
        "expected": {"a": 1}
    },
    {
        "json": """{"key": "This is a test value with an apostrophe's"}""",
        "description": "Apostrophe in string",
        "expected": {"key": "This is a test value with an apostrophe's"}
    },
    {
        "json": '{"key": "random sentence2}',
        "description": "Unclosed string with random sentence",
        "expected": {"key": "random sentence2"}
    },
    {
        "json": '{"array": [1, 2, 3],,,}',
        "description": "Multiple trailing commas",
        "expected": {"array": [1, 2, 3]}
    },
    {
        "json": '{"a": "\\uXYZ1"}',
        "description": "Invalid Unicode escape sequence",
        "expected": {"a": "\ufffdXYZ1"}
    },
    {
        "json": '{"key": "\t \b \f \r \\\\ \\""}',
        "description": "Valid escape sequences test",
        "expected": {"key": "\t \b \f \r \\ \""}
    },
    {
        "json": '{"key": "\\x00"}',
        "description": "Invalid hex escape sequence",
        "expected": {"key": "\\x00"}
    },
    {
        "json": '{"key": "\0"}',
        "description": "Null byte in string",
        "expected": {"key": "\0"}
    },
    {
        "json": '{"key": -}',
        "description": "Invalid Token: minus sign without number",
        "expected": {"key": "-"}
    },
    {
        "json": '{"key": 12345678901234567890123456789}',
        "description": "Number too large for 64-bit",
        "expected": {"key": 12345678901234567890123456789}
    },
    {
        "json": '{"duplicate": 1, "duplicate": 2}',
        "description": "Duplicate keys",
        "expected": {"duplicate": 2}  # Last one wins
    },
    {
        "json": '{"a": "control\u0000character"}',
        "description": "Control character in Unicode escape",
        "expected": {"a": "control\u0000character"}
    },
    {
        "json": '{"a": 1, "b": @}',
        "description": "Invalid character",
        "expected": {"a": 1, "b": "@"}
    },
    {
        "json": '{"": "empty key", "": "empty key 2"}',
        "description": "Empty key",
        "expected": {"": "empty key 2"}
    },
    {
        "json": '[,1,2,3]',
        "description": "Leading comma in array",
        "expected": [1, 2, 3]
    },
    {
        "json": '{"key": --1}',
        "description": "Double negative",
        "expected": {"key": 1}
    },
    {
        "json": '[1, 2, 3] [4, 5, 6]',
        "description": "Multiple top-level arrays without separator",
        "expected": [[1, 2, 3], [4, 5, 6]]
    },
    {
        "json": '{"a": 1} {"b": 2}',
        "description": "Multiple top-level objects without separator",
        "expected": [{"a": 1}, {"b": 2}]
    },
    {
        "json": '{"key": ++1}',
        "description": "Double positive",
        "expected": {"key": 1}
    },
    {
        "json": "Here is the json: \n{'key': 'value'}",
        "description": "Extra text before JSON",
        "expected": {"key": "value"}
    },
    {
        "json": """{"key": "This is a test value with a "double quote" "}""",
        "description": "Double quote in string",
        "expected": {"key": "This is a test value with a \"double quote\" "}
    },
    {
        "json": """{'key': 'This is a test value with a 'single quote' '}""",
        "description": "Single quote in string of single quotes",
        "expected": {"key": "This is a test value with a 'single quote' "}
    },
    {
        "description": "Missing closing quote and multiple keys",
        "json": '{"key": "This is test value with a missing closing quote, "key2": "value2"}',
        "expected": {"key": "This is test value with a missing closing quote", "key2": "value2"}
    },
    # New invalid test cases
    {
        "json": '{"array": [1, 2, [3, 4]]}',
        "description": "Nested arrays",
        "expected": {"array": [1, 2, [3, 4]]}
    },
    {
        "json": '/*Multi-line\ncomment*/\n{"a": 1}',
        "description": "JavaScript-style block comments",
        "expected": {"a": 1}
    },
    {
        "json": '"""Multi-line\nPython string"""\n{"a": 1}',
        "description": "Python-style triple quotes",
        "expected": {"a": 1}
    },
    {
        "json": '{"a": """Multi-line\nPython string"""}',
        "description": "Python-style triple quotes as value",
        "expected": {"a": "Multi-line\nPython string"}
    },
    {
        "json": '{"mixed_types": ["string", 42, null, {"nested": true}, [1,2]]}',
        "description": "Array with mixed value types",
        "expected": {"mixed_types": ["string", 42, None, {"nested": True}, [1, 2]]}
    },
    {
        "json": '{"a": 1} // Comment at the end',
        "description": "Comment at the end",
        "expected": {"a": 1}
    }
]

llm_json_tests = [
    {
        "json": """
{
     "summary": "The current stage is to calculate the sine of the number of qbits using the math agent, but since there are no outputs collected yet, the calculation cannot be performed. The output from stage 1 is needed to determine the number of qbits."
},
""",
        "description": "LLM Extra Comma",
        "expected": {
            "summary": "The current stage is to calculate the sine of the number of qbits using the math agent, but since there are no outputs collected yet, the calculation cannot be performed. The output from stage 1 is needed to determine the number of qbits."
        }
    },
    {
        "json": """
{
    "meets_requirements": true,
    "suggestions": "The code is overly complex for a simple task and includes unnecessary error handling and validation. It can be simplified to print('hello world')",
    "improved_code": "print('hello world')"
},
{
    "meets_requirements": true,
    "suggestions": "Consider adding comments to explain the purpose of the code, but in this case, it's a simple print statement",
    "improved_code": "print('hello world')"
},
{
    "meets_requirements": true,
    "suggestions": "Consider adding comments to explain the purpose of the code, but in this case, that's a simple print statement",
    "improved_code": "print('hello world')"
}
""",
        "description": "Missing List brackets",
        "expected": [
            {
                "meets_requirements": True,
                "suggestions": "The code is overly complex for a simple task and includes unnecessary error handling and validation. It can be simplified to print('hello world')",
                "improved_code": "print('hello world')"
            },
            {
                "meets_requirements": True,
                "suggestions": "Consider adding comments to explain the purpose of the code, but in this case, it's a simple print statement",
                "improved_code": "print('hello world')"
            },
            {
                "meets_requirements": True,
                "suggestions": "Consider adding comments to explain the purpose of the code, but in this case, that's a simple print statement",
                "improved_code": "print('hello world')"
            }
        ]
    },
    {
        "json": '''
{
    "key": "random sentence"
},
{
    "key": "random sentence2
}
''',
        "description": "LLM JSON with missing quotes",
        "expected": [
            {"key": "random sentence"},
            {"key": "random sentence2"}
        ]
    },
    {
        "json": '''
[
    {
        "key": "value",
    },
    {
        "key": "value2,
    }
]
''',
        "description": "List of objects with missing quotes",
        "expected": [
            {"key": "value"},
            {"key": "value2"}
        ]
    },
    {
        "json": '''
{
   "key": "value" 
},
''',
        "description": "Extra comma after object",
        "expected": {"key": "value"}
    },
    {
        "json": '''
{
    "key": "value"
}

{
    "key": "value",
}
''',
        "description": "No Comma between objects. Extra comma in the second object.",
        "expected": [
            {"key": "value"},
            {"key": "value"}
        ]
    },
    {
        "json": '''
{
    "key": "value"
}


{
    "key": "value"
}
''',
        "description": "No Comma between objects",
        "expected": [
            {"key": "value"},
            {"key": "value"}
        ]
    },
    {
        "json": '''
{
    "key": "value",
}
''',
        "description": "Extra comma in the object",
        "expected": {"key": "value"}
    },
    {
        "json": '''
Sure, Here is a JSON object:

{
   "key": "value" 
}
''',
        "description": "Extra text before JSON",
        "expected": {"key": "value"}
    },
    {
        "json": '''
Sure, Here is a JSON object:
{
    "key": "value"
}

''',
        "description": "Extra text before JSON and newline after JSON",
        "expected": {"key": "value"}
    },
    {
        "json": '''
{
    "stage_1": {
        "task": "Browse the web for information on types of sewage treatment plants in India",
        "details": "Use the internet agent to search for relevant information on sewage treatment plants in India, including small, medium, and large-scale plants, and their treatment methods for industrial and domestic sewage",
        "expected_output": "A list of different types of sewage treatment plants in India, including their sizes and treatment methods"
    },
    "stage_2": {
        "task": "Categorize sewage treatment plants by size",
        "details": "Use the information gathered in stage 1 to categorize sewage treatment plants in India into small, medium, and large-scale plants, based on their capacity to treat sewage",
        "expected_output": "A categorized list of small, medium, and large-scale sewage treatment plants in India"
    },
    "stage_3": {
        "task": "Research sewage treatment methods for domestic sewage",
        "details": "Use the internet agent to research and gather information on different sewage treatment methods used for domestic sewage in India, including physical, chemical, and biological treatment methods",
        "expected_output": "A detailed list of sewage treatment methods used for domestic sewage in India, including their advantages and disadvantages"
    },
    "stage_4": {
        "task": "Research sewage treatment methods for industrial sewage",
        "details": "Use the internet agent to research and gather information on different sewage treatment methods used for industrial sewage in India, including physical, chemical, and biological treatment methods",
        "expected_output": "A detailed list of sewage treatment methods used for industrial sewage in India, including their advantages and disadvantages"
    },
    "stage_5": {
        "task": "Compile and structure the final output",
        "details": "Use the information gathered in stages 1-4 to compile a clean and structured output that includes the different types of sewage treatment plants in India, their sizes, and treatment methods for industrial and domestic sewage",
        "expected_output": "A detailed and structured output that includes the following information: 
            - Types of sewage treatment plants in India (small, medium, large)
            - Treatment methods for domestic sewage (physical, chemical, biological)
            - Treatment methods for industrial sewage (physical, chemical, biological)
            - Advantages and disadvantages of each treatment method
            - Examples of sewage treatment plants in India that use each treatment method"
    }
}
''',
        "description": "Complex JSON with multiple stages",
        "expected": {
            "stage_1": {
                "task": "Browse the web for information on types of sewage treatment plants in India",
                "details": "Use the internet agent to search for relevant information on sewage treatment plants in India, including small, medium, and large-scale plants, and their treatment methods for industrial and domestic sewage",
                "expected_output": "A list of different types of sewage treatment plants in India, including their sizes and treatment methods"
            },
            "stage_2": {
                "task": "Categorize sewage treatment plants by size",
                "details": "Use the information gathered in stage 1 to categorize sewage treatment plants in India into small, medium, and large-scale plants, based on their capacity to treat sewage",
                "expected_output": "A categorized list of small, medium, and large-scale sewage treatment plants in India"
            },
            "stage_3": {
                "task": "Research sewage treatment methods for domestic sewage",
                "details": "Use the internet agent to research and gather information on different sewage treatment methods used for domestic sewage in India, including physical, chemical, and biological treatment methods",
                "expected_output": "A detailed list of sewage treatment methods used for domestic sewage in India, including their advantages and disadvantages"
            },
            "stage_4": {
                "task": "Research sewage treatment methods for industrial sewage",
                "details": "Use the internet agent to research and gather information on different sewage treatment methods used for industrial sewage in India, including physical, chemical, and biological treatment methods",
                "expected_output": "A detailed list of sewage treatment methods used for industrial sewage in India, including their advantages and disadvantages"
            },
            "stage_5": {
                "task": "Compile and structure the final output",
                "details": "Use the information gathered in stages 1-4 to compile a clean and structured output that includes the different types of sewage treatment plants in India, their sizes, and treatment methods for industrial and domestic sewage",
                "expected_output": "A detailed and structured output that includes the following information: \n            - Types of sewage treatment plants in India (small, medium, large)\n            - Treatment methods for domestic sewage (physical, chemical, biological)\n            - Treatment methods for industrial sewage (physical, chemical, biological)\n            - Advantages and disadvantages of each treatment method\n            - Examples of sewage treatment plants in India that use each treatment method"
            }
        }
    },
]

class JSONParserTest(unittest.TestCase):
    """Unit tests for JSON parsing"""

    def test_valid_json(self):
        for test_case in valid_json_tests:
            with self.subTest(test_case=test_case["description"]): # Use description for better subtest naming
                result = parse_json(test_case["json"])
                self.assertEqual(result, test_case["expected"])

    def test_invalid_json(self):
        for test_case in invalid_json_tests:
            with self.subTest(test_case=test_case["description"]):
                result, _ = parse_json_with_fixes(test_case["json"])
                self.assertEqual(result, test_case["expected"])

    def test_llm_json(self):
        for test_case in llm_json_tests:
            with self.subTest(test_case=test_case["description"]):
                result, _ = parse_json_with_fixes(test_case["json"])
                self.assertEqual(result, test_case["expected"])
    
# Run the test cases
if __name__ == "__main__":
    run_test_cases(valid_json_tests, "VALID JSON EXAMPLES")
    run_test_cases(invalid_json_tests, "INVALID JSON EXAMPLES")
    run_test_cases(llm_json_tests, "LLM JSON EXAMPLES")
    print(f"\n\n=== UNIT TESTS ===")
    unittest.main()