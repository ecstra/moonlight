import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Any, Tuple

class TokenType(Enum):
    LEFT_BRACE = auto()    # {
    RIGHT_BRACE = auto()   # }
    LEFT_BRACKET = auto()  # [
    RIGHT_BRACKET = auto() # ]
    COLON = auto()         # :
    COMMA = auto()         # ,
    STRING = auto()        # "text"
    NUMBER = auto()        # 123, 45.67
    TRUE = auto()          # true
    FALSE = auto()         # false
    NULL = auto()          # null
    IDENTIFIER = auto()    # unquoted text
    EOF = auto()           # end of file

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
    
class Lexer:
    """
    Lexer for parsing and fixing JSON-like text.
    """
    
    def __init__(
        self, 
        text: str,
        auto_fix_mode: bool = False,
    ):
        self.text = text
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.messages = []
        self.auto_fix_mode = auto_fix_mode
        
        # Constants
        self.HEX_DIGITS = "0123456789abcdefABCDEF"
        
        # Common misspellings of JSON literals
        self.TRUE_VARIANTS = ('t', 'tru', 'tr', 'ture', 'true', 'truee', 'truw', 'treu')
        self.FALSE_VARIANTS = ('f', 'fal', 'fals', 'false', 'fales', 'falsy', 'flase')
        self.NULL_VARIANTS = ('n', 'nu', 'nul', 'null', 'nill', 'nall', 'none', 'nil')
    
    def tokenize(self) -> Tuple[List[Token], List[str]]:
        """
        Tokenize the input text into a list of tokens.
        
        Returns:
            A tuple containing the list of tokens and any messages generated during tokenization.
        """
        while self.position < len(self.text):
            start_line = self.line
            start_column = self.column
            char = self._current_char()
            
            if char.isspace():
                self._handle_whitespace()
            elif char == '{':
                self._add_token(TokenType.LEFT_BRACE, '{', start_line, start_column)
                self._advance()
            elif char == '}':
                self._add_token(TokenType.RIGHT_BRACE, '}', start_line, start_column)
                self._advance()
            elif char == '[':
                self._add_token(TokenType.LEFT_BRACKET, '[', start_line, start_column)
                self._advance()
            elif char == ']':
                self._add_token(TokenType.RIGHT_BRACKET, ']', start_line, start_column)
                self._advance()
            elif char == ':':
                self._add_token(TokenType.COLON, ':', start_line, start_column)
                self._advance()
            elif char == ',':
                self._add_token(TokenType.COMMA, ',', start_line, start_column)
                self._advance()
            elif char == '"':
                self._tokenize_string(start_line, start_column, '"')
            elif self.auto_fix_mode and char == "'":
                self._log_message(f"Interpreting single-quoted string as double-quoted at line {start_line}, column {start_column}")
                self._tokenize_string(start_line, start_column, "'")
            elif char.isdigit() or \
                 (char == '-' and (self._peek().isdigit() or self._peek() == '.' or self._peek() == '-')) or \
                 (char == '.' and self._peek().isdigit()) or \
                 (self.auto_fix_mode and char == '+' and (self._peek().isdigit() or self._peek() == '.')):
                
                if char == '-' and self._peek() == '-' and self.auto_fix_mode:
                    self._advance(2)  # Skip both minus signs
                    self._tokenize_number(start_line, start_column, is_positive_sign_enforced=True)
                else:
                    self._tokenize_number(start_line, start_column)
            elif char.isalpha() or char == '_':
                self._tokenize_identifier_or_keyword(start_line, start_column)
            elif char == '/' and self.auto_fix_mode:
                if self._peek() == '/':
                    self._skip_single_line_comment(start_line, start_column)
                elif self._peek() == '*':
                    self._skip_multi_line_comment(start_line, start_column)
                else:
                    self._log_message(f"Unexpected character '{char}' at line {start_line}, column {start_column}")
                    self._add_token(TokenType.IDENTIFIER, char, start_line, start_column)
                    self._advance()
            else:
                self._log_message(f"Unexpected character '{char}' at line {start_line}, column {start_column}")
                self._add_token(TokenType.IDENTIFIER, char, start_line, start_column)
                self._advance()
                
        self._add_token(TokenType.EOF, None, self.line, self.column)
        return self.tokens, self.messages
    
    def _current_char(self) -> str:
        """
        Get the character at the current position.
        """
        if self.position >= len(self.text):
            return ''
        return self.text[self.position]
    
    def _peek(
            self, 
            offset: int = 1
        ) -> str:
        """
        Look ahead at a character without advancing the position.
        """
        pos = self.position + offset
        if pos >= len(self.text):
            return ''
        return self.text[pos]
    
    def _advance(
            self, 
            count: int = 1
        ):
        """
        Advance the position by the specified count.
        """
        for _ in range(count):
            if self.position < len(self.text):
                if self._current_char() == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.position += 1
            else:
                break
    
    def _add_token(
        self, 
        type: TokenType, 
        value: Any,
        line: int,
        column: int,
    ):
        """
        Add a token to the tokens list.
        """
        self.tokens.append(Token(type, value, line, column))
    
    def _log_message(
            self, 
            message: str
        ):
        """
        Log a message during tokenization.
        """
        self.messages.append(f"lexer-level fix: {message}")
    
    def _handle_whitespace(self):
        """
        Skip whitespace characters.
        """
        while self.position < len(self.text) and self._current_char().isspace():
            self._advance()
    
    def _skip_single_line_comment(
            self, 
            start_line: int, 
            start_column: int
        ):
        """
        Skip a single-line comment in auto-fix mode.
        """
        self._advance(2)  # Skip //
        while self.position < len(self.text) and self._current_char() != '\n':
            self._advance()
        self._log_message(f"Skipped single-line comment at line {start_line}, column {start_column}")
    
    def _skip_multi_line_comment(
            self, 
            start_line: int, 
            start_column: int
        ):
        """
        Skip a multi-line comment in auto-fix mode.
        """
        self._advance(2)  # Skip /*
        comment_end_found = False
        while self.position < len(self.text):
            if self._current_char() == '*' and self._peek() == '/':
                self._advance(2)
                comment_end_found = True
                self._log_message(f"Skipped multi-line comment from line {start_line}, column {start_column} to line {self.line}, column {self.column}")
                return
            self._advance()
        
        if not comment_end_found:
            self._log_message(f"Unterminated multi-line comment starting at line {start_line}, column {start_column}")
    
    def _tokenize_string(
            self, 
            start_line: int, 
            start_column: int, 
            quote_char: str
        ):
        """
        Tokenize a string value, with support for triple-quoted strings.
        """
        self._advance()  # Skip opening quote
        
        # Check for triple quotes (Python-style multiline strings)
        is_triple_quoted = False
        if self.position + 1 < len(self.text) and self._current_char() == quote_char and self._peek() == quote_char:
            # We have detected the start of a triple-quoted string
            self._advance(2)  # Skip the next two quotes
            is_triple_quoted = True
            if self.auto_fix_mode:
                self._log_message(f"Detected Python-style triple-quoted string at L{start_line}C{start_column}")
        
        string_value = ""
        escaped = False
        
        while self.position < len(self.text):
            char_to_process = self._current_char()
            current_line_in_loop = self.line
            current_col_in_loop = self.column

            if escaped:
                # Handle escape sequences (as before)
                # Existing code for handling escaped characters
                escape_char_col = current_col_in_loop # Column of the char after '\'

                if char_to_process in '"\\bfnrt/\'':
                    if char_to_process == 'b': string_value += '\b'
                    elif char_to_process == 'f': string_value += '\f'
                    elif char_to_process == 'n': string_value += '\n'
                    elif char_to_process == 'r': string_value += '\r'
                    elif char_to_process == 't': string_value += '\t'
                    elif char_to_process == quote_char: string_value += quote_char
                    elif char_to_process == '\\': string_value += '\\'
                    elif char_to_process == '/': string_value += '/'
                    elif char_to_process == "'": string_value += "'" 
                    elif char_to_process == '"': string_value += '"'
                    else: string_value += char_to_process 
                elif char_to_process == 'u':
                    hex_digits_list = []
                    for _ in range(4):
                        self._advance() 
                        if self.position >= len(self.text): break 
                        
                        hex_char_candidate = self._current_char()
                        if hex_char_candidate in self.HEX_DIGITS:
                            hex_digits_list.append(hex_char_candidate)
                        else:
                            self.position -= 1
                            self.column -= 1
                            if self.text[self.position] == '\n': 
                                self.line -= 1
                                last_nl_idx = self.text.rfind('\n', 0, self.position)
                                self.column = (self.position - last_nl_idx) if last_nl_idx != -1 else (self.position + 1)
                            break 
                    
                    hex_digits_str = "".join(hex_digits_list)
                    
                    if len(hex_digits_str) == 4:
                        try:
                            string_value += chr(int(hex_digits_str, 16))
                        except ValueError: 
                            self._log_message(f"Invalid Unicode value in \\u{hex_digits_str} at L{current_line_in_loop}C{escape_char_col}")
                            string_value += '\uFFFD'
                            self._log_message(f"  Replaced with U+FFFD due to invalid hex value.")
                    else:
                        self._log_message(f"Incomplete Unicode escape sequence \\u{hex_digits_str} (expected 4 hex digits) at L{current_line_in_loop}C{escape_char_col}")
                        string_value += '\uFFFD' 
                        self._log_message(f"  Replaced with U+FFFD (Unicode Replacement Character).")
                else: 
                    self._log_message(f"Invalid escape sequence '\\{char_to_process}' at L{current_line_in_loop}C{escape_char_col-1}")
                    if self.auto_fix_mode:
                        string_value += char_to_process 
                    else:
                        string_value += '\\' + char_to_process 
                
                escaped = False
                self._advance()

            elif char_to_process == '\\':
                escaped = True
                self._advance() 
            elif char_to_process == quote_char:
                # Triple-quoted string end detection
                if is_triple_quoted:
                    # Check if we have three matching quotes in a row
                    if (self.position + 2 < len(self.text) and 
                        self._current_char() == quote_char and 
                        self._peek() == quote_char and 
                        self._peek(2) == quote_char):
                        # End of triple-quoted string
                        self._advance(3)  # Skip all three quotes
                        self._add_token(TokenType.STRING, string_value, start_line, start_column)
                        return
                    else:
                        # It's just a single quote in the string, not the end
                        string_value += char_to_process
                        self._advance()
                        continue
                
                # Regular string terminator handling (existing code)
                # Auto-fix for unescaped quotes within quotes
                if self.auto_fix_mode and self._peek() != '':  # Not at end of text
                    look_ahead = 1
                    
                    # Skip whitespace when looking ahead
                    while self.position + look_ahead < len(self.text) and self.text[self.position + look_ahead].isspace():
                        look_ahead += 1
                    
                    # Check for missing comma pattern
                    if self.position + look_ahead < len(self.text):
                        next_char = self.text[self.position + look_ahead]
                        
                        # If quote is followed by colon, it's likely a key in a new property
                        if next_char == ':' or (
                            next_char == quote_char and 
                            self.position + look_ahead + 1 < len(self.text) and
                            any(c == ':' for c in self.text[self.position + look_ahead + 1:self.position + look_ahead + 15])
                        ):
                            # This is likely a new property key missing a comma
                            self._advance()
                            self._add_token(TokenType.STRING, string_value, start_line, start_column)
                            return
                    
                    # Regular string terminator check
                    next_char = self.text[self.position + look_ahead] if self.position + look_ahead < len(self.text) else ''
                    if next_char in ',:}]' or next_char == '':
                        # This is likely the actual string terminator
                        self._advance()  
                        self._add_token(TokenType.STRING, string_value, start_line, start_column)
                        return
                    else:
                        # This quote should be escaped
                        self._log_message(f"Auto-escaped {quote_char} within string at L{current_line_in_loop}C{current_col_in_loop}")
                        string_value += quote_char
                        self._advance()
                        continue
                
                # Normal case (not auto-fixing or this is the actual end)
                self._advance()  
                self._add_token(TokenType.STRING, string_value, start_line, start_column)
                return
            
            # NEW CODE: Handle unterminated strings intelligently by detecting JSON structural elements
            elif self.auto_fix_mode and char_to_process in '{}[]:,' and not escaped:
                # Check if this structural character is likely the end of the string
                # We need to look ahead to see if this is part of a pattern indicating a new JSON element
                
                # Common patterns after a string should end:
                # 1. ," - Start of another key-value pair
                # 2. }: - End of object, start of another property
                # 3. }, - End of object in an array
                # 4. }} - Nested object closings
                # 5. }] - End of object in array
                
                if char_to_process == ',':
                    # Is this comma followed by a quote or whitespace then quote?
                    lookahead_pos = self.position + 1
                    while lookahead_pos < len(self.text) and self.text[lookahead_pos].isspace():
                        lookahead_pos += 1
                    
                    if lookahead_pos < len(self.text) and self.text[lookahead_pos] in '"\'':
                        # This looks like a comma followed by the start of a new string (likely a new key)
                        self._log_message(f"Unterminated string - detected pattern indicating new key-value pair at L{current_line_in_loop}C{current_col_in_loop}")
                        self._add_token(TokenType.STRING, string_value, start_line, start_column)
                        return
                
                elif char_to_process in ']}':
                    # This is likely the end of an object or array containing this string
                    self._log_message(f"Unterminated string - detected closing bracket/brace '{char_to_process}' at L{current_line_in_loop}C{current_col_in_loop}")
                    self._add_token(TokenType.STRING, string_value, start_line, start_column)
                    return
                
                elif char_to_process == ':':
                    # If we find a colon and we have a surrounding quote detected earlier, this could be a new key-value pair
                    lookahead = 20  # Look ahead a reasonable number of characters
                    look_text = self.text[max(0, self.position-lookahead):self.position]
                    
                    # If we find an open quote without a matching close quote before the colon
                    if '"' in look_text or "'" in look_text:
                        last_quote = max(look_text.rfind('"'), look_text.rfind("'"))
                        if last_quote >= 0:
                            # This pattern suggests we're looking at a new key-value pair
                            self._log_message(f"Unterminated string - detected pattern indicating new key-value pair at L{current_line_in_loop}C{current_col_in_loop}")
                            self._add_token(TokenType.STRING, string_value, start_line, start_column)
                            return
                
                # If we're inside an object or array, key-value pairs pattern might be another signal
                pattern_size = 25  # Reasonable size to detect patterns
                lookahead_text = self.text[self.position:min(len(self.text), self.position + pattern_size)]
                
                # Look for patterns like: ,"key": or ,"key" : indicating a new key-value pair
                if re.search(r',\s*["\']\w+["\']?\s*:', lookahead_text):
                    self._log_message(f"Unterminated string - detected likely start of new property at L{current_line_in_loop}C{current_col_in_loop}")
                    self._add_token(TokenType.STRING, string_value, start_line, start_column)
                    return
                
                # Handle embedded structural characters - don't treat them as signals unless part of a pattern
                string_value += char_to_process
                self._advance()

            elif 0 <= ord(char_to_process) <= 0x1F and not escaped:  
                if self.auto_fix_mode:
                    desc = {'\n': 'newline', '\r': 'CR', '\t': 'tab', '\b': 'BS', '\f': 'FF'}.get(char_to_process, f"U+{ord(char_to_process):04X}")
                    escape_map = {'\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f'}
                    string_value += escape_map.get(char_to_process, f"\\u{ord(char_to_process):04x}")
                    self._log_message(f"Auto-escaped {desc} in string at L{current_line_in_loop}C{current_col_in_loop}")
                elif quote_char == '"':
                    self._log_message(f"Unescaped control character U+{ord(char_to_process):04X} in string at L{current_line_in_loop}C{current_col_in_loop}. Invalid JSON.")
                    string_value += char_to_process 
                else: 
                    string_value += char_to_process
                self._advance()
            else: 
                string_value += char_to_process
                self._advance()
        
        self._log_message(f"Unterminated string (EOF, expected {quote_char}) starting at L{start_line}C{start_column}")
        self._add_token(TokenType.STRING, string_value, start_line, start_column)
    
    def _tokenize_number(
            self, 
            start_line: int, 
            start_column: int, 
            is_positive_sign_enforced: bool = False,
        ):
        """Tokenize a number value."""
        num_text_start_pos = self.position

        # Check for malformed numbers with multiple decimal points first
        if self.auto_fix_mode:
            _current_pos_for_dot_check = num_text_start_pos
            if self.text[_current_pos_for_dot_check] in '+-': _current_pos_for_dot_check += 1
            
            dot_count = 0
            _scan_idx = _current_pos_for_dot_check
            while _scan_idx < len(self.text) and (self.text[_scan_idx].isdigit() or self.text[_scan_idx] == '.'):
                if self.text[_scan_idx] == '.': dot_count += 1
                _scan_idx += 1
            
            if dot_count > 1:
                malformed_str = self.text[num_text_start_pos:_scan_idx]
                self._advance(len(malformed_str))
                self._log_message(f"Malformed number with multiple decimal points: '{malformed_str}'. Treating as string.")
                self._add_token(TokenType.STRING, malformed_str, start_line, start_column)
                return

        # Determine actual sign and position after sign
        actual_sign = 1
        pos_after_sign = self.position

        if is_positive_sign_enforced:
            actual_sign = 1
            if self.auto_fix_mode:
                self._log_message(f"Converted double negative (--) to positive at line {start_line}, column {start_column}")
        elif self._current_char() == '-':
            actual_sign = -1
            self._advance()
            pos_after_sign = self.position
        elif self.auto_fix_mode and self._current_char() == '+':
            actual_sign = 1
            self._advance()
            pos_after_sign = self.position
            self._log_message(f"Removed leading + sign from number at line {start_line}, column {start_column}")
        
        # Handle hex and binary notations
        if self.auto_fix_mode and self._current_char() == '0' and self.position + 1 < len(self.text):
            next_char_peek = self._peek().lower()
            hex_scan_start_pos = self.position

            if next_char_peek == 'x':
                self._advance(2)  # Skip '0x'
                hex_digits_only_start = self.position
                while self.position < len(self.text) and self._current_char().lower() in self.HEX_DIGITS:
                    self._advance()
                    
                if self.position > hex_digits_only_start:
                    hex_magnitude_str = self.text[hex_scan_start_pos:self.position]
                    try:
                        value = int(hex_magnitude_str, 16)
                        self._add_token(TokenType.NUMBER, value * actual_sign, start_line, start_column)
                        self._log_message(f"Converted hexadecimal '{hex_magnitude_str}' to {value * actual_sign}")
                        return
                    except ValueError:
                        self._log_message(f"Invalid hexadecimal number: {hex_magnitude_str}")
                        self._add_token(TokenType.STRING, self.text[num_text_start_pos:self.position], start_line, start_column)
                        return
                else:
                    self.position = hex_scan_start_pos
            
            elif next_char_peek == 'b':
                bin_scan_start_pos = self.position
                self._advance(2)  # Skip '0b'
                bin_digits_only_start = self.position
                while self.position < len(self.text) and self._current_char() in '01':
                    self._advance()
                    
                if self.position > bin_digits_only_start:
                    bin_magnitude_str = self.text[bin_scan_start_pos:self.position]
                    try:
                        value = int(bin_magnitude_str, 2)
                        self._add_token(TokenType.NUMBER, value * actual_sign, start_line, start_column)
                        self._log_message(f"Converted binary '{bin_magnitude_str}' to {value * actual_sign}")
                        return
                    except ValueError:
                        self._log_message(f"Invalid binary number: {bin_magnitude_str}")
                        self._add_token(TokenType.STRING, self.text[num_text_start_pos:self.position], start_line, start_column)
                        return
                else:
                    self.position = bin_scan_start_pos

        # Check for missing leading digit with decimal point
        if self._current_char() == '.' and self._peek().isdigit():
            if self.auto_fix_mode:
                self._log_message(f"Added leading zero to decimal number at line {start_line}, column {start_column}")
                fraction_part_start = self.position
                self._advance()
                while self.position < len(self.text) and self._current_char().isdigit():
                    self._advance()
                
                fraction_str = self.text[fraction_part_start:self.position]
                parsed_magnitude = float('0' + fraction_str)
                self._add_token(TokenType.NUMBER, parsed_magnitude * actual_sign, start_line, start_column)
                return
            else:
                self._add_token(TokenType.IDENTIFIER, '.', start_line, start_column)
                self._advance()
                return
        
        # Integer part
        while self.position < len(self.text) and self._current_char().isdigit():
            self._advance()
        
        # Decimal part
        if self._current_char() == '.':
            if self._peek().isdigit() or self.auto_fix_mode:
                self._advance()
                
                decimal_digits_start = self.position
                while self.position < len(self.text) and self._current_char().isdigit():
                    self._advance()
                    
                if decimal_digits_start == self.position:
                    if self.auto_fix_mode:
                        self._log_message(f"Added trailing zero to number with decimal point at line {start_line}, column {start_column}")
                    else:
                        self.position -= 1
        
        # Exponent part
        _end_slice_for_magnitude = self.position

        if self.position < len(self.text) and self._current_char().lower() == 'e':
            is_valid_exponent_context = (self.position > pos_after_sign and
                                        self.position > 0 and
                                        self.text[self.position-1].isdigit())

            if is_valid_exponent_context:
                e_char_actual_pos = self.position
                self._advance()
                
                if self.position < len(self.text) and self._current_char() in '+-':
                    self._advance()
                
                exponent_digits_start_pos = self.position
                while self.position < len(self.text) and self._current_char().isdigit():
                    self._advance()
                
                if self.position == exponent_digits_start_pos:
                    if self.auto_fix_mode:
                        invalid_exponent_str = self.text[e_char_actual_pos:self.position]
                        self._log_message(f"Invalid number exponent format: '{invalid_exponent_str}'. Stripping invalid exponent.")
                        _end_slice_for_magnitude = e_char_actual_pos
                    else:
                        self.position = e_char_actual_pos
                        _end_slice_for_magnitude = e_char_actual_pos
                else:
                    _end_slice_for_magnitude = self.position
        
        # Construct the magnitude string
        magnitude_str = self.text[pos_after_sign:_end_slice_for_magnitude]
        
        if self.auto_fix_mode:
            if magnitude_str.startswith('.'):
                magnitude_str = '0' + magnitude_str
            
            if magnitude_str.endswith('.') and magnitude_str != ".":
                magnitude_str += '0'
        
        try:
            if not magnitude_str:
                if self.auto_fix_mode:
                    self._log_message(f"Isolated sign character '{self.text[num_text_start_pos:self.position]}' treated as string.")
                    self._add_token(TokenType.STRING, self.text[num_text_start_pos:self.position], start_line, start_column)
                else:
                    self._add_token(TokenType.IDENTIFIER, self.text[num_text_start_pos:self.position], start_line, start_column)
                return
            
            if magnitude_str == "." and not self.auto_fix_mode:
                self._add_token(TokenType.IDENTIFIER, ".", start_line, start_column)
                return

            # Parse the number
            value_numeric = 0
            if '.' in magnitude_str or 'e' in magnitude_str.lower():
                value_numeric = float(magnitude_str)
            else:
                if len(magnitude_str) > 1 and magnitude_str.startswith('0') and self.auto_fix_mode:
                    self._log_message(f"Removed leading zeros from number '{magnitude_str}' at line {start_line}, column {start_column}")
                value_numeric = int(magnitude_str)
            
            self._add_token(TokenType.NUMBER, value_numeric * actual_sign, start_line, start_column)
        except ValueError:
            full_num_token_text = self.text[num_text_start_pos:self.position]
            if self.auto_fix_mode:
                self._log_message(f"Could not parse number '{full_num_token_text}'. Treating as string.")
                self._add_token(TokenType.STRING, full_num_token_text, start_line, start_column)
            else:
                self._log_message(f"Invalid number sequence: '{full_num_token_text}' at L{start_line}C{start_column}. Tokenizing as IDENTIFIER.")
                self._add_token(TokenType.IDENTIFIER, full_num_token_text, start_line, start_column)

    def _tokenize_identifier_or_keyword(
            self, 
            start_line: int, 
            start_column: int
        ):
        """
        Tokenize an identifier or keyword.
        """
        start_pos = self.position
        
        while (self.position < len(self.text) and
               (self._current_char().isalnum() or self._current_char() == '_')):
            self._advance()
        
        identifier = self.text[start_pos:self.position]
        
        # Handle exact keywords
        if identifier == "true":
            self._add_token(TokenType.TRUE, True, start_line, start_column)
        elif identifier == "false":
            self._add_token(TokenType.FALSE, False, start_line, start_column)
        elif identifier == "null":
            self._add_token(TokenType.NULL, None, start_line, start_column)
        elif self.auto_fix_mode:
            # Check for misspelled keywords in auto-fix mode
            identifier_lower = identifier.lower()
            
            # Handle common misspellings
            if identifier_lower in self.TRUE_VARIANTS:
                self._add_token(TokenType.TRUE, True, start_line, start_column)
                self._log_message(f"Fixed misspelled 'true' keyword: '{identifier}' at line {start_line}, column {start_column}")
            elif identifier_lower in self.FALSE_VARIANTS:
                self._add_token(TokenType.FALSE, False, start_line, start_column)
                self._log_message(f"Fixed misspelled 'false' keyword: '{identifier}' at line {start_line}, column {start_column}")
            elif identifier_lower in self.NULL_VARIANTS:
                self._add_token(TokenType.NULL, None, start_line, start_column)
                self._log_message(f"Fixed misspelled 'null' keyword: '{identifier}' at line {start_line}, column {start_column}")
            # Handle JavaScript specific literals
            elif identifier_lower == "undefined":
                self._add_token(TokenType.NULL, None, start_line, start_column)
                self._log_message(f"Converted JavaScript '{identifier}' to null at line {start_line}, column {start_column}")
            elif identifier_lower == "nan":
                self._add_token(TokenType.NUMBER, 0, start_line, start_column)
                self._log_message(f"Converted JavaScript '{identifier}' to 0 at line {start_line}, start_column")
            elif identifier_lower == "infinity":
                self._add_token(TokenType.NUMBER, float('inf'), start_line, start_column)
                self._log_message(f"Converted JavaScript '{identifier}' to float('inf') at line {start_line}, column {start_column}")
            elif identifier_lower == "-infinity":
                self._add_token(TokenType.NUMBER, float('-inf'), start_line, start_column)
                self._log_message(f"Converted JavaScript '-{identifier}' to float('-inf') at line {start_line}, column {start_column}")
            else:
                # Regular identifier (treated as unquoted string)
                self._add_token(TokenType.IDENTIFIER, identifier, start_line, start_column)
        else:
            # In strict mode, any non-keyword is an identifier
            self._add_token(TokenType.IDENTIFIER, identifier, start_line, start_column)