import re
from io import StringIO

class ProcessorException(Exception):
    pass

class LogFilteredStringIO(StringIO):
    def write(self, text):
        # Skip common logging patterns
        log_patterns = [
            r'^\s*logging\.',
            r'^\s*log\.',
            r'^\s*logger\.',
            r'^\s*print\s*\(["\'](?:DEBUG|INFO|WARNING|ERROR|CRITICAL)',
            r'^\s*(?:debug|info|warning|error|critical)\s*\(',
        ]
        
        if not any(re.match(pattern, text, re.IGNORECASE) for pattern in log_patterns):
            super().write(text)