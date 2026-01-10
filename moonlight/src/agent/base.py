from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Content:
    text: str
    images: Optional[List[str]] = None
    video: Optional[str] = None