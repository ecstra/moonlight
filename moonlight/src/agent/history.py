import asyncio, httpx, base64, mimetypes
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Union

from .base import Content

async def process_images(images):
    """
    Process a list of images that can be:
    - File paths: Convert to base64 with proper mimetype
    - URLs: Download and convert to base64
    - Base64 data URLs: Return as-is
    - Invalid inputs: Skip
    
    Returns a list of base64 data URLs (or None for skipped items)
    """
    async with httpx.AsyncClient(timeout=30) as client:
        async def process_one(image):
            # Check if already base64 data URL
            if isinstance(image, str) and image.startswith("data:") and ";base64," in image:
                return image
            
            # Check if it's a valid URL
            try:
                parsed = urlparse(image)
                if parsed.scheme in ("http", "https"):
                    # Download from URL
                    try:
                        r = await client.get(image)
                        r.raise_for_status()
                        mime = r.headers.get("content-type", "application/octet-stream").split(";")[0]
                        b64 = base64.b64encode(r.content).decode("ascii")
                        return f"data:{mime};base64,{b64}"
                    except Exception as e:
                        return None
            except:
                pass
            
            # Check if it's a file path
            try:
                path = Path(image)
                if path.exists() and path.is_file():
                    # Guess mimetype from file extension
                    mime = mimetypes.guess_type(str(path))[0] or ""
                    
                    # Only process if it's an image file
                    if not mime.startswith("image/"):
                        return None
                    
                    # Read file and convert to base64
                    with open(path, "rb") as f:
                        content = f.read()
                    
                    b64 = base64.b64encode(content).decode("ascii")
                    return f"data:{mime};base64,{b64}"
            except Exception as e:
                return None
            
            # Skip invalid inputs
            return None

        return await asyncio.gather(*(process_one(img) for img in images))
    
class AgentHistory:
    def __init__(
        self,
        system_role: str
    ):
        self._history: List[Dict[str, Union[str, List[str]]]] = []
        self._system_role = system_role
        self._history.append({"role": "system", "content": system_role})
    
    def get_history(self) -> List[Dict[str, Union[str, List[str]]]]: return self._history
        
    async def add(
        self, 
        role: str, 
        content: Content
    ):
        # Convert all images to base64
        # this includes downlaoding and local file processing
        if content.images and len(content.images) > 0:
            content.images = await process_images(content.images)
            # some images might be None 
            # if they are deemed to be invalid while processing,
            # so just remove them
            content.images = [i for i in content.images if i is not None]
        
            # Construct an object of all the images + text
            # this should be the format
            # "content": [
            #     {
            #         "type": "text",
            #         "text": "What's in this image?"
            #     },
            #     {
            #         "type": "image_url",
            #         "image_url": {
            #             "url": "data:image/jpeg;base64,{base64_image_data}"
            #         }
            #     },
            #     {
            #         "type": "video_url",
            #         "video_url": {
            #             "url": "data:video/mp4;base64,{base64_video_data}"
            #         }
            #     }
            # 
            
            message_content = []
            message_content.append({ "type": "text", "text": content.text })
            
            for image in content.images:
                message_content.append({ "type": "image_url", "image_url": { "url": image } })
            
            self._history.append({ "role": role, "content": message_content })
        else:
            # if there are no images
            # then a simple text without a list of wierd objects
            # should work
            self._history.append({ "role": role, "content": content.text })            

    def update_system_role(
        self, 
        system_role: str,
    ):
        self._system_role = system_role
        if len(self._history) > 0 and self._history[0].get("role") == "system":
            self._history.pop(0)
        self._history.insert(0, { "role": "system", "content": system_role })
    
    def clear_history(self):
        self._history = []
        self.update_system_role(self._system_role)

    def __str__(self) -> str:
        return str(self._history)

    def __repr__(self) -> str:
        return f"AgentHistory(system_role={self._system_role!r}, messages={len(self._history)})"