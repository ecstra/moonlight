import base64
from PIL import Image
from io import BytesIO

class BinaryImageProcessor:
    def __init__(
            self, 
            binary_data
        ):
        """
        Initialize with binary image data
        """
        self.binary_data = binary_data
        self._image = None
    
    def _load_image(self):
        """
        Load image from binary data
        """
        if self._image is None:
            self._image = Image.open(BytesIO(self.binary_data))
        return self._image
    
    def get_dimensions(self):
        """
        Get width and height of the binary image
        """
        img = self._load_image()
        return img.size
    
    def resize(
        self,
        width, 
        height
    ):
        """
        Resize the image to specified width and height
        """
        
        img = self._load_image()
        resized_img = img.resize((width, height), Image.LANCZOS)
        output = BytesIO()
        # Preserve original format if possible
        format_name = img.format if img.format else "PNG"
        resized_img.save(output, format=format_name)
        self.binary_data = output.getvalue()
        self._image = None  # Reset image cache
        return self.binary_data
    
    def reduce_image(self, max_dimension=500):
        """
        Reduce image size if larger than max_dimension
        """
        
        width, height = self.get_dimensions()
        max_dim = max(width, height)
        scale = max_dimension / max_dim if max_dim > max_dimension else 1.0
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img = self._load_image()
        img_low = img.resize((new_width, new_height), Image.LANCZOS)
        output = BytesIO()
        img_low.save(output, format="PNG")
        self.binary_data = output.getvalue()
        base64_data = base64.b64encode(output.getvalue()).decode('utf-8')
        b64_image = f"data:image/png;base64,{base64_data}"
        self._image = None  # Reset image cache
        
        return new_width, new_height, b64_image
    
    def to_base64(self):
        """
        Convert binary image to base64 string
        """
        img = self._load_image()
        format_name = img.format.lower() if img.format else "png"
        mime_type = f"image/{format_name}"
        base64_data = base64.b64encode(self.binary_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_data}"

    def get_format_info(self):
        """
        Returns information about the current image format
        """
        img = self._load_image()
        return {
            "format": img.format,
            "mode": img.mode,
            "info": img.info
        }