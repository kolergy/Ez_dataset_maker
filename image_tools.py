"""
Here are the image manipulation tools that will be used to process the images
"""

import os

import numpy   as np

from   typing  import Union
from   pathlib import Path
from   PIL     import Image
from   utils   import debug_print
from   io      import BytesIO
from   base64  import b64encode

class ImageTools:
    pil_max_pix: int = Image.MAX_IMAGE_PIXELS

    def __init__(self):
        self.clear_image_data()

        
    def clear_image_data(self):
        """Clears the image data."""
        self.path                    = None
        self.handle_very_large_image = False
        self.initial_image           = None
        self.down_sampled            = None


    def load(self, path: Union[str, os.PathLike], handle_very_large_image: bool = False):
        """Loads an image and sets the image and metadata."""
        self.clear_image_data()
        
        self.path                    = path
        self.handle_very_large_image = handle_very_large_image
        
        if self.handle_very_large_image:
            Image.MAX_IMAGE_PIXELS = None
            im_array               = np.array(Image.open(self.path), dtype=np.uint8)
            self.initial_image     = Image.fromarray(im_array)
        else:
            Image.MAX_IMAGE_PIXELS = self.pil_max_pix
            self.initial_image     = Image.open(self.path)

    def save(self, path: Union[str, os.PathLike], postfix_string: str, format: str) -> str:
        """Saves the image with the specified format and postfix."""
        new_path = Path(path).with_name(f"{Path(path).stem}_{postfix_string}{Path(path).suffix}")
        self.down_sampled.save(new_path, format=format, quality=95)
        return str(new_path)

    def down_sample_fix_AR(self, target_size: int, smallest_side: bool) -> None:
        """Downsamples the image while preserving the aspect ratio."""
        if smallest_side:
            if self.initial_image.width < self.initial_image.height:
                new_width  = target_size
                new_height = int((new_width / self.initial_image.width) * self.initial_image.height)
            else:
                new_height = target_size
                new_width  = int((new_height / self.initial_image.height) * self.initial_image.width)
        else:
            if self.initial_image.width > self.initial_image.height:
                new_width  = target_size
                new_height = int((new_width / self.initial_image.width) * self.initial_image.height)
            else:
                new_height = target_size
                new_width  = int((new_height / self.initial_image.height) * self.initial_image.width)

        # Downsample using bicubic resampling
        self.down_sampled = self.initial_image.resize((new_width, new_height), Image.BICUBIC)

    def get_file_name(self) -> str:
        """Returns the file name."""
        return Path(self.path).name
    
    def get_dir_name(self) -> str:
        """Returns the directory name."""
        return Path(self.path).parent.name

    def get_down_sampled_image(self, size=333, smallest_side=True, convert2RGB = False) -> Image:
        """Returns the downsampled image."""
        if not self.down_sampled:
            self.down_sample_fix_AR(size, smallest_side)
            self.convert_downsampled_image_to_rgb()
        if convert2RGB:
            return self.down_sampled.convert("RGB")
        else:
            return self.down_sampled


    def get_base64_img_string(self, size=333, smallest_side=True) -> str:
        """Returns the base64 encoded image string."""
        if not self.down_sampled:
            self.down_sample_fix_AR(size, smallest_side)
            self.convert_downsampled_image_to_rgb()

        img_bytes  = BytesIO()
        self.down_sampled.save(img_bytes, format="PNG")
        img_bytes  = img_bytes.getvalue()
        img_base64 = b64encode(img_bytes).decode("utf-8")
        return img_base64




