"""
Here are all the means to manipulmate the dataset

"""


import os
from   typing      import List, Any, Generator, Tuple, Dict
from   pathlib     import Path
from   PIL         import Image
from   image_tools import ImageTools
from   caption     import ImageCaptioner
from   utils       import debug_print

class DatasetHandler:
    def __init__(self):
        self.file_list: List[str]          = []
        self.handle_very_large_image: bool = False
        self.target_size: int              = 1024
        self.smallest_side: bool           = True
        self.postfix_string: str           = "down_sampled"
        self.output_dir: str               = "Output"
        self.format: str                   = "PNG"
        self.generate_caption: bool        = False
        self.image_tools                   = ImageTools()
        self.image_captioner               = ImageCaptioner(self.image_tools)
        self.file_name_in_context: bool    = False
        self.dir_name_in_context: bool     = False
        self.crop_list: List[Dict[str, float]] = []

    def set_file_list(self, file_list: List[str]):
        self.file_list = self.clean_file_list(file_list)
        self.crop_list = [{
            "min_x": 0.0,
            "max_x": 100.0,
            "min_y": 0.0,
            "max_y": 100.0
        } for _ in self.file_list]
        
    def get_crop_values_at_index(self, index: int) -> Dict[str, float]:
        """Returns the crop values at the given index."""
        if 0 <= index < len(self.crop_list):
            return self.crop_list[index]
        return {"min_x": 0.0, "max_x": 100.0, "min_y": 0.0, "max_y": 100.0}

    def set_handle_very_large_image(self, value: bool):
        self.handle_very_large_image = value

    def set_target_size(self, value: int):
        self.target_size = value

    def set_smallest_side(self, value: bool):
        self.smallest_side = value

    def set_postfix_string(self, value: str):
        self.postfix_string = value

    def set_format(self, value: str):
        self.format = value

    def set_caption_prompt(self, user_prompt: str):
        self.image_captioner.set_user_prompt(user_prompt)

    def set_image_captioner_enabled_flag(self, enabled_flag: bool) -> None:
        if self.image_captioner:
            self.image_captioner.set_enabled_flag(enabled_flag)
        self.generate_caption = enabled_flag
        return
    
    def set_file_name_in_context(self, value: bool):
        self.file_name_in_context = value

    def set_dir_name_in_context(self, value: bool):
        self.dir_name_in_context = value

    def get_current_image(self) -> Image:
        img = self.image_tools.down_sampled
        if img:
            return img
        else:
            return self.image_tools.initial_image
            
    def load_image_at_index(self, index: int) -> Tuple[Image.Image, str, str]:
        """Loads and returns image, size and caption at given index"""
        if 0 <= index < len(self.file_list):
            file_path = self.file_list[index]
            self.image_tools.load(file_path, self.handle_very_large_image)
            self.image_tools.down_sample_fix_AR(self.target_size, self.smallest_side)
            
            current_image = self.get_current_image()
            image_size = f"{current_image.width}x{current_image.height}"
            
            # Try to read existing caption if any
            caption = ""
            caption_path = str(Path(file_path).with_suffix('.txt'))
            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    caption = f.read().strip()
                    
            return current_image, image_size, caption
        return None, "", "",

    def clean_file_list(self, input_file_list: List[str], keep_type: str = 'image') -> List[str]:
        """Strips the file list to keep only files of the desired type."""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        if keep_type == 'image':
            cleaned_list = [file for file in input_file_list if Path(file).suffix.lower() in image_extensions]
            debug_print(f"DEBUG: clean_file_list - Input files: {len(input_file_list)}, Cleaned files: {len(cleaned_list)}")
            return cleaned_list
        else: 
            debug_print(f"DEBUG: clean_file_list - Returning empty list for keep_type: {keep_type}")
            return []

    def process_image_files(self) -> Generator[Tuple[int, int, int, str], None, None]:
        """Processes the list of image files by loading, resizing, saving each image, and optionally generating a caption."""
        total_images     = len(self.file_list)
        remaining_images = total_images

        debug_print(f"Total images to process: {total_images}")

        #get the base directory of the first file
        base_dir = os.path.dirname(self.file_list[0])

        # Create the output directory in the base directory where the files are if it does not yet exist
        output_dir = os.path.join(base_dir, self.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        

        for index, file in enumerate(self.file_list):
            debug_print(f"Processing file: {file}")
            file_name = Path(file).name
            self.image_tools.load(file, self.handle_very_large_image)
            self.image_tools.down_sample_fix_AR(self.target_size, self.smallest_side)
            
            # Crop the image
            crop_values = self.crop_list[index]
            cropped_image = self.image_tools.crop_image(
                crop_values["min_x"],
                crop_values["max_x"],
                crop_values["min_y"],
                crop_values["max_y"]
            )
            self.image_tools.down_sampled = cropped_image
            
            save_path = os.path.join(output_dir, file_name)
            new_path  = self.image_tools.save(save_path, self.postfix_string, self.format)
            remaining_images -= 1
            
            debug_print(f"Image saved to: {new_path}")
            
            caption = ""
            if self.generate_caption and self.image_captioner:
                debug_print("Generating caption...")
                caption = self.image_captioner.generate_image_caption(self.file_name_in_context, self.dir_name_in_context)
                self.image_captioner.save_caption(new_path)
                debug_print(f"Caption saved to: {new_path}")
            
            debug_print(f"Remaining images: {remaining_images}")
            
            yield total_images, total_images - remaining_images, remaining_images, caption
