"""
Here is the caption class that will be used to generate the caption for the image
"""
import time
import torch
import os
import threading
import torch

from   base64       import b64encode
from   typing       import List, Union
from   PIL          import Image
from   pathlib      import Path
from   transformers import BitsAndBytesConfig, StoppingCriteria, GenerationConfig, AutoModelForCausalLM
from   transformers import AutoProcessor, AutoModelForVision2Seq, BlipForConditionalGeneration, AutoTokenizer, AutoImageProcessor
from   image_tools  import ImageTools
from   utils        import debug_print

class ImageCaptioner:
    llava_model_id_path: str = "mistral-community/pixtral-12b"
    molmo_model_id_path: str = "allenai/Molmo-7B-D-0924"
    xgen_model_id_path: str  = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
    blip_model_id_path: str  = "Salesforce/blip-image-captioning-base"
    max_num_tokens: int      = 128
    bnb_config               = BitsAndBytesConfig( load_in_8bit = True)
    target_model             = "xgen"
    valid_models             = ("blip",  "xgen", "llava", "molmo")

    class EosListStoppingCriteria(StoppingCriteria):
        def __init__(self, eos_sequence = [32007]):
            self.eos_sequence = eos_sequence

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
            return self.eos_sequence in last_ids    
                
    def __init__(self, image_tools: ImageTools):
        self.image_tools               = image_tools
        self.model                     = None
        self.tokenizer                 = None
        self.image_processor           = None
        self.model_loaded              = threading.Event()
        self.model_loadeding           = False
        self.model_loadeding_completed = False
        self.user_prompt               = ""
        self.prompt                    = ""
        self.enabled_flag              = False
        self.caption                   = ""

    def set_enabled_flag(self, enabled_flag: bool) -> None:
        self.enabled_flag = enabled_flag
        if enabled_flag and not self.model_loaded.is_set():
            self.load_multi_modal_model_background()
        return


    def generate_prompt_txt(self, file_name:str=None, dir_name:str=None) -> None:
        """Define the prompt template for captionning"""
        file_context = f"File name: {file_name}\n" if file_name else ""
        dir_context  = f"Directory name: {dir_name}\n" if dir_name else ""
        self.prompt = f"{file_context}{dir_context}{self.user_prompt}"
        if self.target_model == "llava" or self.target_model == "molmo":
            self.prompt =  f"<s>[INST]{self.user_prompt} {file_context} {dir_context}\n[IMG][/INST]"
        elif self.target_model == "xgen":
            self.prompt =  (
             "<|system|>\nYou are an image caption generator. Provide image caption only with tags separated by commas. "
             "If a person's name, style, object or activity is mentioned by the user, make sure to include it in your caption<|end|>\n"
            f'<|user|>\n<image>\n{self.user_prompt} {file_context} {dir_context}<|end|>\n<|assistant|>\n'
        )
        elif self.target_model == "molmo":
            self.prompt =  f"<s>[INST]{self.user_prompt} {file_context} {dir_context}\n[/INST]"

        debug_print(f"Prompt: {self.prompt}")

    def load_multi_modal_model_background(self) -> None:
        """Start loading the model in a background thread"""
        if not self.model_loaded.is_set() and not self.model_loadeding:
            self.model_loadeding = True
            thread               = threading.Thread(target=self._load_multi_modal_model)
            thread.start()

    def _load_multi_modal_model(self) -> None:
        """Load the provided model"""
        debug_print(f"Loading {self.target_model} type model in the background")
        if self.target_model == "llava":
            self.model_id_path   = self.llava_model_id_path
            self.model           = AutoModelForVision2Seq.from_pretrained(
                                        self.model_id_path, 
                                        low_cpu_mem_usage   = True, 
                                        quantization_config = self.bnb_config, 
                                        torch_dtype         = torch.float16,
                                        device_map          = 'auto',
                                        )
            #self.tokenizer       = AutoProcessor.from_pretrained(self.model_id_path)
            self.image_processor = AutoProcessor.from_pretrained(self.model_id_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto') 
        elif self.target_model == "xgen":
            self.model_id_path   = self.xgen_model_id_path  
            # load models
            model                = AutoModelForVision2Seq.from_pretrained(self.model_id_path, trust_remote_code=True)
            tokenizer            = AutoTokenizer.from_pretrained(self.model_id_path, trust_remote_code=True, use_fast=False, legacy=False)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_id_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
            self.tokenizer       = model.update_special_tokens(tokenizer)
            self.model           = model.cuda()

        elif self.target_model == "blip":
            self.model_id_path   = self.blip_model_id_path
            self.model           = BlipForConditionalGeneration.from_pretrained(
                                        self.model_id_path, 
                                        low_cpu_mem_usage   = True, 
                                        quantization_config = self.bnb_config, 
                                        torch_dtype         = torch.float16,
                                        #device_map          = 'auto', Dosen not work for BLIP!
                                        )
            self.image_processor = AutoProcessor.from_pretrained(self.model_id_path, torch_dtype=torch.bfloat16) #, device_map='auto') 
        elif self.target_model == "molmo":
            self.model_id_path   = self.molmo_model_id_path
            self.model           = AutoModelForCausalLM.from_pretrained(
                                        self.model_id_path, 
                                        trust_remote_code   = True,
                                        low_cpu_mem_usage   = True,
                                        torch_dtype         = torch.bfloat16, #'auto',
                                        device_map          = 'auto',
                                        #quantization_config = self.bnb_config, 
                                        )
            self.image_processor = AutoProcessor.from_pretrained(self.model_id_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto') 
        else:
            debug_print(f"Invalid model type selected: {self.target_model}")
            return
        debug_print(f"Model {self.model_id_path} loaded successfully")
        self.model_loaded.set()
        self.model_loadeding_completed = True

    def wait_for_model(self) -> None:
        """Wait for the model to be loaded"""
        self.model_loaded.wait()

    def set_user_prompt(self, user_prompt):
        self.user_prompt = user_prompt
        self.generate_prompt_txt()

    def generate_image_caption(self, file_name_in_caption:bool=False, dir_name_in_caption:bool=False) -> str:
        """Function to evaluate the model"""
        if not self.model_loaded.is_set():
            print("Image processor is not yet fully initialized waiting a bit.")
            while not self.model_loaded.is_set():
                print(".", end="")
                time.sleep(10)

        if file_name_in_caption:
            file_name = self.image_tools.get_file_name()
            dir_name  = self.image_tools.get_dir_name()

        self.image_to_caption = self.image_tools.get_down_sampled_image(512, smallest_side=False, convert2RGB = False)  
        debug_print(f"Image type: {type(self.image_to_caption)}")

        debug_print(f"Processing image. Size: {self.image_to_caption.width}x{self.image_to_caption.height}")
        debug_print(f"generating {self.target_model} model inputs")
        if self.target_model == "llava":
            debug_print("llava and molmo")
            inputs                = self.image_processor(text=self.prompt,images=[self.image_to_caption], return_tensors="pt").to("cuda")

        elif self.target_model == "xgen":
            inputs          = self.image_processor([self.image_to_caption], return_tensors="pt", image_aspect_ratio='anyres')
            language_inputs = self.tokenizer([self.prompt], return_tensors="pt")
            inputs.update(language_inputs)
            inputs          = {name: tensor.cuda() for name, tensor in inputs.items()}

        elif self.target_model == "blip":
            inputs                = self.image_processor(images=self.image_to_caption, return_tensors="pt").to("cuda")

        elif self.target_model == "molmo":
            inputs                = self.image_processor.process(images=[self.image_to_caption], text=self.prompt )
            inputs                = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}


            
        debug_print(f"Input generated inputs length: {len(inputs)}")

        if self.target_model == "llava":
            generated_tokens = self.model.generate(
                                            **inputs,
                                            pad_token_id       = self.image_processor.tokenizer.eos_token_id,
                                            #do_sample         = True,
                                            max_new_tokens     = self.max_num_tokens,
                                            repetition_penalty = 1.5
                                            #top_p             = 0.95,
                                            #top_k             = 50,
                                            #num_beams         = 1,
                                        )
        elif self.target_model == "xgen":
            generated_tokens = self.model.generate(
                                            **inputs,
                                            image_size         = [self.image_to_caption.size],
                                            max_new_tokens     = self.max_num_tokens,
                                            do_sample          = True,
                                            repetition_penalty = 1.5,
                                            top_p              = 0.95,
                                            top_k              = 50,
                                            num_beams          = 1,
                                            stopping_criteria  = [self.EosListStoppingCriteria()],
                                        )
        elif self.target_model == "blip":
            generated_tokens = self.model.generate(
                                            **inputs,
                                            max_new_tokens=self.max_num_tokens,
                                            repetition_penalty=1.5
                                            )
        elif self.target_model == "molmo":
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_tokens = self.model.generate_from_batch( 
                                            batch              = inputs,  # Pass 'inputs' as the 'batch' argument
                                            #max_new_tokens     = self.max_num_tokens,
                                            #repetition_penalty = 1.5,
                                            tokenizer          = self.image_processor.tokenizer,
                                            generation_config  = GenerationConfig(max_new_tokens=self.max_num_tokens, stop_strings="<|endoftext|>"),            
                                        )
        debug_print(f"Number of generated tokens: {len(generated_tokens[0])}")

        if self.target_model == "llava":
            self.caption = self.image_processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        elif self.target_model == "xgen":
            self.caption = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        elif self.target_model == "blip":
            self.caption = self.image_processor.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        elif self.target_model == "molmo":
            generated_tokens_cln = generated_tokens[0,inputs['input_ids'].size(1):]
            self.caption     = self.image_processor.tokenizer.decode(generated_tokens_cln, skip_special_tokens=True)
            #self.caption     = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        debug_print(f"Proposed captioning: {self.caption[1:]}")
        if self.user_prompt in self.caption:
            self.caption = self.caption.replace(self.user_prompt, "")
        if "<|end|>" in self.caption:
            self.caption = self.caption.split("<|end|>")[0]

        return self.caption

  
    def save_caption(self, image_path: Union[str, os.PathLike]) -> str:
        """Saves the caption with the same filename as the image and a .txt extension."""
        caption_path = Path(image_path).with_suffix('.txt')
        debug_print(f"Saving caption to: {caption_path}")
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(self.caption)
        debug_print("Caption saved successfully")
        return str(caption_path)

    def set_caption_model(self, model_type: str) -> None:
        """
        Sets the caption model type based on user selection.
        """
        
        model_type   = model_type.lower()   # put modeltype to lower case
        
        
        if model_type not in ImageCaptioner.valid_models:
            debug_print(f"Invalid model type selected: {model_type}")
            return
        
        self.target_model = model_type
        debug_print(f"Caption model set to: {model_type}")
        

