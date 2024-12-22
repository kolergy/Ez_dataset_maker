# Easy Dataset Maker

This tool ams at making it easy to generate image datasets for model training for stable difusion or flux or others

For now it is able to
 - Select image in the working folder and sub folder (need to improve that)
 - Define a target size to be applied to small size or large size (this will keep the original image aspect ratio, most model just have size limitation )
 - Select an output format (most model need all images to be the same format)
 - Simple cropping abilities
 - Ability to generate caption text for the images  with the folowing models:
    - blip  - The good old blip model. It has a low VRAM requirement but dose not use prompt
    - xgen  - The new name of blip-3 much more capable  
    - lava  - The lava type model for now is pixtral 12b (running on 8 bit but sill use a lot of VRAM)
    - molmo - The Molmo 7b model use massive ammount of VRAM ~20Gb (specially because I have not yet been able to run it in 8bit) but give excellent rtesults

The tool will generate an Output directory where your images are. It will store there the resized images and the captions 

It is possible to add the image's file name and or directory name in the context if those bring sementic value.
It is able to handle very large image files.

Test Images are from Wikimedia Commons

## Todo:
 - Add the possibility to unload models and reload new ones
 - Find a way to load Molmo in 8b or less
 - Add the possibility of images viewing and zooming
 - Improve the possiblilty of image cropping 




 ## Instalation:

 `git clone https://github.com/kolergy/Ez_dataset_maker.git`

 `cd Ez_dataset_maker`

 `conda create -y -n EDM_env python=3.12`

 `conda activate EDM_env`

 `pip install -U -r requirements.txt`


 ## Useage:

 `conda activate EDM_env`

 `python ui.py`

 Select the images files you want in your dataset

 If you have super large files ~10k piwels Select Handle very large images 

 If you want captionning select the caption model then select generate caption

*Warning:* once you select generate caption the selected model will load and for now it is not yet possible to unload a model while the code is running so to change the model you need to stop the code and restart it.  

Tha caption prompt will feed the prompt for all the models except for the blip model whch do not use a prompt
