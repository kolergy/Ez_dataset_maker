"""
here are all the things realted to the UI of the application and ONLY the us no processing happens there!
"""


import gradio          as     gr

from   typing          import Tuple, Any, List, Generator
from   dataset_handler import DatasetHandler
from   caption         import ImageCaptioner
from   utils           import debug_print

def gradio_interface() -> None:
    
    image_dataset_handler = DatasetHandler()
    image_captioner       = image_dataset_handler.image_captioner
    valid_models          = image_captioner.valid_models
    with gr.Blocks() as demo:
        gr.Markdown("# Image Batch Downsampler and Captioner")
        with gr.Row():
            with gr.Column():
                file_explorer           = gr.FileExplorer( file_count = "multiple",                 interactive = True                           )
                handle_very_large_image = gr.Checkbox(     label      = "Handle very large images", value       = False                          )
                target_size             = gr.Number(       label      = "Target Size",              value       = 1024                           )
                smallest_side           = gr.Checkbox(     label      = "Apply to smallest side",   value       = True                           )
                postfix_string          = gr.Textbox(      label      = "Postfix String",           value       = "re_sampled"                   )
                format_selector         = gr.Dropdown(     label      = "Output Format",            choices     = ["PNG", "jpeg"], value = "PNG" )
                caption_model_selector  = gr.Dropdown(     label      = "Caption Model Type",       value       = "xgen", choices=valid_models   )                        
                caption_checkbox        = gr.Checkbox(     label      = "Generate caption",         value       = False                          )
                file_name_in_context    = gr.Checkbox(     label      = "add file name to context", value       = False                          )
                dir_name_in_context     = gr.Checkbox(     label      = "add dir name to context",  value       = False                          )
                
                caption_prompt          = gr.Textbox(      label      = "Caption prompt",           value       = "Provide a detailed image caption with tags separated by commas.")#, visible = False )
                start_button            = gr.Button(       "Start"                                                                               )

            with gr.Column():
                total_files             = gr.Textbox( label = "Number of files selected",               value="0" )
                initial_images          = gr.Textbox( label = "Initial number of images to be treated", value="0" )
                remaining_images        = gr.Textbox( label = "Remaining number of images to treat",    value="0" )
                current_image_display   = gr.Image(   label = "Current Image",                                    visible = True )
                image_size_display      = gr.Textbox( label = "Image Size",                             value="", visible = True,           interactive=False )
                caption_output          = gr.Textbox( label = "Generated Caption",                      value="", visible = False, lines=5, interactive=True  )
                caption_text_display    = gr.Textbox( label = "Caption Text",                           value="", visible = True,           interactive=False )
                console_output          = gr.Textbox( label = "Console Output",                                                    lines=5, interactive=False )

        def update_file_count(selected_files: List[str]) -> Tuple[int, int, int]:
            image_dataset_handler.set_file_list(selected_files)
            return len(selected_files), len(selected_files), len(selected_files)

        def toggle_caption_prompt(checkbox: bool) -> gr.update:
            image_dataset_handler.set_image_captioner_enabled_flag(checkbox)
            image_dataset_handler.set_caption_prompt(caption_prompt.value)
            image_dataset_handler.set_postfix_string(postfix_string.value)
            return {
                caption_prompt: gr.update(visible=checkbox),
                caption_model_selector: gr.update(interactive=not checkbox),
                caption_checkbox: gr.update(interactive=not checkbox)
            }
            
            

        def set_caption_model(model_type: str) -> None:                                                                                                                                                                                                          
            image_captioner.set_caption_model(model_type)  

        file_explorer.change(fn=update_file_count, inputs=file_explorer, outputs=[total_files, initial_images, remaining_images])
        caption_checkbox.change(fn=toggle_caption_prompt, inputs=caption_checkbox, outputs=[caption_prompt, caption_model_selector, caption_checkbox])
        
        # Update settings when UI elements change
        handle_very_large_image.change(fn=image_dataset_handler.set_handle_very_large_image     , inputs=handle_very_large_image, outputs=[])
        target_size.change(            fn=image_dataset_handler.set_target_size                 , inputs=target_size,             outputs=[])
        smallest_side.change(          fn=image_dataset_handler.set_smallest_side               , inputs=smallest_side,           outputs=[])
        postfix_string.change(         fn=image_dataset_handler.set_postfix_string              , inputs=postfix_string,          outputs=[])
        format_selector.change(        fn=image_dataset_handler.set_format                      , inputs=format_selector,         outputs=[])
        caption_checkbox.change(       fn=image_dataset_handler.set_image_captioner_enabled_flag, inputs=caption_checkbox,        outputs=[])
        caption_model_selector.change( fn=set_caption_model                                     , inputs=caption_model_selector,  outputs=[])
        caption_prompt.change(         fn=image_dataset_handler.set_caption_prompt              , inputs=caption_prompt,          outputs=[])
        file_name_in_context.change(   fn=image_dataset_handler.set_file_name_in_context        , inputs=file_name_in_context,    outputs=[])
        dir_name_in_context.change(    fn=image_dataset_handler.set_dir_name_in_context         , inputs=dir_name_in_context,     outputs=[])
        
        def initiate_image_processing() -> Generator[Tuple[int, int, int, str, str, Any, str, str], None, None]:
            """Initiates the image processing workflow and yields progress updates."""
            print("Initiating image processing - len file list:", len(image_dataset_handler.file_list))
            debug_print(f"remaining images: {remaining_images.value}")
            if len(image_dataset_handler.file_list) == 0:
                debug_print("No files selected for processing")
                return
            
            debug_print(f"Starting image processing with {len(image_dataset_handler.file_list)} files")
            debug_print(f"Parameters: {image_dataset_handler.__dict__}")
        
            if not image_dataset_handler.generate_caption:
                debug_print("Skipping model loading (caption generation disabled)")

            yield 0, 0, len(image_dataset_handler.file_list), "", "Initializing processing...", None, "", ""
        
            for total, processed, remaining, caption in image_dataset_handler.process_image_files():
                current_image   = image_dataset_handler.get_current_image()
                image_size      = f"{current_image.width}x{current_image.height}" if current_image else ""
                console_message = f"Processed {processed} out of {total} images. {remaining} remaining."
                debug_print(console_message)
                if image_dataset_handler.generate_caption:
                    debug_print(f"Generated caption: {caption}")
                yield total, processed, remaining, caption, console_message, current_image, image_size, caption
        
            debug_print("Finished processing all files")
            
        #def save_modified_caption(caption: str, file_list: List[str]) -> None:
        #    if file_list and caption:
        #        last_processed_image = file_list[-1]
        #        ImageCaptioner.save_caption(last_processed_image, caption)

        start_button.click(
            fn=initiate_image_processing,
            inputs=[],
            outputs=[total_files, initial_images, remaining_images, caption_output, console_output, current_image_display, image_size_display, caption_text_display]
        )

        #save_caption_button = gr.Button("Save Modified Caption")
        #save_caption_button.click(
        #    fn=save_modified_caption,
        #    inputs=[caption_output, file_explorer],
        #    outputs=[]
        #)

        def update_console(message):
            return message

        demo.load(lambda: "", outputs=[console_output])

    demo.launch(share=True)

if __name__ == "__main__":
    gradio_interface()
