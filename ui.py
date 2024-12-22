"""
here are all the things realted to the UI of the application and ONLY the us no processing happens there!
"""


import os
import gradio          as     gr

from   typing          import Tuple, Any, List, Generator
from   dataset_handler import DatasetHandler
from   caption         import ImageCaptioner
from   utils           import debug_print

def gradio_interface() -> None:
    
    image_dataset_handler = DatasetHandler()
    image_captioner       = image_dataset_handler.image_captioner
    valid_models          = image_captioner.valid_models
    with gr.Blocks(css="""
        .nav-button {
            max-width: 50px !important;
            min-width: 50px !important;
            height: 300px !important;
            border-radius: 25px !important;
            font-size: 24px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: rgba(0,0,0,0.1) !important;
        }
        .nav-button:hover {
            background-color: rgba(0,0,0,0.2) !important;
        }
    """) as demo:
        gr.Markdown("# Image Batch Downsampler and Captioner")
        with gr.Row():
            with gr.Column():
                file_explorer           = gr.FileExplorer( file_count = "multiple",                 interactive = True                           )
                start_button            = gr.Button(       "Start"                                                                               )

            with gr.Column():
                with gr.Row():
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


                
                current_index      = gr.Number(value=0, label="Current Image Index", interactive=True)
        # Move image display and navigation outside columns
        with gr.Row():
            prev_button = gr.Button("←", elem_classes="nav-button")
            with gr.Column():
                with gr.Row():
                    x_start_slider    = gr.Slider(minimum=0, maximum=100, value=0,   label="X Start %")
                    x_end_slider      = gr.Slider(minimum=0, maximum=100, value=100, label="X End %")
                current_image_display = gr.Image(label="Current Image", height=700, width=1000)
                with gr.Row():
                    y_start_slider    = gr.Slider(minimum=0, maximum=100, value=0,   label="Y Start %")
                    y_end_slider      = gr.Slider(minimum=0, maximum=100, value=100, label="Y End %")
            next_button = gr.Button("→", elem_classes="nav-button")
            
        with gr.Row():
            image_size_display     = gr.Textbox(label="Image Size",                             value="", interactive=False)
            caption_output         = gr.Textbox(label="Generated Caption",                      value="", visible=False, lines=5, interactive=True)
            caption_text_display   = gr.Textbox(label="Caption Text",                           value="", lines=3, interactive=False)
            console_output         = gr.Textbox(label="Console Output",                                   lines=5, interactive=False)
            total_files            = gr.Textbox(label="Number of files selected",               value="0")
            initial_images         = gr.Textbox(label="Initial number of images to be treated", value="0")
            remaining_images       = gr.Textbox(label="Remaining number of images to treat",    value="0")

        def update_file_count(selected_files: List[str]) -> Tuple[int, int, int, Any, str, str]:
            image_dataset_handler.set_file_list(selected_files)
            count = len(image_dataset_handler.file_list)
            if count > 0:
                image, size, caption = image_dataset_handler.load_image_at_index(0)
                return count, count, count, image, size, caption
            return 0, 0, 0, None, "", ""

        def browse_image(index: int) -> Tuple[Any, str, str, float, float, float, float, Any]:
            image, size, caption = image_dataset_handler.load_image_at_index(index)
            crop_values = image_dataset_handler.get_crop_values_at_index(index)
            updated_image = update_crop_preview(crop_values["min_x"], crop_values["max_x"], crop_values["min_y"], crop_values["max_y"], code_triggered_flag=True)
            return image, size, caption, crop_values["min_x"], crop_values["max_x"], crop_values["min_y"], crop_values["max_y"], updated_image

        def update_crop_preview(x_start, x_end, y_start, y_end, code_triggered_flag=False):
            """Update the crop preview overlay"""
            current_image = image_dataset_handler.get_current_image()
            if current_image is None:
                return None
            updated_image = image_dataset_handler.image_tools.draw_crop_bounds(
                current_image, x_start, x_end, y_start, y_end
            )
            return updated_image

        def update_crop_values(index: int, x_start: float, x_end: float, y_start: float, y_end: float, code_triggered_flag=False) -> None:
            """Updates the crop values in the dataset handler; and ensure we have a rectangle > 0."""
            if not code_triggered_flag:
                image_dataset_handler.set_crop_values_at_index(index, x_start, x_end, y_start, y_end)
            return None


        def next_image(current_idx: int, x_start: float, x_end: float, y_start: float, y_end: float) -> Tuple[int, Any, str, str, float, float, float, float, Any]:
            # Save crop of current image before moving
            
            next_idx = min(current_idx + 1, len(image_dataset_handler.file_list) - 1)
            image, size, caption = image_dataset_handler.load_image_at_index(next_idx)
            crop_values = image_dataset_handler.get_crop_values_at_index(next_idx)
            updated_image = update_crop_preview(crop_values["min_x"], crop_values["max_x"], crop_values["min_y"], crop_values["max_y"], code_triggered_flag=True)
            return next_idx, image, size, caption, crop_values["min_x"], crop_values["max_x"], crop_values["min_y"], crop_values["max_y"], updated_image

        def prev_image(current_idx: int, x_start: float, x_end: float, y_start: float, y_end: float) -> Tuple[int, Any, str, str, float, float, float, float, Any]:
            # Save crop of current image before moving
            
            prev_idx = max(current_idx - 1, 0)
            image, size, caption = image_dataset_handler.load_image_at_index(prev_idx)
            crop_values = image_dataset_handler.get_crop_values_at_index(prev_idx)
            updated_image = update_crop_preview(crop_values["min_x"], crop_values["max_x"], crop_values["min_y"], crop_values["max_y"], code_triggered_flag=True)
            return prev_idx, image, size, caption, crop_values["min_x"], crop_values["max_x"], crop_values["min_y"], crop_values["max_y"], updated_image

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

        file_explorer.change(
            fn=update_file_count,
            inputs=file_explorer,
            outputs=[total_files, initial_images, remaining_images, current_image_display, image_size_display, caption_text_display]
        )
        
        current_index.change(
            fn=browse_image,
            inputs=[current_index],
            outputs=[current_image_display, image_size_display, caption_text_display, x_start_slider, x_end_slider, y_start_slider, y_end_slider, current_image_display]
        )
        
        next_button.click(
            fn=next_image,
            inputs=[current_index, x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[current_index, current_image_display, image_size_display, caption_text_display, x_start_slider, x_end_slider, y_start_slider, y_end_slider, current_image_display]
        )
        
        prev_button.click(
            fn=prev_image,
            inputs=[current_index, x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[current_index, current_image_display, image_size_display, caption_text_display, x_start_slider, x_end_slider, y_start_slider, y_end_slider, current_image_display]
        )
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

        # Wire up crop boundary preview updates
        x_start_slider.change(
            fn=lambda x_start, x_end, y_start, y_end: update_crop_preview(x_start, x_end, y_start, y_end),
            inputs=[x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[current_image_display]
        )
        x_end_slider.change(
            fn=lambda x_start, x_end, y_start, y_end: update_crop_preview(x_start, x_end, y_start, y_end),
            inputs=[x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[current_image_display]
        )
        y_start_slider.change(
            fn=lambda x_start, x_end, y_start, y_end: update_crop_preview(x_start, x_end, y_start, y_end),
            inputs=[x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[current_image_display]
        )
        y_end_slider.change(
            fn=lambda x_start, x_end, y_start, y_end: update_crop_preview(x_start, x_end, y_start, y_end),
            inputs=[x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[current_image_display]
        )
        
        # Wire up crop value updates
        x_start_slider.change(
            fn=lambda index, x_start, x_end, y_start, y_end: update_crop_values(index, x_start, x_end, y_start, y_end),
            inputs=[current_index, x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[]
        )
        x_end_slider.change(
            fn=lambda index, x_start, x_end, y_start, y_end: update_crop_values(index, x_start, x_end, y_start, y_end),
            inputs=[current_index, x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[]
        )
        y_start_slider.change(
            fn=lambda index, x_start, x_end, y_start, y_end: update_crop_values(index, x_start, x_end, y_start, y_end),
            inputs=[current_index, x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[]
        )
        y_end_slider.change(
            fn=lambda index, x_start, x_end, y_start, y_end: update_crop_values(index, x_start, x_end, y_start, y_end),
            inputs=[current_index, x_start_slider, x_end_slider, y_start_slider, y_end_slider],
            outputs=[]
        )
        
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

    demo.launch(share=False)

if __name__ == "__main__":
    gradio_interface()
