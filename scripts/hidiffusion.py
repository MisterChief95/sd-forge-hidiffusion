import logging

import gradio as gr
from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import remove_current_script_callbacks

# Now import from your package
from hidiffusion.raunet import (
    apply_monkey_patch,
    remove_monkey_patch,
    apply_rau_net,
    apply_rau_net_simple,
)
from hidiffusion.attention import (
    apply_mswmsaa_attention,
    apply_mswmsaa_attention_simple,
)
from hidiffusion.types import *

logging.info("Imports successful in RAUNet script")


class RAUNetScript(scripts.Script):
    sorting_priority = 15  # Adjust this as needed
    is_patched = False

    def title(self):
        return "Hidiffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            model_type = gr.Radio(
                choices=MODEL_TYPES,
                value=MODEL_TYPES[0],
                label="Model Type",
                info="Use SD 1.5 setting for SD 2.1 as well",
            )

            gr.HTML("<br>")

            with gr.Tab("RAUNet"):
                raunet_enabled = gr.Checkbox(label="RAUNet Enabled", value=True)
                raunet_mode_select = gr.Radio(
                    choices=MODE_LEVELS,
                    value=MODE_LEVELS[0],
                    label="Mode",
                    info="Simple mode is recommended for most users. Advanced mode is for fine-tuning.",
                )
                raunet_res_mode = gr.Radio(
                    interactive=True,
                    choices=RESOLUTION_MODES.keys(),
                    value=RESOLUTION_MODES[1],
                    label="Resolution Mode",
                    info="SIMPLE MODE ONLY - Resolution mode is a preset, exact match to your resolution is not necessary.",
                )

                gr.HTML("<br>")

                upscale_modes: list[str] = UPSCALE_METHODS
                upscale_mode = gr.Dropdown(
                    choices=upscale_modes,
                    value=UPSCALE_METHODS[0],
                    label="Upscale Mode",
                )
                ca_upscale_mode = gr.Dropdown(
                    choices=upscale_modes,
                    value=UPSCALE_METHODS[0],
                    label="CA Upscale Mode",
                )

                gr.HTML("<br>")

                with gr.Accordion("Advanced RAUNet Settings", open=False):
                    raunet_mode_warning = gr.HTML(
                        '<h3 style="color:red;">Enabled Advanced Mode to Change these Settings</h3>',
                        visible=True,
                    )

                    gr.HTML("<br>")

                    raunet_input_blocks = gr.Textbox(
                        label="Input Blocks", value="3", interactive=False
                    )
                    raunet_output_blocks = gr.Textbox(
                        label="Output Blocks", value="8", interactive=False
                    )
                    gr.Markdown(
                        "**Recommended SD1.5**: Input 3 corresponds to Output 8, Input 6 to Output 5, Input 9 to Output 2"
                    )
                    gr.Markdown(
                        "**Recommended SDXL**: Input 3 corresponds to Output 5, Input 6 to Output 2"
                    )

                    gr.HTML("<br>")

                    time_mode = gr.Dropdown(
                        choices=["percent", "timestep", "sigma"],
                        value="percent",
                        label="Time Mode",
                        interactive=False,
                        info="Controls format of start/end times. Use percent if unsure",
                    )

                    start_time = gr.Slider(
                        label="Start Time",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.0,
                        interactive=False,
                    )
                    end_time = gr.Slider(
                        label="End Time",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.45,
                        interactive=False,
                    )
                    skip_two_stage_upscale = gr.Checkbox(
                        label="Skip Two Stage Upscale", value=False, interactive=False
                    )

                    gr.HTML("<br>")

                    with gr.Accordion(open=False, label="Cross-Attention Settings"):
                        ca_start_time = gr.Slider(
                            label="CA Start Time",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.0,
                            interactive=False,
                        )
                        ca_end_time = gr.Slider(
                            label="CA End Time",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.3,
                            interactive=False,
                        )
                        ca_input_blocks = gr.Textbox(
                            label="CA Input Blocks", value="4", interactive=False
                        )
                        ca_output_blocks = gr.Textbox(
                            label="CA Output Blocks", value="8", interactive=False
                        )

                def toggle_raunet_settings(
                    raunet_mode_select,
                    res_mode,
                    raunet_mode_warning,
                    input_blocks,
                    output_blocks,
                    time_mode,
                    start_time,
                    end_time,
                    skip_two_stage_upscale,
                    ca_start_time,
                    ca_end_time,
                    ca_input_blocks,
                    ca_output_blocks,
                ):  # noqa: E501
                    if raunet_mode_select == MODE_LEVELS[0]:  # Simple mode
                        res_mode = gr.Radio(visible=True)
                        raunet_mode_warning = gr.HTML(visible=True)
                        input_blocks = gr.Textbox(interactive=False)
                        output_blocks = gr.Textbox(interactive=False)
                        time_mode = gr.Dropdown(interactive=False)
                        start_time = gr.Slider(interactive=False)
                        end_time = gr.Slider(interactive=False)
                        skip_two_stage_upscale = gr.Checkbox(interactive=False)
                        ca_start_time = gr.Slider(interactive=False)
                        ca_end_time = gr.Slider(interactive=False)
                        ca_input_blocks = gr.Textbox(interactive=False)
                        ca_output_blocks = gr.Textbox(interactive=False)
                    else:
                        res_mode = gr.Radio(visible=False)
                        raunet_mode_warning = gr.HTML(visible=False)
                        input_blocks = gr.Textbox(interactive=True)
                        output_blocks = gr.Textbox(interactive=True)
                        time_mode = gr.Dropdown(interactive=True)
                        start_time = gr.Slider(interactive=True)
                        end_time = gr.Slider(interactive=True)
                        skip_two_stage_upscale = gr.Checkbox(interactive=True)
                        ca_start_time = gr.Slider(interactive=True)
                        ca_end_time = gr.Slider(interactive=True)
                        ca_input_blocks = gr.Textbox(interactive=True)
                        ca_output_blocks = gr.Textbox(interactive=True)

                    return (
                        res_mode,
                        raunet_mode_warning,
                        input_blocks,
                        output_blocks,
                        time_mode,
                        start_time,
                        end_time,
                        skip_two_stage_upscale,
                        ca_start_time,
                        ca_end_time,
                        ca_input_blocks,
                        ca_output_blocks,
                    )

                raunet_mode_select.change(
                    fn=toggle_raunet_settings,
                    inputs=[
                        raunet_mode_select,
                        raunet_res_mode,
                        raunet_mode_warning,
                        raunet_input_blocks,
                        raunet_output_blocks,
                        time_mode,
                        start_time,
                        end_time,
                        skip_two_stage_upscale,
                        ca_start_time,
                        ca_end_time,
                        ca_input_blocks,
                        ca_output_blocks,
                    ],
                    outputs=[
                        raunet_res_mode,
                        raunet_mode_warning,
                        raunet_input_blocks,
                        raunet_output_blocks,
                        time_mode,
                        start_time,
                        end_time,
                        skip_two_stage_upscale,
                        ca_start_time,
                        ca_end_time,
                        ca_input_blocks,
                        ca_output_blocks,
                    ],
                )

            with gr.Tab("MSW-MSA"):
                mswmsa_enabled = gr.Checkbox(label="MSW-MSA Simple Enabled", value=True)
                mswmsa_mode_select = gr.Radio(
                    choices=MODE_LEVELS,
                    value=MODE_LEVELS[0],
                    label="Mode",
                    info="Simple mode is recommended for most users. Advanced mode is for fine-tuning.",
                )

                gr.HTML("<br>")

                with gr.Accordion("Advanced MSW-MSA Settings", open=False):
                    mswmsa_input_blocks = gr.Textbox(label="Input Blocks", value="1,2")
                    mswmsa_middle_blocks = gr.Textbox(label="Middle Blocks", value="")
                    mswmsa_output_blocks = gr.Textbox(
                        label="Output Blocks", value="9,10,11"
                    )
                    gr.Markdown("**Recommended SD1.5**: input 1,2, output 9,10,11")
                    gr.Markdown("**Recommended SDXL**: input 4,5, output 4,5")
                    mswmsa_time_mode = gr.Dropdown(
                        choices=["percent", "timestep", "sigma"],
                        value="percent",
                        label="Time Mode",
                    )
                    mswmsa_start_time = gr.Slider(
                        label="Start Time",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.0,
                    )
                    mswmsa_end_time = gr.Slider(
                        label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=1.0
                    )
                    gr.Markdown(
                        "Note: For very high resolutions (>2048), try starting at 0.2 or after other scaling effects end."
                    )

                gr.HTML(
                    "<p><i>Note: Make sure you use the options corresponding to your model type (SD1.5 or SDXL). Otherwise, it may have no effect or fail.</i></p>"
                )
                gr.HTML(
                    '<p><b><span style="color: orange;">Compatibility: These methods may not work with other attention modifications or scaling effects targeting the same blocks.</span></b></p>'
                )

        def update_settings_from_model_type(model_type: str):
            match model_type:
                case "SD 1.5":
                    return "3", "8", "4", "8", 0.0, 0.45, 0.0, 0.3, "1,2", "", "9,10,11"
                case "SDXL":
                    return "3", "5", "2", "7", 1.0, 1.0, 1.0, 1.0, "4,5", "", "4,5"

        model_type.change(
            fn=update_settings_from_model_type,
            inputs=[model_type],
            outputs=[
                raunet_input_blocks,
                raunet_output_blocks,
                ca_input_blocks,
                ca_output_blocks,
                start_time,
                end_time,
                ca_start_time,
                ca_end_time,
                mswmsa_input_blocks,
                mswmsa_middle_blocks,
                mswmsa_output_blocks,
            ],
        )

        return (
            enabled,
            raunet_mode_select,
            raunet_res_mode,
            raunet_enabled,
            model_type,
            raunet_input_blocks,
            raunet_output_blocks,
            time_mode,
            start_time,
            end_time,
            skip_two_stage_upscale,
            upscale_mode,
            ca_start_time,
            ca_end_time,
            ca_input_blocks,
            ca_output_blocks,
            ca_upscale_mode,
            mswmsa_enabled,
            mswmsa_mode_select,
            mswmsa_input_blocks,
            mswmsa_middle_blocks,
            mswmsa_output_blocks,
            mswmsa_time_mode,
            mswmsa_start_time,
            mswmsa_end_time,
        )

    def before_process(self, _, *script_args):
        (
            enabled,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = script_args

        if enabled:
            logging.info("\x1b[32mRAUNet script enabled\x1b[0m")

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *script_args, **kwargs
    ):
        (
            enabled,
            raunet_mode_select,
            raunet_res_mode,
            raunet_enabled,
            model_type,
            input_blocks,
            output_blocks,
            time_mode,
            start_time,
            end_time,
            skip_two_stage_upscale,
            upscale_mode,
            ca_start_time,
            ca_end_time,
            ca_input_blocks,
            ca_output_blocks,
            ca_upscale_mode,
            mswmsa_enabled,
            mswmsa_mode_select,
            mswmsa_input_blocks,
            mswmsa_middle_blocks,
            mswmsa_output_blocks,
            mswmsa_time_mode,
            mswmsa_start_time,
            mswmsa_end_time,
        ) = script_args

        if not enabled:
            return

        apply_monkey_patch()

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        raunet_res_mode = RESOLUTION_MODES.get(raunet_res_mode, "high")

        # Handle RAUNet
        if raunet_enabled:
            p.extra_generation_params.update(dict(raunet_enabled=True))

            if raunet_mode_select is "Simple":  # Explicit check for True
                unet = apply_rau_net_simple(
                    model_type, raunet_res_mode, upscale_mode, ca_upscale_mode, unet
                )
                p.extra_generation_params.update(
                    dict(
                        raunet_res_mode=raunet_res_mode,
                        raunet_upscale_mode=upscale_mode,
                        raunet_ca_upscale_mode=ca_upscale_mode,
                    )
                )
            elif raunet_enabled is "Advanced":  # Explicit check for True
                unet = apply_rau_net(
                    unet,
                    input_blocks,
                    output_blocks,
                    time_mode,
                    start_time,
                    end_time,
                    skip_two_stage_upscale,
                    upscale_mode,
                    ca_start_time,
                    ca_end_time,
                    ca_input_blocks,
                    ca_output_blocks,
                    ca_upscale_mode,
                )
                p.extra_generation_params.update(
                    dict(
                        raunet_input_blocks=input_blocks,
                        raunet_output_blocks=output_blocks,
                        raunet_time_mode=time_mode,
                        raunet_start_time=start_time,
                        raunet_end_time=end_time,
                        raunet_skip_two_stage_upscale=skip_two_stage_upscale,
                        raunet_upscale_mode=upscale_mode,
                        raunet_ca_start_time=ca_start_time,
                        raunet_ca_end_time=ca_end_time,
                        raunet_ca_input_blocks=ca_input_blocks,
                        raunet_ca_output_blocks=ca_output_blocks,
                        raunet_ca_upscale_mode=ca_upscale_mode,
                    )
                )
        else:
            # Apply RAUNet patch with interactive=False to reset any modifications
            unet = apply_rau_net(unet, "", "", "", 0, 0, False, "", 0, 0, "", "", "")
            unet = apply_rau_net_simple(
                model_type, raunet_res_mode, upscale_mode, ca_upscale_mode, unet
            )
            p.extra_generation_params.update(dict(raunet_enabled=False))

        # Handle MSW-MSA
        if mswmsa_enabled:
            p.extra_generation_params.update(
                dict(mswmsa_enabled=True, mswmsa_mode_select=mswmsa_mode_select)
            )

            if mswmsa_mode_select is "Simple":  # Explicit check for True
                unet = apply_mswmsaa_attention_simple(model_type, unet)

            elif mswmsa_mode_select is "Advanced":  # Explicit check for True
                unet = apply_mswmsaa_attention(
                    unet,
                    mswmsa_input_blocks,
                    mswmsa_middle_blocks,
                    mswmsa_output_blocks,
                    mswmsa_time_mode,
                    mswmsa_start_time,
                    mswmsa_end_time,
                )
                p.extra_generation_params.update(
                    dict(
                        mswmsa_input_blocks=mswmsa_input_blocks,
                        mswmsa_middle_blocks=mswmsa_middle_blocks,
                        mswmsa_output_blocks=mswmsa_output_blocks,
                        mswmsa_time_mode=mswmsa_time_mode,
                        mswmsa_start_time=mswmsa_start_time,
                        mswmsa_end_time=mswmsa_end_time,
                    )
                )
        else:
            # Apply MSW-MSA patch with empty block settings to reset any modifications
            unet = apply_mswmsaa_attention(unet, "", "", "", mswmsa_time_mode, 0, 0)
            unet = apply_mswmsaa_attention_simple(model_type, unet)
            p.extra_generation_params.update(
                dict(mswmsa_enabled=False, mswmsa_simple_enabled=False)
            )

        # Always update the unet
        p.sd_model.forge_objects.unet = unet

        # Add debug logging
        logging.debug(
            f"HiDiffusion [enabled: {enabled}, Model Type: {model_type}, RAUnet Enabled: {raunet_enabled}, MSW-MSA Enabled: {mswmsa_enabled}]"
        )
        if raunet_enabled:
            logging.debug(
                f"""RAUNet Settings:
                    Input Blocks: {input_blocks}, Output Blocks: {output_blocks}
                    Time Mode: {time_mode}
                        - Start Time: {start_time}
                        - End Time: {end_time}
                    Skip Two Stage Upscale: {skip_two_stage_upscale}
                    Upscale Mode: {upscale_mode}
                Cross-Attention Settings:
                    Start Time: {ca_start_time}
                    End Time: {ca_end_time}
                    Input Blocks: {ca_input_blocks}
                    Output Blocks: {ca_output_blocks}
                    Upscale Mode: {ca_upscale_mode}"""
            )
        if mswmsa_enabled:
            logging.debug(
                f"""MSW-MSA Settings:
                    Input Blocks: {mswmsa_input_blocks}, Middle Blocks: {mswmsa_middle_blocks}, Output Blocks: {mswmsa_output_blocks}
                    Time Mode: {mswmsa_time_mode}
                        - Start Time: {mswmsa_start_time}
                        - End Time: {mswmsa_end_time}"""
            )

        remove_monkey_patch()

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
