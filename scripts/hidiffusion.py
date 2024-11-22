import gradio as gr
from modules import scripts
from modules.ui_components import InputAccordion
from modules.script_callbacks import remove_current_script_callbacks

# Now import from your package
from hidiffusion.raunet import (
    apply_unet_patches,
    remove_unet_patches,
    apply_rau_net,
    apply_rau_net_simple,
    UPSCALE_METHODS,
)
from hidiffusion.attention import (
    apply_mswmsaa_attention,
    apply_mswmsaa_attention_simple,
)
from hidiffusion.logger import logger


logger.info("Script Loaded")


MODES = ["Simple", "Advanced"]


class RAUNetScript(scripts.Script):
    sorting_priority = 15  # Adjust this as needed

    def title(self):
        return "HiDiffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            model_type = gr.Radio(
                choices=["SD15", "SDXL"],
                value=lambda: "SDXL",
                label="Model Type",
                info="Note: Use SD15 setting for SD 2.1 as well.",
            )

            with gr.Tab("RAUNet"):
                gr.Markdown("RAUNet helps avoid artifacts at high resolutions.")
                raunet_enabled = gr.Checkbox(label="RAUNet Enabled", value=lambda: True)
                raunet_res_mode = gr.Radio(
                    choices=[
                        "low (1024 or lower)",
                        "high (1536-2048)",
                        "ultra (over 2048)",
                    ],
                    value=lambda: "high (1536-2048)",
                    label="Resolution Mode",
                    info="Note: Resolution mode is a preset, exact match to your resolution is not necessary.",
                )
                raunet_upscale_mode = gr.Dropdown(
                    choices=UPSCALE_METHODS,
                    value=UPSCALE_METHODS[0],
                    label="Upscale Mode",
                )
                raunet_ca_upscale_mode = gr.Dropdown(
                    choices=UPSCALE_METHODS,
                    value=UPSCALE_METHODS[0],
                    label="CA Upscale Mode",
                )

                with InputAccordion(False, label="Advanced Options") as use_raunet_advanced:
                    with gr.Group():
                        raunet_input_blocks = gr.Text(label="Input Blocks", value="3")
                        raunet_output_blocks = gr.Text(label="Output Blocks", value="8")
                        gr.Markdown(
                            "For SD1.5: Input 3 corresponds to Output 8, Input 6 to Output 5, Input 9 to Output 2"
                        )
                        gr.Markdown("For SDXL: Input 3 corresponds to Output 5, Input 6 to Output 2")

                    with gr.Group():
                        raunet_time_mode = gr.Dropdown(
                            choices=["percent", "timestep", "sigma"],
                            value="percent",
                            label="Time Mode",
                            info="Controls format of start/end times. Use percent if unsure.",
                        )
                        raunet_start_time = gr.Slider(
                            label="Start Time",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.0,
                        )
                        raunet_end_time = gr.Slider(
                            label="End Time",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.45,
                        )

                    raunet_skip_two_stage_upscale = gr.Checkbox(label="Skip Two-Stage Upscale", value=False)

                    with gr.Accordion(open=False, label="Cross-Attention Settings"):
                        raunet_ca_start_time = gr.Slider(
                            label="CA Start Time",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.0,
                        )
                        raunet_ca_end_time = gr.Slider(
                            label="CA End Time",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.3,
                        )
                        raunet_ca_input_blocks = gr.Text(label="CA Input Blocks", value="4")
                        raunet_ca_output_blocks = gr.Text(label="CA Output Blocks", value="8")

            with gr.Tab("MSW-MSA Simple"):
                gr.Markdown(
                    "Simplified MSW-MSA for easier setup. Can improve performance and quality at high resolutions."
                )
                mswmsa_enabled = gr.Checkbox(label="MSW-MSA Enabled", value=lambda: True)

                with InputAccordion(False, label="MSW-MSA Advanced") as use_mswmsa_advanced:
                    with gr.Group():
                        gr.Markdown("Advanced MSW-MSA settings. For fine-tuning performance and quality improvements.")
                        mswmsa_input_blocks = gr.Text(label="Input Blocks", value="1,2")
                        mswmsa_middle_blocks = gr.Text(label="Middle Blocks", value="")
                        mswmsa_output_blocks = gr.Text(label="Output Blocks", value="9,10,11")
                        gr.Markdown("Recommended SD15: input 1,2, output 9,10,11")
                        gr.Markdown("Recommended SDXL: input 4,5, output 4,5")

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
                    mswmsa_end_time = gr.Slider(label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                    gr.Markdown(
                        "Note: For very high resolutions (>2048), try starting at 0.2 or after other scaling effects end."
                    )

            gr.HTML(
                "<p><i>Note: Make sure you use the options corresponding to your model type (SD1.5 or SDXL). Otherwise, it may have no effect or fail.</i></p>"
            )
            gr.Markdown(
                "Compatibility: These methods may not work with other attention modifications or scaling effects targeting the same blocks."
            )

        # Add JavaScript to handle visibility and model-specific settings
        def update_raunet_settings(model_type):
            if model_type == "SD15":
                return "3", "8", "4", "8", 0.0, 0.45, 0.0, 0.3
            else:  # SDXL
                return (
                    "3",
                    "5",
                    "2",
                    "7",
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                )  # Disabling both patches by default for SDXL

        model_type.change(
            fn=update_raunet_settings,
            inputs=[model_type],
            outputs=[
                raunet_input_blocks,
                raunet_output_blocks,
                raunet_ca_input_blocks,
                raunet_ca_output_blocks,
                raunet_start_time,
                raunet_end_time,
                raunet_ca_start_time,
                raunet_ca_end_time,
            ],
        )

        def update_mswmsa_settings(model_type):
            if model_type == "SD15":
                return "1,2", "", "9,10,11"
            else:  # SDXL
                return "4,5", "", "4,5"

        model_type.change(
            fn=update_mswmsa_settings,
            inputs=[model_type],
            outputs=[mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks],
        )

        self.infotext_fields = [
            (enabled, lambda d: 'raunet_enabled' in d or 'mswmsa_enabled' in d),
            (model_type, "model_type"),
            (raunet_enabled, "raunet_enabled"),
            (use_raunet_advanced, "use_raunet_advanced"),
            (raunet_res_mode, "raunet_res_mode"),
            (raunet_input_blocks, "raunet_input_blocks"),
            (raunet_output_blocks, "raunet_output_blocks"),
            (raunet_time_mode, "raunet_time_mode"),
            (raunet_start_time, "raunet_start_time"),
            (raunet_end_time, "raunet_end_time"),
            (raunet_skip_two_stage_upscale, "raunet_skip_two_stage_upscale"),
            (raunet_upscale_mode, "raunet_upscale_mode"),
            (raunet_ca_end_time, "raunet_ca_end_time"),
            (raunet_ca_input_blocks, "raunet_ca_input_blocks"),
            (raunet_ca_output_blocks, "raunet_ca_output_blocks"),
            (raunet_ca_start_time, "raunet_ca_start_time"),
            (raunet_ca_upscale_mode, "raunet_ca_upscale_mode"),
            (mswmsa_enabled, "mswmsa_enabled"),
            (use_mswmsa_advanced, "use_mswmsa_advanced"),
            (mswmsa_input_blocks, "mswmsa_input_blocks"),
            (mswmsa_middle_blocks, "mswmsa_middle_blocks"),
            (mswmsa_output_blocks, "mswmsa_output_blocks"),
            (mswmsa_time_mode, "mswmsa_time_mode"),
            (mswmsa_start_time, "mswmsa_start_time"),
            (mswmsa_end_time, "mswmsa_end_time"),
        ]

        return (
            enabled,
            model_type,
            raunet_enabled,
            use_raunet_advanced,
            raunet_res_mode,
            raunet_input_blocks,
            raunet_output_blocks,
            raunet_time_mode,
            raunet_start_time,
            raunet_end_time,
            raunet_skip_two_stage_upscale,
            raunet_upscale_mode,
            raunet_ca_end_time,
            raunet_ca_input_blocks,
            raunet_ca_output_blocks,
            raunet_ca_start_time,
            raunet_ca_upscale_mode,
            mswmsa_enabled,
            use_mswmsa_advanced,
            mswmsa_input_blocks,
            mswmsa_middle_blocks,
            mswmsa_output_blocks,
            mswmsa_time_mode,
            mswmsa_start_time,
            mswmsa_end_time,
        )

    def before_process(self, p, *script_args):
        enabled: bool = script_args[0]

        if enabled:
            apply_unet_patches()

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (
            enabled,
            model_type,
            raunet_enabled,
            use_raunet_advanced,
            raunet_res_mode,
            raunet_input_blocks,
            raunet_output_blocks,
            raunet_time_mode,
            raunet_start_time,
            raunet_end_time,
            raunet_skip_two_stage_upscale,
            raunet_upscale_mode,
            raunet_ca_end_time,
            raunet_ca_input_blocks,
            raunet_ca_output_blocks,
            raunet_ca_start_time,
            raunet_ca_upscale_mode,
            mswmsa_enabled,
            use_mswmsa_advanced,
            mswmsa_input_blocks,
            mswmsa_middle_blocks,
            mswmsa_output_blocks,
            mswmsa_time_mode,
            mswmsa_start_time,
            mswmsa_end_time,
        ) = script_args

        if not enabled:
            return

        p.extra_generation_params.update(dict(model_type=model_type))

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        # Handle RAUNet
        if raunet_enabled:  # Explicit check for True
            p.extra_generation_params.update(dict(raunet_enabled=True, use_raunet_advanced=use_raunet_advanced))

            if use_raunet_advanced:
                unet = apply_rau_net(
                    unet,
                    raunet_input_blocks,
                    raunet_output_blocks,
                    raunet_time_mode,
                    raunet_start_time,
                    raunet_end_time,
                    raunet_skip_two_stage_upscale,
                    raunet_upscale_mode,
                    raunet_ca_start_time,
                    raunet_ca_end_time,
                    raunet_ca_input_blocks,
                    raunet_ca_output_blocks,
                    raunet_ca_upscale_mode,
                )[0]
                p.extra_generation_params.update(
                    dict(
                        raunet_input_blocks=raunet_input_blocks,
                        raunet_output_blocks=raunet_output_blocks,
                        raunet_time_mode=raunet_time_mode,
                        raunet_start_time=raunet_start_time,
                        raunet_end_time=raunet_end_time,
                        raunet_skip_two_stage_upscale=raunet_skip_two_stage_upscale,
                        raunet_upscale_mode=raunet_upscale_mode,
                        raunet_ca_start_time=raunet_ca_start_time,
                        raunet_ca_end_time=raunet_ca_end_time,
                        raunet_ca_input_blocks=raunet_ca_input_blocks,
                        raunet_ca_output_blocks=raunet_ca_output_blocks,
                        raunet_ca_upscale_mode=raunet_ca_upscale_mode,
                    )
                )
            else:  # Explicit check for True
                unet = apply_rau_net_simple(
                    model_type,
                    raunet_res_mode,
                    raunet_upscale_mode,
                    raunet_ca_upscale_mode,
                    unet,
                )[0]
                p.extra_generation_params.update(
                    dict(
                        raunet_res_mode=raunet_res_mode,
                        raunet_upscale_mode=raunet_upscale_mode,
                        raunet_ca_upscale_mode=raunet_ca_upscale_mode,
                    )
                )

        # Handle MSW-MSA
        if mswmsa_enabled:
            p.extra_generation_params.update(dict(mswmsa_enabled=True, use_mswmsa_advanced=use_mswmsa_advanced))

            if use_mswmsa_advanced:
                unet = apply_mswmsaa_attention(
                    unet,
                    mswmsa_input_blocks,
                    mswmsa_middle_blocks,
                    mswmsa_output_blocks,
                    mswmsa_time_mode,
                    mswmsa_start_time,
                    mswmsa_end_time,
                )[0]
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
            elif mswmsa_enabled:  # Explicit check for True
                unet = apply_mswmsaa_attention_simple(model_type, unet)[0]

        # Always update the unet
        p.sd_model.forge_objects.unet = unet

        # Add debug logger
        logger.debug(
            f"""HiDiffusion enabled: {enabled}, Model Type: {model_type}
        RAUNet enabled: {raunet_enabled}, Advanced RAUNet mode: {use_raunet_advanced}
        MSW-MSA enabled: {mswmsa_enabled}, Advanced MSW-MSA mode: {use_mswmsa_advanced}
        MSW-MSA settings: Input Blocks: {mswmsa_input_blocks}, Output Blocks: {mswmsa_output_blocks}"""
        )

    def postprocess(self, p, processed, *args):
        enabled: bool = args[0]
        if enabled:
            remove_unet_patches()
        remove_current_script_callbacks()
