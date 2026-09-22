import gradio as gr
from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.ui_components import InputAccordion
from modules.script_callbacks import remove_current_script_callbacks
from backend.nn.unet import SpatialTransformer

from hidiffusion.attention import (
    apply_mswmsaa_attention,
    apply_mswmsaa_attention_simple,
)
from hidiffusion.logger import logger
from hidiffusion.raunet import (
    apply_unet_patches,
    remove_unet_patches,
    apply_rau_net,
    apply_rau_net_simple,
)
from hidiffusion.types import HDDownsample, HDUpsample, UPSCALE_METHODS
from hidiffusion.utils import parse_blocks


logger.info("Script Loaded")


MODES = ["Simple", "Advanced"]


def validate_blocks(unet, selections):
    """Reject custom block IDs that cannot perform the selected operation."""
    model = unet.model.diffusion_model
    for label, name, values, layer_type in selections:
        blocks = getattr(model, f"{name}_blocks", None)
        if blocks is None:
            blocks = [getattr(model, "middle_block", ())] if name == "middle" else ()
        try:
            selected = parse_blocks(name, values)
        except ValueError as exc:
            raise ValueError(f"{label} blocks must be comma-separated integers") from exc
        for _, index in selected:
            if index < 0 or index >= len(blocks):
                raise ValueError(f"{label} block {index} does not exist in the loaded U-Net")
            if not any(isinstance(layer, layer_type) for layer in blocks[index]):
                raise ValueError(f"{label} block {index} has no {layer_type.__name__} layer")


def validate_raunet_pairs(unet, input_blocks, output_blocks, ca_input_blocks, ca_output_blocks):
    """Each shrink operation must be restored before Forge joins its next skip."""
    input_count = len(unet.model.diffusion_model.input_blocks)
    for label, inputs, outputs, offset in (
        ("RauNet", input_blocks, output_blocks, 1),
        ("RauNet pooling", ca_input_blocks, ca_output_blocks, 0),
    ):
        expected = {input_count - index - offset for _, index in parse_blocks("input", inputs)}
        selected = {index for _, index in parse_blocks("output", outputs)}
        if selected != expected:
            needed = ",".join(map(str, sorted(expected))) or "none"
            actual = ",".join(map(str, sorted(selected))) or "none"
            if label == "RauNet pooling":
                raise ValueError(
                    "RAUNet > Advanced Options > Cross-Attention Settings: "
                    f"CA input blocks {inputs or 'none'} require output blocks {needed}, but CA output blocks are {actual}. "
                    f"Set CA Output Blocks to {needed}, or turn off Use custom pooling settings."
                )
            raise ValueError(
                f"RAUNet > Advanced Options: input blocks {inputs or 'none'} require output blocks {needed}, "
                f"but output blocks are {actual}."
            )


def preset_pooling_settings(model_type, time_mode, predictor):
    """Use the model's matching pooling pair until custom CA settings are enabled."""
    input_block, output_block = ("4", "5") if model_type == "SDXL" else ("1", "11")
    if time_mode == "percent":
        start, end = 0.0, 0.35
    elif time_mode == "timestep":
        start, end = 999, round(999 * (1 - 0.35))
    elif time_mode == "sigma":
        start, end = predictor.percent_to_sigma(0.0), predictor.percent_to_sigma(0.35)
    else:
        raise ValueError(f"Unknown time mode: {time_mode}")
    return input_block, output_block, start, end


class ForgeHiDiffusion(scripts.Script):
    sorting_priority = 15  # Adjust this as needed

    def __init__(self):
        super().__init__()
        self.patch_applied = False
        self._sampler_methods = None

    def _remove_patches(self, force=False):
        if self._sampler_methods is not None:
            sampler, methods = self._sampler_methods
            for name, original in methods.items():
                setattr(sampler, name, original)
            self._sampler_methods = None
        if force or self.patch_applied:
            remove_unet_patches()
        self.patch_applied = False

    def _remove_patches_after_sampling(self, p):
        sampler = p.sampler
        methods = {
            name: getattr(sampler, name)
            for name in ("sample", "sample_img2img")
            if callable(getattr(sampler, name, None))
        }
        self._sampler_methods = sampler, methods

        for name, original in methods.items():

            def sample_with_cleanup(*args, _original=original, **kwargs):
                try:
                    return _original(*args, **kwargs)
                finally:
                    self._remove_patches()

            setattr(sampler, name, sample_with_cleanup)

    def title(self):
        return "Forge HiDiffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            model_type = gr.Radio(
                choices=["SD 1.5/2.1", "SDXL"],
                value=lambda: "SDXL",
                label="Model Type",
            )
            hires_fix_enabled = gr.Checkbox(
                label="Apply during Hires Fix",
                value=False,
                info="Also apply HiDiffusion to the high-resolution pass and Forge quick upscale.",
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
                        gr.HTML(
                            """
                            Recommended block settings:<br>
                            <ul><li>SD 1.5/2.1 main pairs: Input 3 / Output 8, Input 6 / Output 5, Input 9 / Output 2</li>
                            <li>SDXL main pairs: Input 3 / Output 5, Input 6 / Output 2</li>
                            <li>SDXL cross-attention pooling pair: Input 4 / Output 5</li></ul>
                            """
                        )
                        raunet_input_blocks = gr.Text(label="Input Blocks", value="3")
                        raunet_output_blocks = gr.Text(label="Output Blocks", value="5")
                        gr.Markdown(
                            "These are the main RauNet blocks. Matching pooling settings are used by default. "
                            "Open Cross-Attention Settings below to customize pooling."
                        )

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
                        custom_ca_enabled = gr.Checkbox(
                            label="Use custom pooling settings",
                            value=False,
                            info="Off uses the matching model preset. Turn on to apply the CA fields below.",
                        )
                        with gr.Group(visible=False) as custom_ca_controls:
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
                            raunet_ca_output_blocks = gr.Text(label="CA Output Blocks", value="5")
                        custom_ca_enabled.change(
                            fn=lambda enabled: gr.update(visible=enabled),
                            inputs=[custom_ca_enabled],
                            outputs=[custom_ca_controls],
                        )

                use_raunet_advanced.change(
                    fn=lambda use_advanced_raunet: gr.Radio(visible=not use_advanced_raunet),
                    inputs=[use_raunet_advanced],
                    outputs=[raunet_res_mode],
                )

            with gr.Tab("MSW-MSA"):
                gr.Markdown(
                    "Simplified MSW-MSA for easier setup. Can improve performance and quality at high resolutions."
                )
                mswmsa_enabled = gr.Checkbox(label="MSW-MSA Enabled", value=lambda: True)

                with InputAccordion(False, label="Advanced") as use_mswmsa_advanced:
                    gr.Markdown("Advanced MSW-MSA settings. For fine-tuning performance and quality improvements.")
                    with gr.Group():
                        gr.HTML(
                            """
                            Recommended block settings:<br>
                            <ul><li>SD 1.5/2.1 attention blocks: inputs 1,2 and outputs 9,10,11</li>
                            <li>SDXL attention blocks: inputs 4,5 and outputs 3,4,5</li></ul>
                            """
                        )
                        mswmsa_input_blocks = gr.Text(label="Input Blocks", value="4,5")
                        mswmsa_middle_blocks = gr.Text(label="Middle Blocks", value="")
                        mswmsa_output_blocks = gr.Text(label="Output Blocks", value="3,4,5")

                    with gr.Group():
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
                            value=0.2,
                            info="For very high resolutions (>2048), try starting at 0.2 or after other scaling effects end",
                        )
                        mswmsa_end_time = gr.Slider(label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=1.0)

            gr.HTML(
                "<br><p><i>Note: Make sure you use the options corresponding to your model type (SD1.5 or SDXL). Otherwise, it may have no effect or fail.</i></p>"
            )
            gr.Markdown(
                "Compatibility: These methods may not work with other attention modifications or scaling effects targeting the same blocks."
            )

            unpatch_button = gr.Button(value="Remove HiDiffusion Patches")
            gr.HTML("Use this if HiDiffusion appears 'stuck' even after disabling the extension")
            unpatch_button.click(fn=lambda: self._remove_patches(force=True))

        # Add JavaScript to handle visibility and model-specific settings
        def update_raunet_settings(model_type):
            if model_type == "SD 1.5/2.1":
                return "3", "8", "1", "11"
            return "3", "5", "4", "5"

        model_type.change(
            fn=update_raunet_settings,
            inputs=[model_type],
            outputs=[
                raunet_input_blocks,
                raunet_output_blocks,
                raunet_ca_input_blocks,
                raunet_ca_output_blocks,
            ],
        )

        def update_mswmsa_settings(model_type):
            if model_type == "SD 1.5/2.1":
                return "1,2", "", "9,10,11"
            else:  # SDXL
                return "4,5", "", "3,4,5"

        model_type.change(
            fn=update_mswmsa_settings,
            inputs=[model_type],
            outputs=[mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks],
        )

        def update_time_controls(time_mode, start_percent, end_percent):
            if time_mode == "percent":
                settings = (0.0, 1.0, 0.01, start_percent, end_percent)
            elif time_mode == "timestep":
                settings = (0, 999, 1, round(999 * (1 - start_percent)), round(999 * (1 - end_percent)))
            else:
                settings = (0.0, 100.0, 0.01, 100.0, 0.0)
            minimum, maximum, step, start, end = settings
            return (
                gr.update(minimum=minimum, maximum=maximum, step=step, value=start),
                gr.update(minimum=minimum, maximum=maximum, step=step, value=end),
            )

        raunet_time_mode.change(
            fn=lambda mode: (*update_time_controls(mode, 0.0, 0.45), *update_time_controls(mode, 0.0, 0.3)),
            inputs=[raunet_time_mode],
            outputs=[raunet_start_time, raunet_end_time, raunet_ca_start_time, raunet_ca_end_time],
        )
        mswmsa_time_mode.change(
            fn=lambda mode: update_time_controls(mode, 0.2, 1.0),
            inputs=[mswmsa_time_mode],
            outputs=[mswmsa_start_time, mswmsa_end_time],
        )

        self.infotext_fields = [
            (enabled, lambda d: "raunet_enabled" in d or "mswmsa_enabled" in d),
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
            (custom_ca_enabled, "raunet_ca_custom"),
            (hires_fix_enabled, "hidiffusion_hires_fix"),
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
            custom_ca_enabled,
            hires_fix_enabled,
        )

    def process_before_every_sampling(self, p: StableDiffusionProcessing, *script_args, **kwargs):
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
        ) = script_args[:25]
        custom_ca_enabled = bool(script_args[25]) if len(script_args) > 25 else False
        hires_fix_enabled = bool(script_args[26]) if len(script_args) > 26 else False

        self._remove_patches()
        if not enabled:
            return

        if not raunet_enabled and not mswmsa_enabled:
            return

        if (getattr(p, "txt2img_upscale", False) or p.is_hr_pass) and not hires_fix_enabled:
            logger.info("HiDiffusion skipped for Hires Fix pass")
            return

        try:
            if raunet_enabled:
                apply_unet_patches()
                self.patch_applied = True

            p.extra_generation_params.update(dict(model_type=model_type))
            if hires_fix_enabled:
                p.extra_generation_params["hidiffusion_hires_fix"] = True

            # Always start with a fresh clone of the original unet
            unet = p.sd_model.forge_objects.unet.clone()

            # Handle RAUNet
            if raunet_enabled:
                p.extra_generation_params.update(dict(raunet_enabled=True, use_raunet_advanced=use_raunet_advanced))

                if use_raunet_advanced:
                    if custom_ca_enabled:
                        ca_input, ca_output = raunet_ca_input_blocks, raunet_ca_output_blocks
                        ca_start, ca_end = raunet_ca_start_time, raunet_ca_end_time
                    else:
                        ca_input, ca_output, ca_start, ca_end = preset_pooling_settings(
                            model_type, raunet_time_mode, unet.model.predictor
                        )
                    validate_blocks(
                        unet,
                        (
                            ("RauNet input", "input", raunet_input_blocks, HDDownsample),
                            ("RauNet output", "output", raunet_output_blocks, HDUpsample),
                            ("RauNet pooling input", "input", ca_input, object),
                            ("RauNet restore output", "output", ca_output, object),
                        ),
                    )
                    validate_raunet_pairs(
                        unet,
                        raunet_input_blocks,
                        raunet_output_blocks,
                        ca_input,
                        ca_output,
                    )
                    unet = apply_rau_net(
                        unet,
                        raunet_input_blocks,
                        raunet_output_blocks,
                        raunet_time_mode,
                        raunet_start_time,
                        raunet_end_time,
                        raunet_skip_two_stage_upscale,
                        raunet_upscale_mode,
                        ca_start,
                        ca_end,
                        ca_input,
                        ca_output,
                        raunet_ca_upscale_mode,
                    )
                    p.extra_generation_params.update(
                        dict(
                            raunet_input_blocks=raunet_input_blocks,
                            raunet_output_blocks=raunet_output_blocks,
                            raunet_time_mode=raunet_time_mode,
                            raunet_start_time=raunet_start_time,
                            raunet_end_time=raunet_end_time,
                            raunet_skip_two_stage_upscale=raunet_skip_two_stage_upscale,
                            raunet_upscale_mode=raunet_upscale_mode,
                            raunet_ca_start_time=ca_start,
                            raunet_ca_end_time=ca_end,
                            raunet_ca_input_blocks=ca_input,
                            raunet_ca_output_blocks=ca_output,
                            raunet_ca_custom=custom_ca_enabled,
                            raunet_ca_upscale_mode=raunet_ca_upscale_mode,
                        )
                    )
                else:
                    unet = apply_rau_net_simple(
                        model_type,
                        raunet_res_mode,
                        raunet_upscale_mode,
                        raunet_ca_upscale_mode,
                        unet,
                    )
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
                    validate_blocks(
                        unet,
                        (
                            ("MSW-MSA input", "input", mswmsa_input_blocks, SpatialTransformer),
                            ("MSW-MSA middle", "middle", mswmsa_middle_blocks, SpatialTransformer),
                            ("MSW-MSA output", "output", mswmsa_output_blocks, SpatialTransformer),
                        ),
                    )
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
                    unet = apply_mswmsaa_attention_simple(model_type, unet)

            # Always update the unet
            p.sd_model.forge_objects.unet = unet
            if self.patch_applied:
                self._remove_patches_after_sampling(p)
        except Exception:
            self._remove_patches()
            raise

        # Add debug logger
        logger.debug(
            f"""HiDiffusion enabled: {enabled}, Model Type: {model_type}
        RAUNet enabled: {raunet_enabled}, Advanced RAUNet mode: {use_raunet_advanced}
        MSW-MSA enabled: {mswmsa_enabled}, Advanced MSW-MSA mode: {use_mswmsa_advanced}
        MSW-MSA settings: Input Blocks: {mswmsa_input_blocks}, Output Blocks: {mswmsa_output_blocks}"""
        )

    def postprocess(self, p, processed, *args):
        self._remove_patches()
        remove_current_script_callbacks()
