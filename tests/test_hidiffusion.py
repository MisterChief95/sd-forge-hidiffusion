"""Small CPU regressions for Forge HiDiffusion's patch boundaries."""

import ast
import logging
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Forge parses process arguments during import; unittest's flags are unrelated.
_argv = sys.argv
sys.argv = sys.argv[:1]
from hidiffusion.attention import apply_mswmsaa_attention, get_window_args, window_partition, window_reverse  # noqa: E402
from hidiffusion.raunet import (  # noqa: E402
    apply_rau_net,
    apply_rau_net_simple,
    apply_unet_patches,
    configure_blocks,
    hd_forward_timestep_embed,
    remove_unet_patches,
)
from hidiffusion.types import HD_CONFIG, HDDownsample, HDUpsample, ProxyUpsample  # noqa: E402
from hidiffusion.utils import parse_blocks  # noqa: E402
from backend.nn.unet import IntegratedUNet2DConditionModel  # noqa: E402
from backend.nn import unet as forge_unet  # noqa: E402

sys.argv = _argv


class _Patcher:
    def clone(self):
        return self


class _CastingConv(torch.nn.Conv2d):
    """Model Forge's manual casting path without loading a checkpoint."""

    parameters_manual_cast = True

    def forward(self, x):
        return F.conv2d(x, self.weight.to(x), self.bias.to(x), self.stride, self.padding, self.dilation)


class HiDiffusionTests(unittest.TestCase):
    def setUp(self):
        HD_CONFIG.enabled = True
        HD_CONFIG.start_sigma = 10
        HD_CONFIG.end_sigma = 0
        HD_CONFIG.use_blocks = {("input", 3)}

    def tearDown(self):
        HD_CONFIG.enabled = False
        HD_CONFIG.use_blocks = None

    def test_sdxl_high_selects_real_downsample(self):
        enabled, blocks, _, _, _ = configure_blocks("SDXL", "high")
        self.assertTrue(enabled)
        self.assertEqual(blocks, ("3", "5"))

    def test_sdxl_low_returns_patcher(self):
        patcher = _Patcher()
        self.assertIs(apply_rau_net_simple("SDXL", "low", "bicubic", "bicubic", patcher), patcher)

    def test_no_raunet_blocks_does_not_crash(self):
        HD_CONFIG.use_blocks = None
        self.assertFalse(HD_CONFIG.check({"block": ("input", 3), "sigmas": torch.tensor([5.0])}))

    def test_downsample_preserves_manual_cast(self):
        layer = HDDownsample(4, True)
        layer.op = _CastingConv(4, 4, 3, stride=2, padding=1).half()
        x = torch.zeros(1, 4, 16, 16)
        options = {"block": ("input", 3), "sigmas": torch.tensor([5.0])}
        self.assertEqual(layer(x, transformer_options=options).shape[-2:], (4, 4))

    def test_keyword_transformer_options_reach_layer(self):
        layer = HDDownsample(4, True)
        x = torch.zeros(1, 4, 16, 16)
        options = {"block": ("input", 3), "sigmas": torch.tensor([5.0])}
        positional = hd_forward_timestep_embed((layer,), x, None, None, options)
        keyword = hd_forward_timestep_embed((layer,), x, None, transformer_options=options)
        self.assertEqual(positional.shape, keyword.shape)

    def test_odd_attention_shape_round_trip(self):
        for height, width, original in ((33, 32, (132, 128)), (17, 16, (132, 128))):
            with self.subTest(shape=(height, width)):
                x = torch.arange(height * width * 4, dtype=torch.float32).reshape(1, height * width, 4)
                args = get_window_args(x, original, 1)
                self.assertEqual(args[2:], (height, width))
                self.assertTrue(torch.equal(window_reverse(window_partition(x, *args), *args), x))

    def test_proxy_supports_non_convolution_upsample(self):
        self.assertEqual(ProxyUpsample(4, False)(torch.zeros(1, 4, 4, 4)).shape[-2:], (8, 8))

    def test_raunet_runs_complete_three_level_unet(self):
        class Patcher:
            model = types.SimpleNamespace(predictor=types.SimpleNamespace(percent_to_sigma=lambda p: 10 * (1 - p)))

            def __init__(self):
                self.patches = {}

            def clone(self):
                return self

            def set_model_input_block_patch(self, patch):
                self.patches["input_block_patch"] = [patch]

            def set_model_output_block_patch(self, patch):
                self.patches["output_block_patch"] = [patch]

        model = IntegratedUNet2DConditionModel(
            in_channels=4,
            model_channels=32,
            out_channels=4,
            num_res_blocks=2,
            channel_mult=(1, 2, 4),
            num_heads=1,
            use_spatial_transformer=False,
            transformer_depth=[0] * 6,
            transformer_depth_output=[0] * 9,
            transformer_depth_middle=-1,
        )
        patcher = Patcher()
        apply_unet_patches()
        try:
            apply_rau_net(patcher, "3", "5", "sigma", 10.0, 0.0, False, "bicubic", 10.0, 0.0, "4", "5", "bicubic")
            with torch.no_grad():
                output = model(
                    torch.zeros(1, 4, 32, 32),
                    torch.tensor([500.0]),
                    transformer_options={"sigmas": torch.tensor([7.0]), "patches": patcher.patches},
                )
            self.assertEqual(output.shape, (1, 4, 32, 32))

            apply_rau_net(patcher, "3,6", "5,2", "sigma", 10.0, 0.0, False, "bicubic", 10.0, 0.0, "4", "5", "bicubic")
            with torch.no_grad():
                output = model(
                    torch.zeros(1, 4, 64, 64),
                    torch.tensor([500.0]),
                    transformer_options={"sigmas": torch.tensor([7.0]), "patches": patcher.patches},
                )
            self.assertEqual(output.shape, (1, 4, 64, 64))
        finally:
            remove_unet_patches()

    def test_attention_runs_complete_three_level_unet(self):
        class Patcher:
            model = types.SimpleNamespace(predictor=types.SimpleNamespace(percent_to_sigma=lambda p: 10 * (1 - p)))

            def __init__(self):
                self.patches = {}

            def clone(self):
                return self

            def set_model_attn1_patch(self, patch):
                self.patches["attn1_patch"] = [patch]

            def set_model_attn1_output_patch(self, patch):
                self.patches["attn1_output_patch"] = [patch]

        model = IntegratedUNet2DConditionModel(
            in_channels=4,
            model_channels=32,
            out_channels=4,
            num_res_blocks=2,
            channel_mult=(1, 2, 4),
            num_heads=1,
            use_spatial_transformer=True,
            context_dim=32,
            transformer_depth=[0, 0, 1, 1, 0, 0],
            transformer_depth_output=[0, 0, 0, 1, 1, 1, 0, 0, 0],
            transformer_depth_middle=-1,
        )
        patcher = Patcher()
        apply_mswmsaa_attention(patcher, "4,5", "", "3,4,5", "sigma", 10.0, 0.0)
        original_attention = forge_unet.attention_function
        forge_unet.attention_function = lambda q, k, v, heads, mask: F.scaled_dot_product_attention(q, k, v)
        try:
            with torch.no_grad():
                output = model(
                    torch.zeros(1, 4, 32, 32),
                    torch.tensor([500.0]),
                    context=torch.zeros(1, 77, 32),
                    transformer_options={"sigmas": torch.tensor([7.0]), "patches": patcher.patches},
                )
            self.assertEqual(output.shape, (1, 4, 32, 32))
        finally:
            forge_unet.attention_function = original_attention


class _Sampler:
    def sample(self, *args, **kwargs):
        return "txt2img"

    def sample_img2img(self, *args, **kwargs):
        return "img2img"


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        events = self.events = []
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "forge_hidiffusion.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        script_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ForgeHiDiffusion"
        )
        namespace = {
            "scripts": types.SimpleNamespace(Script=object),
            "StableDiffusionProcessing": object,
            "apply_unet_patches": lambda: events.append("apply"),
            "remove_unet_patches": lambda: events.append("remove"),
            "apply_rau_net_simple": lambda *args: events.append("raunet") or args[-1],
            "apply_rau_net": lambda *args: events.append(("raunet_advanced", args[-3], args[-2])) or args[0],
            "apply_mswmsaa_attention_simple": lambda *args: events.append("attention") or args[-1],
            "logger": logging.getLogger("hidiffusion-test"),
            "remove_current_script_callbacks": lambda: None,
            "HDDownsample": HDDownsample,
            "HDUpsample": HDUpsample,
            "parse_blocks": parse_blocks,
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("validate_blocks", "validate_raunet_pairs", "preset_pooling_settings")
        ]
        exec(compile(ast.Module(body=[*helpers, script_class], type_ignores=[]), str(script_path), "exec"), namespace)
        self.namespace = namespace
        self.script = namespace["ForgeHiDiffusion"]()
        self.p = types.SimpleNamespace(
            sampler=_Sampler(),
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=_Patcher())),
            extra_generation_params={},
            is_hr_pass=False,
        )

    def args(self, enabled=True, raunet=True, attention=False):
        return (
            enabled,
            "SDXL",
            raunet,
            False,
            "high (1536-2048)",
            "3",
            "5",
            "percent",
            0.0,
            0.45,
            False,
            "bicubic",
            0.3,
            "4",
            "5",
            0.0,
            "bicubic",
            attention,
            False,
            "4,5",
            "",
            "3,4,5",
            "percent",
            0.2,
            1.0,
        )

    def test_attention_only_does_not_patch_unet_globally(self):
        self.script.process_before_every_sampling(self.p, *self.args(raunet=False, attention=True))
        self.assertEqual(self.events, ["attention"])

    def test_txt2img_and_img2img_restore_after_sampling(self):
        for method in ("sample", "sample_img2img"):
            with self.subTest(method=method):
                self.script.process_before_every_sampling(self.p, *self.args())
                self.assertTrue(self.script.patch_applied)
                self.assertEqual(getattr(self.p.sampler, method)(), "txt2img" if method == "sample" else "img2img")
                self.assertFalse(self.script.patch_applied)
                self.assertEqual(self.events[-1], "remove")

    def test_disable_hr_and_setup_error_restore(self):
        self.script.process_before_every_sampling(self.p, *self.args())
        self.script.process_before_every_sampling(self.p, *self.args(enabled=False))
        self.assertFalse(self.script.patch_applied)
        self.assertEqual(self.events[-1], "remove")
        self.p.is_hr_pass = True
        self.script.process_before_every_sampling(self.p, *self.args())
        self.assertFalse(self.script.patch_applied)
        self.p.is_hr_pass = False

        def fail(*args):
            raise ValueError("setup failed")

        self.namespace["apply_rau_net_simple"] = fail
        with self.assertRaisesRegex(ValueError, "setup failed"):
            self.script.process_before_every_sampling(self.p, *self.args())
        self.assertFalse(self.script.patch_applied)
        self.assertEqual(self.events[-1], "remove")

    def test_hires_fix_opt_in_applies_each_pass_and_quick_upscale(self):
        args = (*self.args(), False, True)
        self.script.process_before_every_sampling(self.p, *args)
        self.assertTrue(self.script.patch_applied)
        self.p.sampler.sample()
        self.assertFalse(self.script.patch_applied)

        self.p.is_hr_pass = True
        self.script.process_before_every_sampling(self.p, *args)
        self.assertTrue(self.script.patch_applied)
        self.p.sampler.sample_img2img()
        self.assertFalse(self.script.patch_applied)

        self.p.txt2img_upscale = True
        self.script.process_before_every_sampling(self.p, *args)
        self.assertTrue(self.script.patch_applied)
        self.p.sampler.sample_img2img()
        self.assertFalse(self.script.patch_applied)
        self.assertTrue(self.p.extra_generation_params["hidiffusion_hires_fix"])

        self.script.process_before_every_sampling(self.p, *self.args())
        self.assertFalse(self.script.patch_applied)

    def test_sampling_error_and_postprocess_restore(self):
        def fail(*args, **kwargs):
            raise RuntimeError("sampling failed")

        self.p.sampler.sample = fail
        self.script.process_before_every_sampling(self.p, *self.args())
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            self.p.sampler.sample()
        self.assertFalse(self.script.patch_applied)
        self.assertIs(self.p.sampler.sample, fail)

        self.script.process_before_every_sampling(self.p, *self.args())
        self.script.postprocess(self.p, None)
        self.assertFalse(self.script.patch_applied)
        self.assertIs(self.p.sampler.sample, fail)

    def test_sdxl_custom_main_ignores_stale_ca_until_opted_in(self):
        inputs = [[torch.nn.Identity()] for _ in range(9)]
        outputs = [[torch.nn.Identity()] for _ in range(9)]
        for index in (3, 6):
            inputs[index] = [HDDownsample(4, True)]
        for index in (2, 5):
            outputs[index] = [HDUpsample(4, True)]
        model = types.SimpleNamespace(input_blocks=inputs, output_blocks=outputs)
        patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(
                diffusion_model=model,
                predictor=types.SimpleNamespace(percent_to_sigma=lambda percent: 10 * (1 - percent)),
            )
        )
        patcher.clone = lambda: patcher
        self.p.sd_model.forge_objects.unet = patcher
        args = list(self.args())
        args[3], args[5], args[6], args[14] = True, "3,6", "5,2", "8"
        args.append(False)
        self.script.process_before_every_sampling(self.p, *args)
        self.assertIn(("raunet_advanced", "4", "5"), self.events)
        self.assertEqual(self.p.extra_generation_params["raunet_ca_output_blocks"], "5")

        args[-1] = True
        with self.assertRaisesRegex(ValueError, "Cross-Attention Settings.*CA output blocks are 8"):
            self.script.process_before_every_sampling(self.p, *args)
        self.assertFalse(self.script.patch_applied)


class BlockValidationTests(unittest.TestCase):
    def test_custom_blocks_reject_sdxl_nonsampling_layers(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "forge_hidiffusion.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        validator = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_blocks"
        )
        namespace = {"parse_blocks": parse_blocks}
        exec(compile(ast.Module(body=[validator], type_ignores=[]), str(script_path), "exec"), namespace)
        model = types.SimpleNamespace(
            input_blocks=[[], [], [], [HDDownsample(4, True)], [torch.nn.Identity()]],
            output_blocks=[[], [], [], [], [], [HDUpsample(4, True)]],
        )
        patcher = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=model))
        check = namespace["validate_blocks"]
        check(patcher, (("RauNet input", "input", "3", HDDownsample), ("RauNet output", "output", "5", HDUpsample)))
        with self.assertRaisesRegex(ValueError, "no HDDownsample"):
            check(patcher, (("RauNet input", "input", "4", HDDownsample),))
        with self.assertRaisesRegex(ValueError, "does not exist"):
            check(patcher, (("RauNet output", "output", "8", HDUpsample),))

    def test_custom_raunet_pairs_follow_forge_skip_order(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "forge_hidiffusion.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        validator = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_raunet_pairs"
        )
        namespace = {"parse_blocks": parse_blocks}
        exec(compile(ast.Module(body=[validator], type_ignores=[]), str(script_path), "exec"), namespace)
        patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(input_blocks=[()] * 9))
        )
        check = namespace["validate_raunet_pairs"]
        check(patcher, "3,6", "2,5", "4", "5")
        with self.assertRaisesRegex(ValueError, "require output blocks 5"):
            check(patcher, "3", "4", "4", "5")
        with self.assertRaisesRegex(ValueError, "require output blocks 5"):
            check(patcher, "3", "5", "4", "4")


class UIControlTests(unittest.TestCase):
    def test_sdxl_defaults_and_time_mode_callbacks(self):
        class Component:
            def __init__(self, *args, **kwargs):
                self.value = kwargs.get("value")
                self.callbacks = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def change(self, **kwargs):
                self.callbacks.append(kwargs)

            def click(self, **kwargs):
                self.callbacks.append(kwargs)

        fake_gradio = types.SimpleNamespace(
            **{
                name: Component
                for name in (
                    "Radio",
                    "Tab",
                    "Checkbox",
                    "Dropdown",
                    "Text",
                    "Slider",
                    "Group",
                    "Accordion",
                    "Markdown",
                    "HTML",
                    "Button",
                )
            },
            update=lambda **kwargs: kwargs,
        )
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "forge_hidiffusion.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        script_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ForgeHiDiffusion"
        )
        namespace = {
            "scripts": types.SimpleNamespace(Script=object),
            "StableDiffusionProcessing": object,
            "gr": fake_gradio,
            "InputAccordion": Component,
            "UPSCALE_METHODS": ("bicubic", "bilinear"),
        }
        exec(compile(ast.Module(body=[script_class], type_ignores=[]), str(script_path), "exec"), namespace)
        controls = namespace["ForgeHiDiffusion"]().ui()
        self.assertEqual(controls[1].value(), "SDXL")
        self.assertEqual((controls[5].value, controls[6].value), ("3", "5"))
        self.assertEqual((controls[13].value, controls[14].value), ("4", "5"))
        self.assertEqual((controls[19].value, controls[21].value), ("4,5", "3,4,5"))
        self.assertFalse(controls[25].value)
        self.assertEqual(controls[25].callbacks[0]["fn"](True), {"visible": True})
        self.assertFalse(controls[26].value)
        timestep = controls[7].callbacks[0]["fn"]("timestep")
        self.assertEqual([control["value"] for control in timestep], [999, 549, 999, 699])
        sigma = controls[22].callbacks[0]["fn"]("sigma")
        self.assertEqual([control["value"] for control in sigma], [100.0, 0.0])


if __name__ == "__main__":
    unittest.main()
