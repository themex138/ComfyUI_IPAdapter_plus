"""Check per-forward sigma reads and scheduling without ComfyUI or CUDA."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_functions():
    namespace = {"optimized_attention": lambda q, k, v, heads: torch.zeros_like(q)}
    for filename, names in [
        ("CrossAttentionPatch.py", {"Attn2Replace", "ipadapter_sigma_wrapper"}),
        ("IPAdapterPlus.py", {"set_model_sigma_wrapper"}),
    ]:
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec"), namespace)
    return namespace


class CountingSigma:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def __getitem__(self, index):
        assert index == 0
        return self

    def item(self):
        self.reads += 1
        return self.value


class SigmaScheduleTests(unittest.TestCase):
    def setUp(self):
        self.functions = load_functions()
        self.wrapper = self.functions["ipadapter_sigma_wrapper"]
        self.q = torch.zeros(1, 1, 1)
        self.hook = self.functions["Attn2Replace"](
            lambda out, q, k, v, options, **kwargs: torch.ones_like(q),
            sigma_start=0.75, sigma_end=0.25,
        )

    def evaluate(self, options):
        return self.hook(self.q, self.q, self.q, {"n_heads": 1, **options}).item()

    def test_one_scalar_read_for_all_blocks_and_fresh_read_each_forward(self):
        sigma = CountingSigma(0.5)
        options = {"sigmas": sigma, "ipadapter_sigma": -1}

        def model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs):
            self.assertIs(transformer_options["sigmas"], sigma)
            return [self.evaluate(transformer_options.copy()) for _ in range(70)]

        self.assertEqual(self.wrapper(model, None, None, transformer_options=options), [1] * 70)
        sigma.value = 0.9
        self.assertEqual(self.wrapper(model, None, None, transformer_options=options), [0] * 70)
        self.assertEqual(sigma.reads, 2)
        self.assertEqual(options["ipadapter_sigma"], -1)

    def test_inclusive_boundaries_and_legacy_fallback_match(self):
        for value, expected in [(0.249, 0), (0.25, 1), (0.5, 1), (0.75, 1), (0.751, 0)]:
            with self.subTest(value=value):
                options = {"sigmas": torch.tensor([value], dtype=torch.float64)}
                legacy = self.evaluate(options)
                wrapped = self.wrapper(
                    lambda x, t, a, b, c, opts: self.evaluate(opts),
                    None, None, transformer_options=options,
                )
                self.assertEqual(legacy, expected)
                self.assertEqual(wrapped, expected)

    def test_absent_sigmas_and_executor_arguments_are_preserved(self):
        args = [object() for _ in range(5)]
        marker = object()

        def executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs):
            self.assertEqual([x, t, c_concat, c_crossattn, control], args)
            self.assertIs(kwargs["marker"], marker)
            self.assertEqual(self.evaluate(transformer_options), self.evaluate({}))
            return marker

        self.assertIs(self.wrapper(executor, *args, marker=marker), marker)

    def test_wrapper_registration_is_idempotent_and_preserves_other_wrappers(self):
        register = self.functions["set_model_sigma_wrapper"]
        other = object()
        wrappers = {("apply_model", "other"): [other]}
        model = SimpleNamespace(
            get_wrappers=lambda kind, key: wrappers.get((kind, key), []),
            add_wrapper_with_key=lambda kind, key, callback: wrappers.setdefault((kind, key), []).append(callback),
        )
        extension = ModuleType("comfy.patcher_extension")
        extension.WrappersMP = SimpleNamespace(APPLY_MODEL="apply_model")
        with patch.dict(sys.modules, {"comfy": ModuleType("comfy"), "comfy.patcher_extension": extension}):
            register(model)
            register(model)
            self.assertEqual(wrappers[("apply_model", "ipadapter_sigma")], [self.wrapper])
            self.assertEqual(wrappers[("apply_model", "other")], [other])
            extension.WrappersMP = SimpleNamespace()
            register(model)
        register(SimpleNamespace())
        with patch.dict(sys.modules, {"comfy": ModuleType("comfy"), "comfy.patcher_extension": None}):
            register(model)


if __name__ == "__main__":
    unittest.main()
