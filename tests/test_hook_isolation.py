"""Run with python -m unittest discover -s tests; no ComfyUI or GPU required."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_patch_functions():
    # Extract the real hook code without importing its GPU/model dependencies.
    namespace = {"copy": copy, "ipadapter_attention": object()}
    for filename, name in [
        ("CrossAttentionPatch.py", "Attn2Replace"),
        ("IPAdapterPlus.py", "set_model_patch_replace"),
    ]:
        path = ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        definition = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["set_model_patch_replace"]


def clone_model(model):
    # ComfyUI copies option containers but preserves the attention hook objects.
    patches = model.model_options["transformer_options"].get("patches_replace", {}).get("attn2", {})
    memo = {id(patch): patch for patch in patches.values()}
    return SimpleNamespace(model_options=copy.deepcopy(model.model_options, memo))


class HookIsolationTests(unittest.TestCase):
    def setUp(self):
        self.attach = load_patch_functions()
        self.key = ("input", 4, 0)
        self.base = SimpleNamespace(model_options={"transformer_options": {}})
        self.conditioning = object()
        self.attach(self.base, {"weight": 1.0, "cond": self.conditioning}, self.key)

    def hook(self, model):
        return model.model_options["transformer_options"]["patches_replace"]["attn2"][self.key]

    def test_repeated_evaluation_and_intentional_chaining(self):
        for _ in range(5):
            combined = clone_model(self.base)
            self.attach(combined, {"weight": 0.25}, self.key)
            self.assertEqual(len(self.hook(self.base).callback), 1)
            self.assertEqual(len(self.hook(combined).callback), 2)
            self.assertIs(self.hook(combined).kwargs[0]["cond"], self.conditioning)

        third = clone_model(combined)
        self.attach(third, {"weight": 0.5}, self.key)
        self.assertEqual(len(self.hook(combined).callback), 2)
        self.assertEqual([item["weight"] for item in self.hook(third).kwargs], [1.0, 0.25, 0.5])

    def test_sibling_branches_remain_independent(self):
        left, right = clone_model(self.base), clone_model(self.base)
        self.attach(left, {"weight": 0.25}, self.key)
        self.attach(right, {"weight": 0.75}, self.key)
        self.assertEqual([item["weight"] for item in self.hook(left).kwargs], [1.0, 0.25])
        self.assertEqual([item["weight"] for item in self.hook(right).kwargs], [1.0, 0.75])
        self.assertEqual(len(self.hook(self.base).callback), 1)

    def test_device_cache_is_rebuilt_for_the_new_hook(self):
        cache = [{"weight": 1.0}]
        self.hook(self.base).multigpu_kwargs["cuda:1"] = cache
        combined = clone_model(self.base)
        self.attach(combined, {"weight": 0.25}, self.key)
        self.assertIs(self.hook(self.base).multigpu_kwargs["cuda:1"], cache)
        self.assertEqual(self.hook(combined).multigpu_kwargs, {})
        self.assertEqual(len(self.hook(combined).get_multigpu_kwargs("cuda:1")), 2)


if __name__ == "__main__":
    unittest.main()
