"""Check projector numerics and hook routing without ComfyUI or a GPU."""

import ast
import math
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def attention_backend(q, k, v, heads):
    batch, queries, width = q.shape
    q, k, v = [tensor.reshape(batch, -1, heads, width // heads).transpose(1, 2) for tensor in (q, k, v)]
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, queries, width)


def load_perceiver(backend):
    tree = ast.parse((ROOT / "image_proj_models.py").read_text(encoding="utf-8-sig"))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"reshape_tensor", "PerceiverAttention"}]
    namespace = {"torch": torch, "nn": nn, "math": math, "optimized_attention": backend}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "image_proj_models.py", "exec"), namespace)
    return namespace["PerceiverAttention"]


def original_forward(model, image, latents):
    image, latents = model.norm1(image), model.norm2(latents)
    q = model.to_q(latents)
    k, v = model.to_kv(torch.cat((image, latents), dim=1)).chunk(2, dim=-1)
    batch, queries, width = q.shape
    q, k, v = [tensor.reshape(batch, -1, model.heads, model.dim_head).transpose(1, 2) for tensor in (q, k, v)]
    scale = model.dim_head ** -0.25
    scores = (q * scale) @ (k * scale).transpose(-2, -1)
    weights = scores.float().softmax(dim=-1).to(scores.dtype)
    output = (weights @ v).transpose(1, 2).reshape(batch, queries, width)
    return model.to_out(output)


class ProjectorAttentionTests(unittest.TestCase):
    def test_optimized_backend_matches_original_projection(self):
        backend = Mock(side_effect=attention_backend)
        perceiver = load_perceiver(backend)
        for dtype in (torch.float32, torch.float16):
            for batch, heads, queries, tokens in [(1, 12, 4, 257), (5, 20, 16, 273)]:
                with self.subTest(dtype=dtype, batch=batch, heads=heads):
                    torch.manual_seed(0)
                    model = perceiver(dim=64, dim_head=64, heads=heads).to(dtype=dtype)
                    image = torch.randn(batch, tokens, 64, dtype=dtype)
                    latents = torch.randn(batch, queries, 64, dtype=dtype)
                    backend.reset_mock()
                    with torch.no_grad():
                        expected = original_forward(model, image, latents)
                        actual = model(image, latents)
                    backend.assert_called_once()
                    q, k, v, actual_heads = backend.call_args.args
                    self.assertEqual(q.shape, (batch, queries, heads * 64))
                    self.assertEqual(k.shape, (batch, tokens + queries, heads * 64))
                    self.assertEqual(v.shape, k.shape)
                    self.assertEqual(actual_heads, heads)
                    tolerance = 2e-3 if dtype == torch.float16 else 1e-5
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    self.assertEqual(set(model.state_dict()), {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "to_q.weight", "to_kv.weight", "to_out.weight"})

    def test_middle_hooks_target_comfyui_block_zero(self):
        tree = ast.parse((ROOT / "IPAdapterPlus.py").read_text(encoding="utf-8-sig"))
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ipadapter_execute")
        routing = next(node for node in execute.body if isinstance(node, ast.If) and ast.unparse(node.test) == "not is_sdxl")
        for is_sdxl, expected_count in [(False, 1), (True, 10)]:
            with self.subTest(is_sdxl=is_sdxl):
                keys = []
                namespace = {"model": object(), "is_sdxl": is_sdxl, "number": 0, "patch_kwargs": {}, "set_model_patch_replace": lambda model, kwargs, key: keys.append(key)}
                exec(compile(ast.Module(body=[routing], type_ignores=[]), "IPAdapterPlus.py", "exec"), namespace)
                middle = [key for key in keys if key[0] == "middle"]
                self.assertEqual(len(middle), expected_count)
                self.assertTrue(all(key[1] == 0 for key in middle))
                if is_sdxl:
                    self.assertEqual([key[2] for key in middle], list(range(10)))


if __name__ == "__main__":
    unittest.main()
