"""Compare the real mask path with its previous dense-channel implementation."""

import ast
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


class AttentionMaskTests(unittest.TestCase):
    def test_broadcast_matches_dense_mask(self):
        source = (Path(__file__).resolve().parents[1] / "CrossAttentionPatch.py").read_text()
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ipadapter_attention")
        block = next(n for n in function.body if isinstance(n, ast.If) and ast.unparse(n.test) == "mask is not None")
        code = ast.unparse(block)
        old_code = code.replace("mask.view(mask.shape[0], -1, 1)", "mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])")
        def resize_batch(tensor, size):
            return torch.cat((tensor, tensor[-1:].repeat(max(0, size - tensor.shape[0]), 1, 1)))[:size]
        for seq, height, width in [(64, 8, 8), (65, 8, 9), (70, 2, 17)]:
            for batch, ad in [(1, None), (2, None), (2, {"sub_idxs": [1, 3], "full_length": 4})]:
                for mask_batch in (1, 4):
                    with self.subTest(seq=seq, batch=batch, ad=ad, mask_batch=mask_batch):
                        output = torch.randn(batch * 2, seq, 32)
                        mask = torch.rand(mask_batch, 13, 19)
                        inputs = dict(torch=torch, F=F, math=math, tensor_to_size=resize_batch,
                                      mask=mask, oh=height, ow=width, seq_len=seq,
                                      ad_params=ad, batch_prompt=batch, cond_or_uncond=[0, 1],
                                      out=output, out_ip=output)
                        actual, expected = inputs.copy(), inputs.copy()
                        exec(code, actual)
                        exec(old_code, expected)
                        torch.testing.assert_close(actual["out_ip"], expected["out_ip"], rtol=0, atol=0)
                        self.assertEqual(actual["mask"].shape[-1], 1)


if __name__ == "__main__":
    unittest.main()
