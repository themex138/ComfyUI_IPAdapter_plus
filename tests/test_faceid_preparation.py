"""Check image-embedding preparation without loading vision or face models."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


def load_preparation(encoder):
    path = Path(__file__).resolve().parents[1] / "IPAdapterPlus.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ipadapter_execute")
    # Execute the actual preparation branch after face extraction, before projection.
    branch = next(node for node in execute.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "image is not None")
    namespace = {"torch": torch, "encode_image_masked": encoder}

    def prepare(is_faceid=False, is_plus=False, composition=None, negative=None):
        values = dict(namespace, image=object(), clipvision=object(),
                      image_composition=composition, image_negative=negative,
                      is_faceid=is_faceid, is_plus=is_plus,
                      face_cond_embeds=torch.tensor([[1., 2.]]),
                      img_comp_cond_embeds=None, encode_batch_size=1,
                      enhance_tiles=2, enhance_ratio=0.5, clipvision_size=224)
        exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec"), values)
        return values

    return prepare


class FaceIdPreparationTests(unittest.TestCase):
    def setUp(self):
        self.embeds = SimpleNamespace(image_embeds=torch.tensor([[3., 4.]]),
                                      penultimate_hidden_states=torch.tensor([[[5., 6.]]]))
        self.encoder = Mock(return_value=self.embeds)
        self.prepare = load_preparation(self.encoder)

    def test_plain_faceid_uses_face_embeddings_without_clip(self):
        values = self.prepare(is_faceid=True, negative=object())
        self.encoder.assert_not_called()
        self.assertIs(values["img_cond_embeds"], values["face_cond_embeds"])
        self.assertTrue(torch.equal(values["img_uncond_embeds"], torch.zeros_like(values["face_cond_embeds"])))
        self.assertIsNotNone(values["image"])

    def test_faceid_composition_still_encodes_and_uses_composition(self):
        composition = object()
        values = self.prepare(is_faceid=True, composition=composition)
        self.encoder.assert_called_once_with(values["clipvision"], composition,
                                             batch_size=1, tiles=2, ratio=0.5, clipvision_size=224)
        self.assertIs(values["img_comp_cond_embeds"], self.embeds.image_embeds)
        self.assertIs(values["img_cond_embeds"], values["face_cond_embeds"])

    def test_plus_and_faceid_plus_keep_both_clip_encodes(self):
        for faceid in (False, True):
            with self.subTest(faceid=faceid):
                self.encoder.reset_mock()
                values = self.prepare(is_faceid=faceid, is_plus=True, negative=object())
                self.assertEqual(self.encoder.call_count, 2)
                self.assertIs(values["img_cond_embeds"], self.embeds.penultimate_hidden_states)
                self.assertIs(values["img_uncond_embeds"], self.embeds.penultimate_hidden_states)

    def test_standard_adapter_keeps_positive_negative_and_composition(self):
        values = self.prepare(composition=object(), negative=object())
        self.assertEqual(self.encoder.call_count, 3)
        self.assertIs(values["img_cond_embeds"], self.embeds.image_embeds)
        self.assertIs(values["img_uncond_embeds"], self.embeds.image_embeds)
        self.assertIs(values["img_comp_cond_embeds"], self.embeds.image_embeds)
        self.assertIsNone(values["image"])


if __name__ == "__main__":
    unittest.main()
