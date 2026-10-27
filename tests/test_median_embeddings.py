"""Check median reference combination for FaceID, general and saved embeddings."""
import ast
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'IPAdapterPlus.py').read_text(encoding='utf-8-sig'))

def definitions(*names):
    ns = {'torch': torch}
    nodes = [n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'IPAdapterPlus.py', 'exec'), ns)
    return ns

class MedianEmbeddingTests(unittest.TestCase):
    def test_outliers_even_counts_and_dtype(self):
        median = definitions('median_embedding')['median_embedding']
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                odd = torch.tensor([[0, 20], [2, 40], [100, 30]], dtype=dtype)
                even = torch.tensor([[0, 20], [2, 40], [100, 30], [4, 10]], dtype=dtype)
                torch.testing.assert_close(median(odd), torch.tensor([[2, 30]], dtype=dtype))
                torch.testing.assert_close(median(even), torch.tensor([[3, 25]], dtype=dtype))
                self.assertEqual(median(even).dtype, dtype)

    def test_face_and_image_reference_combination(self):
        execute = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'ipadapter_execute')
        branch = next(n for n in execute.body if isinstance(n, ast.If) and 'combine_embeds' in ast.unparse(n.test) and 'unfold_batch' in ast.unparse(n.test))
        median = definitions('median_embedding')['median_embedding']
        reference = torch.tensor([[[0., 10.]], [[2., 20.]], [[100., 30.]]])
        for face, composition in ((None, None), (reference, reference)):
            ns = dict(torch=torch, median_embedding=median, combine_embeds='median', unfold_batch=False,
                      img_cond_embeds=reference, img_uncond_embeds=reference,
                      face_cond_embeds=face, img_comp_cond_embeds=composition)
            exec(compile(ast.Module(body=[branch], type_ignores=[]), 'IPAdapterPlus.py', 'exec'), ns)
            torch.testing.assert_close(ns['img_cond_embeds'], torch.tensor([[[2., 20.]]]))
            if face is not None:
                torch.testing.assert_close(ns['face_cond_embeds'], ns['img_cond_embeds'])
                torch.testing.assert_close(ns['img_comp_cond_embeds'], ns['img_cond_embeds'])

    def test_saved_embedding_combine(self):
        ns = definitions('median_embedding', 'IPAdapterCombineEmbeds')
        result = ns['IPAdapterCombineEmbeds']().batch(torch.tensor([[0., 10.]]), 'median',
                                                     torch.tensor([[2., 20.]]), torch.tensor([[100., 30.]]))[0]
        torch.testing.assert_close(result, torch.tensor([[2., 20.]]))

if __name__ == '__main__':
    unittest.main()
