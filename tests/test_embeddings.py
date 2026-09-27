"""Validate API response ordering and vector integrity without network calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.common.embeddings import APIEmbeddingModel


def response(*vectors):
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=vector) for index, vector in vectors
        ]
    )


class APIEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.model = APIEmbeddingModel.__new__(APIEmbeddingModel)
        self.model.client = Mock()
        self.model.model_name = "test-embed"

    def test_batches_preserve_input_order_and_normalize(self):
        create = self.model.client.embeddings.create
        create.side_effect = [
            response((1, [0, 4]), (0, [3, 4])),
            response((0, [0, 0])),
        ]
        vectors = self.model.encode(["first", "", "third"], batch_size=2)
        np.testing.assert_allclose(vectors, [[0.6, 0.8], [0, 1], [0, 0]])
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(create.call_args_list[0].kwargs["input"], ["first", " "])
        self.assertEqual(create.call_args_list[1].kwargs["input"], ["third"])

    def test_rejects_missing_or_duplicate_indices(self):
        for invalid in (response((0, [1, 0])), response((0, [1, 0]), (0, [0, 1]))):
            with self.subTest(response=invalid):
                self.model.client.embeddings.create.return_value = invalid
                with self.assertRaisesRegex(RuntimeError, "input indices"):
                    self.model.encode(["first", "second"])

    def test_rejects_invalid_or_inconsistent_vectors(self):
        for invalid in (
            response((0, [1, float("nan")]), (1, [0, 1])),
            response((0, []), (1, [])),
            response((0, [1, 0]), (1, [1])),
        ):
            with self.subTest(response=invalid):
                self.model.client.embeddings.create.return_value = invalid
                with self.assertRaisesRegex(RuntimeError, "vectors"):
                    self.model.encode(["first", "second"])

    def test_empty_inputs_and_nonpositive_batches_do_not_call_api(self):
        self.assertEqual(self.model.encode([]).shape, (0, 0))
        with self.assertRaises(ValueError):
            self.model.encode(["first"], batch_size=0)
        self.model.client.embeddings.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
