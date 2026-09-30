"""Regression for reporting a cell before all of its seeds have completed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from decompose import get_gram


class GramCacheTest(unittest.TestCase):
    def test_new_seed_is_included_without_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, cell = root / "reference", root / "cell"
            ref.mkdir()
            (cell / "seed1").mkdir(parents=True)
            np.save(ref / "logits.npy", np.zeros((2, 3), dtype=np.float32))
            np.save(cell / "seed1/delta.npy", np.ones((2, 3), dtype=np.float32))
            seeds, gram = get_gram(str(cell), str(ref))
            self.assertEqual(seeds, ["seed1"])
            self.assertEqual(gram.shape, (2, 1, 1))
            (cell / "seed2").mkdir()
            np.save(cell / "seed2/delta.npy", np.tile([0., 1., 2.], (2, 1)))
            seeds, gram = get_gram(str(cell), str(ref))
            self.assertEqual(seeds, ["seed1", "seed2"])
            self.assertEqual(gram.shape, (2, 2, 2))
            np.testing.assert_allclose(gram[:, 1, 1], 1 / 3)
            again, cached = get_gram(str(cell), str(ref))
            self.assertEqual(again, seeds)
            np.testing.assert_array_equal(cached, gram)


if __name__ == "__main__":
    unittest.main()
