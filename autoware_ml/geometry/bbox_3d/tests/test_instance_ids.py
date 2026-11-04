"""Stable integer identities of annotation instance tokens."""

import unittest

from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids


class TestInstanceIds(unittest.TestCase):
    def test_equal_tokens_map_to_one_non_negative_id(self) -> None:
        ids = hash_instance_ids(["a3f1", "77c0", "a3f1"])

        self.assertEqual(ids.dtype.name, "int64")
        self.assertEqual(ids[0], ids[2])
        self.assertNotEqual(ids[0], ids[1])
        self.assertTrue((ids >= 0).all())
        # Independent of the process, unlike the salted built-in hash
        self.assertEqual(int(ids[0]), int(hash_instance_ids(["a3f1"])[0]))
