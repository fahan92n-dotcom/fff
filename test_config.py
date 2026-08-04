"""Tests for environment configuration parsing."""

import unittest

import config


class TestPortParsing(unittest.TestCase):
    def test_valid_port(self):
        self.assertEqual(config._read_port("8080"), 8080)

    def test_non_numeric_port_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            config._read_port("invalid")

    def test_out_of_range_port_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between"):
            config._read_port("70000")


if __name__ == "__main__":
    unittest.main()
