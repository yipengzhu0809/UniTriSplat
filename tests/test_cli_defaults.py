import unittest
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams


class CommandLineDefaultTests(unittest.TestCase):
    def test_safe_public_defaults(self):
        parser = ArgumentParser()
        ModelParams(parser)
        PipelineParams(parser)

        args = parser.parse_args([])
        self.assertEqual(args.data_device, "cpu")
        self.assertFalse(args.use_ring)

    def test_ring_query_is_opt_in(self):
        parser = ArgumentParser()
        ModelParams(parser)
        PipelineParams(parser)

        args = parser.parse_args(["--use_ring"])
        self.assertTrue(args.use_ring)


if __name__ == "__main__":
    unittest.main()
