"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from SAE.cache_activations_runner_unlearn_canvas import CacheActivationsRunner
from SAE.config import CacheActivationsRunnerConfig


def run():
    args = parse(CacheActivationsRunnerConfig)
    CacheActivationsRunner(args).run()


if __name__ == "__main__":
    run()
