import argparse
import os
import random
import sys
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from fastargs.validation import BoolAsInt, Folder, OneOf

sys.path.append("..")
sys.path.append("src")
from UnlearnCanvas_resources.const import theme_available

Section("overall", "Overall configs").params(
    task=Param(
        OneOf(["classifier", "sd_guidence", "P4D", "transfer"]),
        required=True,
        desc="Task type to attack",
    ),
    attacker=Param(
        OneOf(
            [
                "gcg",
                "text_grad",
                "hard_prompt",
                "hard_prompt_multi",
                "random",
                "seed_search",
                "no_attack",
            ]
        ),
        required=True,
        desc="Attack algorithm",
    ),
    logger=Param(OneOf(["json", "none"]), default="none", desc="Logger to use"),
    resume=Param(Folder(), default=None, desc="Path to resume"),
    seed=Param(int, default=0, desc="Random seed"),
)

Section("task", "General task configs").params(
    concept=Param(
        OneOf(theme_available),
        required=True,
        desc="Concept to attack",
    ),
    sld=Param(
        OneOf(["weak", "medium", "strong", "max"]), default=None, desc="SLD type"
    ),
    sld_concept=Param(str, default=None, desc="SLD concept to be unlearned"),
    negative_prompt=Param(str, default=None, desc="Negative prompt to be used"),
    model_name_or_path=Param(str, required=True, desc="Model directory"),
    # target_ckpt = Param(File(), required=True, desc='Target model checkpoint'),
    cache_path=Param(Folder(True), default=".cache", desc="Cache directory"),
    dataset_path=Param(Folder(), required=True, desc="Path to dataset"),
    criterion=Param(OneOf(["l1", "l2"]), default="l2", desc="Loss criterion"),
    classifier_dir=Param(Folder(), default=None, desc="Classifier directory"),
    sampling_step_num=Param(int, default=50, desc="Sampling step number"),
    style_latents_path=Param(str, required=True, desc="Style latents path"),
    sae_path=Param(str, required=True, desc="SAE path"),
    hookpoint=Param(str, required=True, desc="Hookpoint"),
    percentile=Param(float, required=True, desc="Percentile"),
    multiplier=Param(float, required=True, desc="Multiplier"),
    style_ckpt=Param(str, required=True, desc="Style checkpoint"),
    class_ckpt=Param(str, required=True, desc="Classifier checkpoint"),
    class_name=Param(str, required=True, desc="Class name"),
    cls_atk=Param(str, required=True, default="false", desc="Class attack"),
)

Section("attacker", "General attacker configs").params(
    insertion_location=Param(
        OneOf(["prefix_k", "suffix_k", "mid_k", "insert_k", "per_k_words"]),
        default="prefix_k",
        desc="Insertion location",
    ),
    k=Param(int, default=3, desc="k in insertion_location"),
    iteration=Param(int, default=40, desc="Number of iterations"),
    seed_iteration=Param(int, default=20, desc="Number of seed iterations"),
    eval_seed=Param(int, default=0, desc="Evaluation seed"),
    universal=Param(BoolAsInt(), default=False, desc="Universal attack"),
    attack_idx=Param(int, required=False, desc="Attack index"),
    sequential=Param(BoolAsInt(), default=False, desc="Sequential optimization"),
)

Section("attacker.gcg", "Zeroth-Order GCG").enable_if(
    lambda cfg: cfg["overall.attacker"] == "gcg"
).params(
    candidate_size=Param(int, default=256, desc="Candidate size"),
    search_size=Param(int, default=512, desc="Random search size"),
)

Section("attacker.hard_prompt", "Hard Prompt").enable_if(
    lambda cfg: cfg["overall.attacker"] == "hard_prompt"
).params(
    lr=Param(float, default=0.01, desc="Learning rate"),
    weight_decay=Param(float, default=0.1, desc="Weight decay"),
    num_data=Param(int, default=5, desc="Number of data to use"),
)

Section("attacker.no_attack", "No Attack").enable_if(
    lambda cfg: cfg["overall.attacker"] == "no_attack"
).params(
    dataset_path=Param(Folder(), required=True, desc="Path to dataset"),
)

Section("attacker.hard_prompt_multi", "Hard Prompt Multi").enable_if(
    lambda cfg: cfg["overall.attacker"] == "hard_prompt_multi"
).params(
    lr=Param(float, default=0.01, desc="Learning rate"),
    weight_decay=Param(float, default=0.1, desc="Weight decay"),
    num_data=Param(int, default=5, desc="Number of data to use"),
    batch_size=Param(int, default=5, desc="Batch size"),
    noise_sd=Param(float, default=0.1, desc="Noise standard deviation"),
)

Section("attacker.text_grad", "Text Gradient").enable_if(
    lambda cfg: cfg["overall.attacker"] == "text_grad"
).params(
    lr=Param(float, default=0.01, desc="Learning rate"),
    weight_decay=Param(float, default=0.1, desc="Weight decay"),
    rand_init=Param(BoolAsInt(), default=False, desc="Random initialization"),
)

Section("logger", "General logger configs").params(
    name=Param(
        str,
        default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        desc="Name of this run",
    ),
)

Section("logger.json", "JSON logger").enable_if(
    lambda cfg: cfg["overall.logger"] == "json"
).params(
    root=Param(Folder(True), default="files/logs", desc="Path to log folder"),
)


class Main:
    def __init__(self) -> None:
        self.make_config()
        self.setup_seed()
        self.init_task()
        self.init_attacker()
        self.init_logger()
        self.run()

    def make_config(self, quiet=False):
        self.config = get_current_config()
        parser = argparse.ArgumentParser("Stable Diffusion Text-level Attack")
        self.config.augment_argparse(parser)
        self.config.collect_argparse_args(parser)

        if self.config["overall.resume"] is not None:
            self.config.collect_config_file(
                os.path.join(self.config["overall.resume"], "config.json")
            )

        self.config.validate()
        if not quiet:
            self.config.summary()

    @param("overall.seed")
    def setup_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @param("overall.task")
    def init_task(self, task):
        kwargs = self.config.get_section("task")
        kwargs.update(self.config.get_section(f"task.{task}"))
        self.task = import_module(f"tasks.{task}_").get(**kwargs)

    @param("overall.attacker")
    def init_attacker(self, attacker):
        kwargs = self.config.get_section("attacker")
        kwargs.update(self.config.get_section(f"attacker.{attacker}"))
        self.attacker = import_module(f"attackers.{attacker}_").get(**kwargs)

    @param("overall.logger")
    def init_logger(self, logger):
        kwargs = self.config.get_section("logger")
        kwargs.update(self.config.get_section(f"logger.{logger}"))
        kwargs["config"] = self.config.get_all_config()
        self.logger = import_module(f"loggers.{logger}_").get(**kwargs)

    def run(self):
        self.attacker.run(self.task, self.logger)


if __name__ == "__main__":
    Main()
