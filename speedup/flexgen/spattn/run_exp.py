import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights, init_opt_weights
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import, TorchTensor)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

from infinigen.skewing_controller import weight_bias_concat
from infinigen.kv_selection_controller import select_kv
from infinigen.partial_weight_generation_controller import set_partial_cache, set_partial_weight

import argparse
import datetime
import json
import logging
import os
import socket
import subprocess
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm

# from offloading.offload_model import load_gptq_offloaded_model, load_offloaded_model
from spattn import SpecExecBeams, SpecExecBase, SpecInfer, utils
# import engine
from spattn.utils import colored
import engine

device = torch.device("cuda:0")
device1 = torch.device("cuda:1")
_DEFAULT_DEVICE_SIZE = 2
DISPLAY_WIDTH = 160
pd.set_option("display.width", DISPLAY_WIDTH)
pd.set_option("display.max_columns", 32)


class EngineSPA:
    """wrapper for regular transformers model with regular cache"""

    def __init__(self, model_name, max_len, dtype=torch.float16, device="cuda:0"):
        self.model_name = model_name
        self.max_len = max_len
        self.device = device
        if isinstance(model_name, str):
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        else:
            self.model = model_name
        self.config = self.model.config

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):
        # assert torch.equal(cache_position, torch.arange(cache_position[0], cache_position[-1] + 1, device=cache_position.device)), "reconsider use of cache_position in amask slicing"
        attention_mask = attention_mask[..., : cache_position.max() + 1]
        assert attention_mask.shape[-2] == input_ids.shape[-1]

        cache_position_models = ["llama"]

        if self.model.config.model_type in cache_position_models:
            output = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
                cache_position=cache_position,
            )
        else:
            assert torch.equal(cache_position, torch.arange(cache_position[0], cache_position[0] + cache_position.numel(), device=cache_position.device))

            output = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
            )

        self.kv_cache = output.past_key_values

        return output.logits

    @property
    def kv_len_used(self):
        if isinstance(self.kv_cache, transformers.DynamicCache):
            return self.kv_cache.get_seq_length()  # pass the call to DynamicCache
        else:  # if cache is in legacy form
            return 0 if self.kv_cache is None else self.kv_cache[0][0].shape[2]

    def clear_kv(self):
        self.kv_cache = DynamicCachePlus()

    def reorder_cache_tokens(self, source_token_idxs: torch.tensor, dest_token_idxs: torch.tensor = None):
        """Applies indices mask to KV cache or truncates it"""

        cache_size = self.kv_cache[0][0].shape[-2]  # replace with self.kv_len_used
        if source_token_idxs.dtype == torch.bool:
            source_token_idxs = torch.where(source_token_idxs)[0]

        left_edge = dest_token_idxs.min() if dest_token_idxs is not None else 0

        if source_token_idxs.max() >= cache_size:  # source includes elements outside of cache
            source_token_idxs = source_token_idxs[source_token_idxs < cache_size]
            dest_token_idxs = torch.arange(left_edge, left_edge + source_token_idxs.shape[-1], device=self.device)

        if dest_token_idxs is None:  # assumed that destination starts from cache beginning
            dest_token_idxs = torch.arange(source_token_idxs.shape[-1], device=self.device)

        new_cache = []
        for layer_cache_k, layer_cache_v in self.kv_cache:
            new_cache.append(
                (
                    torch.cat([layer_cache_k[:, :, :left_edge, :], layer_cache_k[:, :, source_token_idxs, :]], dim=-2),
                    torch.cat([layer_cache_v[:, :, :left_edge, :], layer_cache_v[:, :, source_token_idxs, :]], dim=-2),
                )
            )
        self.kv_cache = DynamicCachePlus.from_legacy_cache(tuple(new_cache))

    def set_max_len(self, new_max_len):
        if self.kv_cache is not None and self.kv_len_used > new_max_len:
            raise ValueError(f"Current cache size {self.kv_len_used()} is greater than new `max_len` {new_max_len}.")
        self.max_len = new_max_len

def create_spec_generator(
    model_name_0,
    model_name_1,
    draft_engine_class,
    target_opt_config,
    target_env,
    target_policy,
    gen_type="SX",
    offload=False,
    device_size=_DEFAULT_DEVICE_SIZE,
    check_tokenizer=False,
):
    """Creates a SpecGenerator object for different generation types.

    This function loads draft and target pre-trained language models specified by their names
    and creates a SpecBase subclass object based on the provided generation type.
    It also handles several configuration options like device placement and tokenizer verification.

    Args:
        model_name_0 (str): Name of the draft model.
        model_name_1 (str): Name of the target model.
        gen_type (str, optional): Generation type. Defaults to "SX" (SpecExec).
            Valid options include:
                - "SpecExecBase", : SpecExec generator
                - "SI", "spec_infer", "specinfer": SpecInfer generator
        offload (bool, optional): Whether to offload model 1 using offloading library. Defaults to False.
        device_size (int, optional): Device size for offloading. Defaults to `_DEFAULT_DEVICE_SIZE`.
        check_tokenizer (bool, optional): Whether to verify if both models have the same tokenizer. Defaults to False.

    Returns:
        SpecGenerator: An instance of a SpecBase subclass object based on the provided parameters.

    Raises:
        ValueError: If an invalid `gen_type` is provided.
    """

    if len(model_name_0.split("::")) == 2:
        model_name_0, rev_0 = model_name_0.split("::")
    else:
        rev_0 = "main"  # default in `from_pretrained()`

    if len(model_name_1.split("::")) == 2:
        model_name_1, rev_1 = model_name_1.split("::")
    else:
        rev_1 = "main"  # default in `from_pretrained()`

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_0, legacy=False)

    if check_tokenizer:
        # verify that the two models have the same tokenizer
        tokenizer_1 = transformers.AutoTokenizer.from_pretrained(model_name_1, legacy=False)
        vv0 = tokenizer.get_vocab()
        vv1 = tokenizer_1.get_vocab()

        ignored_tokens = ["[PAD]"]  # disregard these tokens when comparing the cokonizers' vocabs
        assert set(vv0.keys()).difference(ignored_tokens) == set(vv1.keys()).difference(ignored_tokens)
        for k in set(vv0.keys()).difference(ignored_tokens):
            assert vv0[k] == vv1[k]
        del tokenizer_1, vv0, vv1

    logger.info(f"Loading Model 0: `{model_name_0}`, {draft_engine_class=}")
    if draft_engine_class.lower() in ("es", "static", "enginestatic"):
        model_0 = transformers.AutoModelForCausalLM.from_pretrained(model_name_0, device_map=device, torch_dtype=torch.float16, revision=rev_0)
        draft_engine = engine.EngineStatic(model_0, max_len=args.tree_max_len)
    # elif draft_engine_class.lower() in ("esc", "staticcompiled", "enginestaticcompiled"):
    #     model_0 = transformers.AutoModelForCausalLM.from_pretrained(model_name_0, device_map=device, torch_dtype=torch.float16, revision=rev_0)
    #     draft_engine = engine.EngineStaticCompiled(model_0, max_len=args.tree_max_len)
    # elif draft_engine_class.lower() in ("ie", "inferenceengine"):
    #     draft_engine = engine.InferenceEngine(model_name_0, max_len=args.tree_max_len)
    elif draft_engine_class.lower() in ("padded", "inferenceenginepadded"):
        draft_engine = engine.InferenceEnginePadded(model_name_0, max_len=args.tree_max_len)
    elif draft_engine_class.lower() in ("er", "regular", "engineregular"):
        draft_engine = engine.EngineRegular(model_name_0, max_len=args.tree_max_len)
    else:
        raise ValueError(f"Unsupported engine class: {draft_engine_class} !")

    logger.info(f"Loading Model 1: `{model_name_1}`")
    gptq_max_input_length = 16384  # constant for GPTQ models

    # if offload:
    #     if "gptq" in model_name_1.lower():
    #         model_1 = load_gptq_offloaded_model(model_name_1, device_size=device_size, main_device=device, max_input_length=gptq_max_input_length)
    #     else:
    #         model_1 = load_offloaded_model(model_name_1, device_size=device_size, main_device=device)

    # else:
    #     model_1 = transformers.AutoModelForCausalLM.from_pretrained(model_name_1, device_map=device, torch_dtype=torch.float16, revision=rev_1)

    #     if "gptq" in model_name_1.lower():
    #         model_1_config = transformers.AutoConfig.from_pretrained(model_name_1)
    #         if getattr(model_1_config.quantization_config, "act_order", False) and (model_1_config.config.max_length < 16384):
    #             try:
    #                 from auto_gptq import exllama_set_max_input_length

    #                 model_1 = exllama_set_max_input_length(model_1, gptq_max_input_length)
    #                 print("set `exllama_set_max_input_length` OK")
    #             except (AttributeError, ValueError, ImportError):
    #                 # AttributeError may happen if GPTQ-quantized model has no attribute 'device_to_buffers'
    #                 # could be fixed by using code from post_init()
    #                 # ImportError resembles https://github.com/open-mmlab/mmdetection3d/issues/1152
    #                 logger.warning("Failed to set `exllama_set_max_input_length`")

    # target_engine = EngineStatic(model_1, max_len=args.tree_max_len)
    # target_engine = engine.EngineRegular(model_1, max_len=args.tree_max_len)
    target_engine = OptLM(target_opt_config, target_env, args.path, target_policy, args.partial_weight_ratio, args.alpha, args.max_num_kv)

    if gen_type.lower() in ("sx_base", "base", "sx2", "spec_exec_base", "specexecbase"):
        spec_generator = SpecExecBase(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("spec_exec_beams", "specexecbeams", "sx_beams"):
        spec_generator = SpecExecBeams(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sa", "a", "spec_adaptive", "specadaptive"):
        spec_generator = SpecAdaptive(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sf", "f", "spec_fixed", "specfixed"):
        spec_generator = SpecFixed(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("si", "spec_infer", "specinfer"):
        spec_generator = SpecInfer(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sis", "spec_infer_stems", "specinferstems"):
        spec_generator = SpecInferStems(draft_engine, target_engine, tokenizer)
    else:
        raise ValueError(f"unknown {gen_type=}")

    logger.info(f"Created spec_generator of type {gen_type}; Models: {model_name_0}, {model_name_1}")
    return spec_generator


def run_tests(
    spec_generator,
    dataset,
    args,
    max_budget=None,
    max_n_beams=None,
    max_beam_len=None,
    max_branch_width=None,
    min_log_prob=None,
    **kwargs,
):
    """runs uniform experiments from dataset using same set of parameters"""
    test_logs = []

    for i in range(args.dataset_start_index, min(args.dataset_start_index + args.n_tests, len(dataset))):
        prompt = dataset[i]
        _ = spec_generator.generate(
            prompt,
            max_n_beams=max_n_beams,
            max_beam_len=max_beam_len,
            max_new_tokens=args.max_new_tokens,
            branching=args.branching,
            max_budget=max_budget,
            max_branch_width=max_branch_width,
            replacement=args.replacement,
            verbose=args.verbose,
            temperature=args.temperature,
            draft_temperature=args.draft_temperature,
            top_p=args.top_p,
            min_log_prob=min_log_prob,
            seed=args.seed,
            tree_max_len=args.tree_max_len,
            **kwargs,
        )

        test_logs.append(spec_generator.summary)
        generated_text = spec_generator.tokenizer.decode(spec_generator.prefix_tokens[spec_generator.original_num_tokens :]).__repr__().strip("'")

        excl_keys = ["ver", "model_name_0", "model_name_1"]
        log1 = {k: v for k, v in spec_generator.summary.items() if k not in excl_keys}
        log1 = {"run": i, **log1, "text": generated_text[:32]}
        log1["prompt_text"] = log1["prompt_text"].replace(r" [\INST] ", "")[-32:]  # last 32 prompt chars

        stdout_whitelist = (
            "run",
            "prompt_len",
            "iters",
            "new_tok",
            "tree_h",
            "tree_w",
            "tree_size",
            "t0",
            "t1",
            "input_0",
            "input_1",
            "min_CLP",
            "gen_rate",
            "speed",
            "mem_use",
        )
        log_one_line(log1, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="summary", stdout_whitelist=stdout_whitelist)

    df = pd.DataFrame(test_logs)

    exp_summary = dict(
        max_n_beams=max_n_beams,
        max_beam_len=max_beam_len,
        min_log_prob=min_log_prob,
        max_budget=max_budget,
        max_branch_width=max_branch_width,
        gen_rate=round((df.new_tokens / df.iters).mean(), 2),  # macro-averaged generation rate
        gen_rate_micro=round(df.new_tokens.sum() / df.iters.sum(), 2),
        gen_speed=round(df.speed.mean(), 3),
        gen_speed_micro=round(df.new_tokens.sum() / (df.new_tokens / df.speed).sum(), 3),
        t0=round(df.t0.mean(), 4),
        t1=round(df.t1.mean(), 4),
        input_0=round(df.input_0.mean(), 1),
        input_1=round(df.input_1.mean(), 1),
        tree_size=round(df.tree_size.mean(), 1),
        tree_w=round(df.tree_w.mean(), 1),
        tree_h=round(df.tree_h.mean(), 1),
        prompt_len=round(df.prompt_len.mean(), 1),
        min_CLP=round(df.min_CLP.mean(), 2),
        mem_use=round(df.mem_use.max(), 2),
    )

    torch.cuda.empty_cache()
    return exp_summary, test_logs


def log_one_line(data_dict, verbose, save_dir=None, exp_name=None, msg_type=None, stdout_whitelist=None):
    """
    Logs key-value pairs from a dictionary to both the console (as a single line) and a JSONL file,
    with optional filtering for certain keys and conditional logging based on verbosity.

    Args:
        data_dict (dict): A dictionary containing the data to be logged.
        verbose (bool): If True, logs to stdout regardless of logger level.
        save_dir (str): Path to the directory where the log file will be saved.
        exp_name (str): Name of the experiment, used for the log file name.
        msg_type (str, optional): A message type to be included in the log file. Defaults to None.
    """
    stdout_blacklist = ["prompt_text", "text"]
    message_colors = {"exp": "GREEN", "summary": "WHITE", "config": "YELLOW_DARK", "zero": "blue", "info": "GREEN_DARK"}

    if verbose or (logger.level >= logging.INFO):
        if stdout_whitelist:
            log_line = "  ".join([f"{k}:{v}" for k, v in data_dict.items() if k in stdout_whitelist and v is not None])
        else:
            log_line = "  ".join([f"{k}:{v}" for k, v in data_dict.items() if k not in stdout_blacklist and v is not None])

        print(colored(log_line, message_colors.get(msg_type, "WHITE")))

    # logging to file
    if (msg_type is not None) and (save_dir is not None) and (exp_name is not None):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        log_filename = save_path / f"{exp_name}.jsonl"
        with log_filename.open("a") as file:
            file.write(json.dumps({"msg_type": msg_type, **data_dict}) + "\n")


def arg_to_list(args, arg):
    """
    Converts a command-line argument value to a list of appropriate types.
    Handles different value formats (single value, comma-separated values, "None"),
    converts to integers or floats as needed, and returns a list of parsed values.

    Args:
        args: An object containing command-line arguments (e.g., argparse.Namespace).
        arg (str): The name of the argument to retrieve and convert.

    Returns:
        list: A list of parsed values from the argument.
    """
    arg_value = getattr(args, arg)
    float_args = ["min_log_prob"]
    if arg_value is None:
        return [None]

    def from_str(s):
        """
        Parses a string value into an integer, float, or None.
        Args:  s (str): The string to parse.
        Returns: int, float, or None: The parsed value.
        """
        s = s.strip()
        if s.lower() == "none":
            return None
        elif arg in float_args:
            return float(s)
        else:
            return int(s)

    return [from_str(s) for s in arg_value.split(",")]

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h


class SelfAttention:
    def __init__(self, config, env, policy, layer_id, enable_prefetching, partial_weight_ratio=0.2, alpha=4, max_num_kv=400):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None
        self.enable_prefetching = enable_prefetching
        self.prefetch_idx = None
        self.prefetch_kv = None
        self.partial_index = None
        self.alpha = alpha
        self.max_num_kv = max_num_kv
        if self.layer_id > 1:
            self.partial_weight_ratio = partial_weight_ratio
        else:
            self.partial_weight_ratio = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        # WQ
        head_dim = h // self.config.n_head
        weights[0].data = weight_bias_concat(weights[0].data, weights[1].data, True, head_dim)
        weights[0].shape = (h, h+1)
        # WK
        weights[2].data = weight_bias_concat(weights[2].data, weights[3].data)
        weights[2].shape = (h, h+1)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)
        if self.layer_id > 1:
            self.prefetch_kv = device.allocate((2, self.max_num_kv, cache_home.val[0].shape[1], cache_home.val[0].shape[2]), np.float16, pin_memory=True)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")
    
    def prefetch_cache(self, cache_home, cache_read_buf, i, prefetch_idx, prefetch_cache_stream):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, prefetch_idx.shape[0]),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                self.prefetch_kv.data[0, :prefetch_idx.shape[0]], self.prefetch_kv.data[1, :prefetch_idx.shape[0]] = select_kv(prefetch_idx, k_home.data, v_home.data)
                k_c = TorchTensor((prefetch_idx.shape[0], k_home.shape[1], k_home.shape[2]), k_home.dtype, self.prefetch_kv.data[0, :prefetch_idx.shape[0]], k_home.device)
                v_c = TorchTensor((prefetch_idx.shape[0], v_home.shape[1], v_home.shape[2]), v_home.dtype, self.prefetch_kv.data[1, :prefetch_idx.shape[0]], v_home.device)

                with torch.cuda.stream(prefetch_cache_stream):
                    cache_read_buf.store((
                        k_c.smart_copy(dst, indices),
                        v_c.smart_copy(dst, indices),
                    ))
        elif path == 1 or path == 2:
            raise ValueError(f"Not implemented path: {path}")
        else:    
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, warmup, partial_weight_read_buf, partial_cache_read_buf, speculation_stream, prev_partial_cache_read_buf, prev_partial_weight_read_buf, weight_home):
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True
        head_dim = h.shape[-1] // n_head

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val
        if self.enable_prefetching and (i > 0):
            p_w_q = partial_weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache, w_q, w_k, self.partial_index = self.compute.mha(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config, warmup, self.partial_weight_ratio)
            cache_write_buf.store((new_k_cache, new_v_cache))
            if (prev_partial_cache_read_buf is not None) and (not warmup):
                prev_partial_cache_read_buf.store(set_partial_cache(new_k_cache.data, self.partial_index, n_head, head_dim))
                prev_partial_weight_read_buf.store(set_partial_weight(w_q.data, self.partial_index, n_head, head_dim))
            if warmup:
                weight_home.val[0] = w_q.smart_copy(weight_home.val[0].device)[0]
                weight_home.val[2] = w_k.smart_copy(weight_home.val[2].device)[0]
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            if self.enable_prefetching:
                partial_k_cache = partial_cache_read_buf.val
            if self.enable_prefetching:
                h, new_k_cache, new_v_cache, self.prefetch_idx = self.compute.mha_gen(h, mask, w_q,
                    b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                    k_cache, v_cache, donate, self.policy.attn_sparsity,
                    self.policy.compress_cache, self.policy.comp_cache_config, p_w_q, partial_k_cache, speculation_stream, self.alpha, self.max_num_kv)
            else:
                h, new_k_cache, new_v_cache, _ = self.compute.mha_gen(h, mask, w_q,
                    b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                    k_cache, v_cache, donate, self.policy.attn_sparsity,
                    self.policy.compress_cache, self.policy.comp_cache_config, None, None, None, None, None)
            cache_write_buf.store((new_k_cache, new_v_cache))
            if (prev_partial_cache_read_buf is not None) and (self.layer_id > 1):
                prev_partial_cache_read_buf.val = torch.cat((prev_partial_cache_read_buf.val, set_partial_cache(new_k_cache.data, self.partial_index, n_head, head_dim)))

        hidden.val = h


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight"),
            # bi
            ((4 * h,), dtype, path + "fc1.bias"),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight"),
            # bo
            ((h,), dtype, path + "fc2.bias"),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h


class TransformerLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)


class OptLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy,
                 partial_weight_ratio,
                 alpha,
                 max_num_kv
                 ):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        self.attn_layer = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                if (i == 0) or (i == (self.config.num_hidden_layers - 1)):
                    layers.append(SelfAttention(self.config, self.env, self.policy, i, False, partial_weight_ratio, alpha, max_num_kv))
                else:
                    layers.append(SelfAttention(self.config, self.env, self.policy, i, True, partial_weight_ratio, alpha, max_num_kv))
                self.attn_layer.append(len(layers) - 1)
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
                self.attn_layer.append(len(layers) - 1)
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()
        self.speculation_stream = torch.cuda.Stream()
        # CUDA streams [j][k]
        self.prefetch_cache_stream = torch.cuda.Stream()

        # Event (To start self attention after prefetching)
        self.prefetch_evt = torch.cuda.Event()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.partial_cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        self.partial_weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            # download_opt_weights(self.config.name, self.path)
            init_opt_weights(self.config.name, self.path)
        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                if j not in self.attn_layer[2:]:
                    self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            if j not in self.attn_layer[2:]:
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
    
    def prefetch_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        prefetch_idx = self.layers[j].prefetch_idx
        next_attn = self.attn_layer[self.attn_layer.index(j) + 1]

        # Load from cache_home to cache_read_buf
        self.layers[next_attn].prefetch_cache(self.cache_home[next_attn][k], self.cache_read_buf[next_attn][k], i, prefetch_idx, self.prefetch_cache_stream)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        if isinstance(self.layers[j], SelfAttention) or isinstance(self.layers[j], TransformerLayer):
            if k==0: # Generate skewing matrix for first input of warmup
                warmup_state = (not k) and self.warmup
                if j in self.attn_layer[2:]:
                    prev_attn = self.attn_layer[self.attn_layer.index(j) - 1]
                    self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                        self.weight_read_buf[j], self.attention_mask[k],
                        self.cache_write_buf[j][k], i, k, warmup_state, self.partial_weight_read_buf[j], self.partial_cache_read_buf[j][k], self.speculation_stream,
                        self.partial_cache_read_buf[prev_attn][k], self.partial_weight_read_buf[prev_attn], self.weight_home[j])
                else:
                    self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                        self.weight_read_buf[j], self.attention_mask[k],
                        self.cache_write_buf[j][k], i, k, warmup_state, self.partial_weight_read_buf[j], self.partial_cache_read_buf[j][k], self.speculation_stream,
                        None, None, self.weight_home[j])
        else:
            self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                self.weight_read_buf[j], self.attention_mask[k],
                self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):
        # assert torch.equal(cache_position, torch.arange(cache_position[0], cache_position[-1] + 1, device=cache_position.device)), "reconsider use of cache_position in amask slicing"
        # attention_mask = attention_mask[..., : cache_position.max() + 1]
        # assert attention_mask.shape[-2] == input_ids.shape[-1]

        # cache_position_models = ["llama"]

        # if self.model.config.model_type in cache_position_models:
        output = self.generate(
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_key_values=self.kv_cache,
            # cache_position=cache_position,
            # infinigener specific args
            inputs=input_ids,
            # max_new_tokens: int = 32,
            # do_sample: bool = False,
            # temperature: float = 1.0,
            # stop: Optional[int] = None,
            # debug_mode: Optional[str] = None,
            # cut_gen_len: Optional[int] = None,
            # verbose: int = 0,
            # warmup: bool = False
        )
        # else:
        #     assert torch.equal(cache_position, torch.arange(cache_position[0], cache_position[0] + cache_position.numel(), device=cache_position.device))

        #     output = self.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=self.kv_cache,
        #     )


        return output.logits

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0,
                 warmup: bool = False):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
        self.warmup = warmup

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
                self.partial_cache_read_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
            self.partial_weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    if (j in self.attn_layer[1:-1]) and (i > 0):
                        self.sync()
                    self.compute_layer(i, j, k)
                    self.sync()
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
                    if j in self.attn_layer[1:-1] and (i > 0):
                        self.prefetch_cache(i, j, k, overlap=True)
                        self.prefetch_evt.record()
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_inputs(prompt_len, num_prompts, tokenizer, path):
    prompts = []
    with open(path, 'r') as file:
        prompts.append(file.read())
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    input_ids[0] = input_ids[0][:prompt_len]
    # return (input_ids[0],) * num_prompts
    return [input_ids[0]] * num_prompts 

def main(args):

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model_1)

    logger.warning(f"Starting test with models {args.model_0}, {args.model_1}")
    spec_generator = create_spec_generator(
        model_name_0=args.model_0,
        model_name_1=args.model_1,
        draft_engine_class=args.draft_engine_class,
        target_opt_config=opt_config,
        target_env=env,
        target_policy=policy,
        gen_type=args.gen_type,
        offload=args.offload,
        device_size=args.device_size,
        check_tokenizer=False,
    )
    logger.debug(f"mem use {0}")

    if args.dataset.lower().startswith("oasst"):
        logger.warning("loading OASST-based prompts set")
        dataset = utils.get_dataset("oasst_prompts")
    elif args.dataset.lower().startswith("wiki"):
        logger.warning("loading Wikitext2-based prompts set")
        dataset = utils.get_dataset("wikitext_prompts")
    else:
        dataset_file_name = f"{args.dataset.lower()}_prompts"
        logger.warning(f"loading {dataset_file_name}")
        dataset = utils.get_dataset(dataset_file_name)

    if args.device_size != _DEFAULT_DEVICE_SIZE and not args.offload:
        logger.warning(f"Passed --device_size of {args.device_size}, but offloading is disabled")

    logs = []
    summaries = []

    config_dict = dict(
        gen_type=args.gen_type,
        model_0=args.model_0,
        model_1=args.model_1,
        temperature=args.temperature,
        max_n_beams=args.max_n_beams,
        max_beam_len=args.max_beam_len,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_budget=args.max_budget,
        max_branch_width=args.max_branch_width,
        branching=args.branching,
        min_log_prob=args.min_log_prob,
        replacement=args.replacement,
        n_tests=args.n_tests,
        seed=args.seed,
        dataset=args.dataset,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        date=datetime.datetime.today().strftime("%y%m%d"),
        hostname=socket.gethostname(),
        commit="none",
        offload=args.offload,
        device=torch.cuda.get_device_name(device).replace("NVIDIA ", ""),
    )
    if args.offload:
        config_dict["device_size"] = args.device_size
    log_one_line(config_dict, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="config")

    with torch.inference_mode():
        if args.zero:
            log_one_line({"mode": "zero"}, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="zero")
            spec_generator.tokenizer.pad_token_id = spec_generator.tokenizer.eos_token_id
            total_time = 0

            gene_config = transformers.GenerationConfig(
                max_new_tokens=32,
                do_sample=True,  # Use sampling
                temperature=0.6,  # Sampling temperature
                top_p=0.9,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=2,
            )

            for i in range(args.dataset_start_index, args.dataset_start_index + args.n_tests):
                try:
                    prompt = dataset[i]
                    inputs = spec_generator.tokenizer(prompt, return_tensors="pt").to(device)
                    with utils.Timing() as t:
                        spec_generator.target_engine.model.generate(**inputs, generation_config=gene_config)
                    log_one_line(
                        {"prompt": i, "elapsed": round(t.elapsed, 3)}, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="zero"
                    )
                    total_time += t.elapsed
                except RuntimeError:
                    print(colored(f"RuntineError in test {i}; skipping...", "RED"))
                    pass

            log_dict_zero = {"total_time": round(total_time, 3), "speed": round(args.max_new_tokens * args.n_tests / total_time, 3)}

            log_one_line(
                log_dict_zero,
                save_dir=args.save_dir,
                exp_name=args.exp_name,
                verbose=args.verbose,
                msg_type="zero",
            )
            print("-" * 120 + "\n   S U M M A R Y  (run without speculative decoding) \n" + "-" * 120)
            print(log_dict_zero)
            print("-" * 120)

            return None, None

    budget_classes = ["SpecFixed", "SpecExecBase"]  # classes driven by token budgets
    if spec_generator.__class__.__name__ not in budget_classes:
        args.max_budget = "0"
        args.max_branch_width = "0"

    # Convert string arguments to lists of integers
    sweep_args_present = []
    args_can_sweep = ["max_n_beams", "max_beam_len", "max_budget", "min_log_prob", "max_branch_width"]  # "max_branch_width" removed
    arg_lists = []
    for arg in args_can_sweep:
        arg_list = arg_to_list(args, arg)
        arg_lists.append(arg_list)
        if len(arg_list) > 1:
            sweep_args_present.append(arg)

    if len(sweep_args_present) > 2:
        logger.warning(f"More than two sweep arguments detected: {sweep_args_present}.")

    combinations = product(*arg_lists)
    combo_pbar = tqdm(combinations, desc=colored("hyperparameters sweep", "HIGHLIGHTED_GREEN"))
    for max_n_beams, max_beam_len, max_budget, min_log_prob, max_branch_width in combo_pbar:  # align with `args_can_sweep`
        print()
        exp_env = dict(
            gen_type=args.gen_type,
            model_0=args.model_0,
            model_1=args.model_1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            branching=args.branching,
            n_tests=args.n_tests,
            seed=args.seed,
            dataset=args.dataset,
            max_n_beams=max_n_beams,
            max_beam_len=max_beam_len,
            min_log_prob=min_log_prob,
            max_budget=max_budget,
            max_branch_width=max_branch_width,
        )
        log_one_line(exp_env, verbose=args.verbose, msg_type="info")

        with utils.Timing() as t:
            summary, test_logs = run_tests(
                spec_generator=spec_generator,
                dataset=dataset,
                args=args,
                max_n_beams=max_n_beams,
                max_beam_len=max_beam_len,
                max_budget=max_budget,
                max_branch_width=max_branch_width,
                min_log_prob=min_log_prob,
            )
        summary["exp_time"] = round(t.elapsed, 2)
        summaries.append(summary)
        logs.extend(test_logs)
        log_one_line(summary, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="exp")

        if args.wandb:
            wandb.init(project=args.wandb_project, name=f"{args.exp_name}__b{max_n_beams}x{max_beam_len}")
            wandb.log({**config_dict, **summary})
            wandb.finish()

        torch.cuda.empty_cache()

    # printing the summary table
    df = pd.DataFrame(summaries)
    sep = colored("-" * DISPLAY_WIDTH, "GREEN_DARK")
    print(sep + f"\n       A R G U M E N T S   {args.exp_name}\n" + sep)
    print(args)
    print(sep + f"\n       S U M M A R Y   R E S U L T S   {args.exp_name} \n" + sep)
    output_renames = {"max_branch_width": "branch", "max_n_beams": "beams", "max_beam_len": "max_h", "max_budget": "budget", "min_log_prob": "minLP"}
    print(df[[*args_can_sweep, "t0", "t1", "tree_h", "tree_size", "min_CLP", "exp_time", "gen_rate", "gen_speed", "mem_use"]].rename(columns=output_renames))
    print(sep)

    return summaries, logs


if __name__ == "__main__":

    if "logger" not in globals():
        logger = utils.get_logger()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='/home/ubuntu/clk/specexec/logs/debug.log',  # ??????
        filemode='a'           # ????
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoiding warnings

    # DEFAULT MODEL NAMES
    model_name_0 = "/mnt/data/clk/TinyLlama-1.1B-Chat-v1.0"
    model_name_1 = "/mnt/data/clk/opt-6.7b"

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="/mnt/data/clk/opt-6.7b",
    #     help="The model name.")
    parser.add_argument("--path", type=str, default="/mnt/data/clk/opt-6.7b/opt-weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="/mnt/data/clk/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)

    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--partial-weight-ratio", type=float, default=0.2)
    parser.add_argument("--max-num-kv", type=int, default=400)
    
    parser.add_argument("--warmup-input-path", type=str)
    parser.add_argument("--test-input-path", type=str)

    parser.add_argument("--exp_name", help="Experiment name", default="experiment")
    parser.add_argument("--save_dir", help="Experiments directory", default="logs")
    parser.add_argument("--model_0", help="Model 0 name", default=model_name_0)
    parser.add_argument("--model_1", help="Model 1 name", default=model_name_1)
    parser.add_argument("-d", "--dataset", help="Datastet for testing. oasst or wikitext only for now", default="oasst")
    parser.add_argument("--dataset_start_index", help="Dataset index to start from", default=0, type=int)
    parser.add_argument("-g", "--gen_type", help="SpecExecBase, SpecInfer or other class", default="SpecExecBase")
    parser.add_argument("--temperature", help="Sampling temperature", default=1.0, type=float)  # 0 for greedy
    parser.add_argument("--top_p", help="Sampling top_p", default=1.0, type=float)
    parser.add_argument("-t", "--temp", help="Sampling temperature and top_p as 4 digit string. '0609'-> 0.6, 0.9", default=None)
    parser.add_argument("--n_tests", help="Num of tests in each config", default=10, type=int)
    parser.add_argument("-b", "--max_n_beams", "--n_beams", help="Num of beams in each exp; CAN SWEEP", default="128")
    parser.add_argument("-m", "--max_beam_len", help="max beam len; CAN SWEEP", default="32")
    parser.add_argument("--branching", help="tree styles for fixed trees", default=None)
    parser.add_argument("--max_budget", help="speculation token budget for fixed trees; CAN SWEEP", default=None)
    parser.add_argument("--max_branch_width", help="max_branch_width for fixed trees and SX; CAN SWEEP", default="32")
    parser.add_argument(
        "--tree_max_len", help="max length of tree and engine cache, should fit prompt, generated and speculated tokens", default=4096, type=int
    )
    parser.add_argument("--replacement", help="draft model sampling with replacement", action="store_true")
    parser.add_argument("--repack", help="repack draft tree by combining identical node paths", action="store_true")
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--min_log_prob", help="min log proba threshold for added leafs; CAN SWEEP", default=None)
    # parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--loglevel", default="WARNING")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-o", "--offload", action="store_true")
    parser.add_argument("--device_size", type=int, default=_DEFAULT_DEVICE_SIZE)
    parser.add_argument("--wandb", help="Wandb enabled", action="store_true")
    parser.add_argument("--draft_temperature", default=None, type=float),
    parser.add_argument("--wandb_project", help="Wandb project name", default="spec_trees")
    parser.add_argument("--zero", help="zero speculation", action="store_true")
    parser.add_argument("--draft_engine_class", "--draft_engine", help="EngineStatic or other class", default="EngineRegular")

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))
    # logger.setLevel(getattr(logging, args.loglevel.upper(), logging.DEBUG))

    if args.wandb:
        import wandb

    if args.branching:
        # trying to converting string argument to int (except non-numerical strings)
        try:
            args.branching = int(args.branching)
        except ValueError:
            pass

    if args.temp is not None:
        # overriding args.temperature and args.top_p with decoded args.temp
        assert len(args.temp) == 4, f"args.temp should be a 4-digit string, received {args.temp}."
        args.temperature = float(f"{args.temp[0]}.{args.temp[1]}")
        args.top_p = float(f"{args.temp[2]}.{args.temp[3]}")

    with utils.Timing() as t:
        summaries, logs = main(args)
    logging.info(f"tests completed in {t.elapsed:.1f} s.")
