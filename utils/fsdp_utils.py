# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# DynaMOPD adaptation: selected FSDP2 routines from verl.utils.fsdp_utils.
# Use the CUDA bindings directly for the supported PyTorch 2.9 runtime.
from abc import ABC
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, fully_shard

fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard
get_device_id = torch.cuda.current_device


def get_torch_device():
    return torch.cuda


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = get_device_id()
    model.to(device)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap):
    """Select modules to wrap individually with fully_shard in FSDP2.

    Matches transformer layers by class name, and embed_tokens/lm_head by name
    (with isinstance fallback). Name-based matching is needed because peft wraps
    embed_tokens in ModulesToSaveWrapper, breaking isinstance(module, nn.Embedding).
    When tie_word_embeddings is True, embed_tokens and lm_head share weights and
    must not be wrapped separately.
    """
    _tie = getattr(model.config, "tie_word_embeddings", False)
    _wrap_by_name = set() if _tie else {"embed_tokens", "lm_head"}

    modules = []
    for name, module in model.named_modules():
        leaf_name = name.rsplit(".", 1)[-1] if "." in name else name
        if (
            module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap
            or (isinstance(module, nn.Embedding) and not _tie)
            or (leaf_name in _wrap_by_name and hasattr(module, "weight"))
        ):
            modules.append(module)
    return modules


def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    if isinstance(fsdp_transformer_layer_cls_to_wrap, set):
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)
    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap)

    for idx, module in enumerate(modules):
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"wrap module {module.__class__.__name__}")
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f"wrap module {model.__class__.__name__}")
    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module

    # Honor `forward_prefetch` config to match the FSDP1 path. Depth=1 mirrors
    # PyTorch FSDP1's hardcoded `forward_prefetch_limit=1`. Static-graph models
    # only (see PyTorch FSDP1's docstring on `forward_prefetch`).
    # `set_modules_to_forward_prefetch` was introduced in PyTorch 2.5; guard with
    # `hasattr` so users on PyTorch 2.4 silently fall back to the no-prefetch
    # behavior (same as before this PR — no regression).
    if config.get("forward_prefetch", False):
        fsdp_modules = [m for m in modules if isinstance(m, FSDPModule)]
        for i, m in enumerate(fsdp_modules):
            next_targets = fsdp_modules[i + 1 : i + 2]  # depth=1, mirrors FSDP1's forward_prefetch_limit=1
            if next_targets and hasattr(m, "set_modules_to_forward_prefetch"):
                m.set_modules_to_forward_prefetch(next_targets)
