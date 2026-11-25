# Code from verl

The [FlashAttention log probability function](../utils/torch_functional.py) copies `logprobs_from_logits_flash_attn` from `verl/utils/torch_functional.py` at the same commit. The imports are limited to the CUDA implementation used by DynaMOPD.

The [FSDP2 utilities](../utils/fsdp_utils.py) include code from `verl/utils/fsdp_utils.py` at upstream commit `c791da0bfcd7d7b560b1e461d2c188145b39c353`. The original file has SHA-256 `90d99cc9d28777f5593453c6cbda7c8c30e006e203962d9d483430e11ba6c6b3`. [Upstream source](https://github.com/verl-project/verl/blob/c791da0bfcd7d7b560b1e461d2c188145b39c353/verl/utils/fsdp_utils.py)

The copied functions are `apply_fsdp2`, `maybe_patch_fsdp_module`, `_select_fsdp2_wrap_targets`, `offload_fsdp2_model_to_cpu`, `load_fsdp2_model_to_gpu` and `load_fsdp_optimizer`. Function bodies and decorators are preserved. DynaMOPD supplies the imports for its PyTorch 2.9 CUDA runtime and binds device lookup to `torch.cuda`. The original license and notice accompany the code. [License](verl-LICENSE.txt) · [Notice](verl-NOTICE.txt)

The sampled K1 adapter in `algorithms/verl_k1.py` follows the vanilla clipped policy loss and K1 estimator in `verl/trainer/ppo/core_algos.py` at the same revision. The parity tests call the native functions directly.
