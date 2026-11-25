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
"""FlashAttention log probabilities from verl.utils.torch_functional."""
import torch
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss


def logprobs_from_logits_flash_attn(
    logits: torch.Tensor, labels: torch.Tensor, inplace_backward: bool = True
) -> torch.Tensor:
    """Compute log-probabilities using Flash Attention's optimized cross-entropy.

    Uses the Flash Attention library's Triton-based cross-entropy implementation
    for efficient computation on NVIDIA GPUs.

    Args:
        logits: Model output logits of shape (batch_size, vocab_size).
        labels: Target token indices of shape (batch_size,).
        inplace_backward: If True, perform backward pass in-place for memory efficiency.

    Returns:
        torch.Tensor: Log-probabilities for target labels, shape (batch_size,).

    Raises:
        AssertionError: If flash-attn version < 2.4.3 (different return format).
    """
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]
