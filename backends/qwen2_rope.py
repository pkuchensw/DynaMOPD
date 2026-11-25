"""SGLang custom weight loader for unscaled Qwen2 rotary embeddings."""
import math

LOADER = "DynaMOPD.backends.qwen2_rope.restore"


def restore(model, named_tensors):
    import torch
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.models.qwen2 import Qwen2ForCausalLM

    if not isinstance(model, Qwen2ForCausalLM) or len(named_tensors) != 1 or named_tensors[0][0] != "rope_theta":
        raise ValueError("this loader only supports unscaled Qwen2")
    theta = float(named_tensors[0][1].item())
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("invalid rope_theta")
    attention = [layer.self_attn for layer in model.model.layers]
    embeddings = {attn.rotary_emb for attn in attention}
    if not embeddings or any(type(rope) is not RotaryEmbedding for rope in embeddings):
        raise ValueError("unexpected rotary embedding implementation")
    with torch.inference_mode():
        for rope in embeddings:
            cache = rope.cos_sin_cache
            rope.base = theta
            with torch.device(cache.device):
                updated = rope._compute_cos_sin_cache()
            if updated.shape != cache.shape:
                raise ValueError("rotary cache shape changed during in-place update")
            cache.copy_(updated)
        for attn in attention:
            attn.rope_theta = theta
    # CUDA graphs captured the original buffers; copy_ preserves their addresses.
    torch.cuda.synchronize()
