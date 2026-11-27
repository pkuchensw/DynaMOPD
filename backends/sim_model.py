"""Standard-library cost models and paged KV accounting for virtual execution.

These models estimate time and capacity; they do not execute neural networks.
Timing coefficients are explicit configuration inputs, in seconds. In particular,
the optional quadratic prefill coefficient defaults to zero, so uncalibrated
profiles use only the configured linear prefill cost.
"""
from dataclasses import dataclass
import math


def _nonnegative(name, value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class UnifiedRoofline:
    """Step time max(F, W + G*N + A_p*P/b + A_d*C), or zero for no requests.

    N is the number of input tokens processed by the step, P the sum of squared
    uncached prefill lengths, C the sum of decode context lengths, and b the
    number of active requests. F is the launch floor, W fixed work, and G the
    per-token coefficient. No hardware-specific coefficients are implicit.
    """

    F: float
    W: float
    G: float
    A_p: float = 0.
    A_d: float = 0.

    def __post_init__(self):
        for name in ("F", "W", "G", "A_p", "A_d"):
            _nonnegative(name, getattr(self, name))

    def step_time(self, tokens, prefill_squared, decode_context, batch_size):
        for name, value in (("tokens", tokens), ("prefill_squared", prefill_squared),
                            ("decode_context", decode_context)):
            _nonnegative(name, value)
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        if not batch_size:
            if tokens or prefill_squared or decode_context:
                raise ValueError("an empty batch cannot contain token work")
            return 0.
        return max(self.F, self.W + self.G * tokens + self.A_p * prefill_squared / batch_size
                   + self.A_d * decode_context)

    def decode_coeffs(self, batch_size, context):
        """Return floor, first-step work, and growth per subsequent decode step.

        Context is summed over all requests. Each step processes one token per
        request and adds batch_size tokens to that sum. The batch stays fixed
        for this estimate; callers split at request completion boundaries.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("decode batch_size must be a positive integer")
        _nonnegative("context", context)
        return self.F, self.W + self.G * batch_size + self.A_d * context, self.A_d * batch_size


@dataclass(frozen=True)
class ScoringCost:
    """Fixed batch overhead plus the total uncached input tokens on one lane."""

    launch_s: float
    token_s: float

    def __post_init__(self):
        _nonnegative("launch_s", self.launch_s)
        _nonnegative("token_s", self.token_s)

    def predict(self, lengths):
        lengths = tuple(lengths)
        for length in lengths:
            _nonnegative("scoring input length", length)
        return self.launch_s + self.token_s * sum(lengths) if lengths else 0.


@dataclass(eq=False)
class Request:
    """Identity and committed KV prefix of one request on one inference lane."""

    request_id: int
    prompt_len: int
    target_len: int
    version: int
    prefilled: int = 0


class KvCacheManager:
    """Exclusive pages per request, without cross-request prefix caching.

    Reserve an entire extension atomically. A failed allocation leaves every
    counter unchanged. The executor updates Request.prefilled only after a
    successful reservation. Separate managers model independent TP lanes;
    teacher-prefix retention is accounted for separately by score_cache.py.
    """

    def __init__(self, num_blocks, block_size=16):
        if type(num_blocks) is not int or num_blocks < 0:
            raise ValueError("num_blocks must be a nonnegative integer")
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self._capacity = num_blocks
        self._used = 0
        self._blocks = {}

    @property
    def num_used_blocks(self):
        return self._used

    @property
    def num_free_blocks(self):
        return self._capacity - self._used

    def blocks_held(self, request):
        return self._blocks.get(request, 0)

    def allocate(self, request, additional_tokens):
        """Reserve through prefilled + additional_tokens; return new page count or None."""
        if any(type(n) is not int or n < 0 for n in (request.prefilled, additional_tokens)):
            raise ValueError("KV prefix and extension lengths must be nonnegative integers")
        tokens = request.prefilled + additional_tokens
        needed = (tokens + self.block_size - 1) // self.block_size
        additional = needed - self.blocks_held(request)
        if additional < 0:
            raise ValueError("KV reservations cannot shrink; free the request instead")
        if additional > self.num_free_blocks:
            return None
        self._blocks[request] = needed
        self._used += additional
        return additional

    def free(self, request):
        """Release all pages for a request; an already released request is a no-op."""
        self._used -= self._blocks.pop(request, 0)
