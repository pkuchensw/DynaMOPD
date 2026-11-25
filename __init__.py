"""Synchronous prefix-streamed multi-teacher distillation."""
from pathlib import Path

# Keep previously published "python -m DynaMOPD.<tool>" commands usable.
# Training code imports the scheduler and backends packages directly.
__path__.append(str(Path(__file__).with_name("tools")))
