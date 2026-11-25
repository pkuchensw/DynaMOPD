"""SGLang entrypoint for deterministic prefill with standard decode CUDA graphs."""
import os
import sys
from functools import wraps


def install_decode_capture():
    from sglang.srt import batch_invariant_ops
    from sglang.srt.model_executor.model_runner import ModelRunner

    capture = ModelRunner.init_device_graphs
    if getattr(capture, "_dynamopd_decode_capture", False):
        return

    @wraps(capture)
    def capture_decode(self):
        if (self.device != "cuda" or self.server_args.disable_cuda_graph
                or not self.is_generation or not batch_invariant_ops.is_batch_invariant_mode_enabled()):
            return capture(self)
        # Only the decode graph captures these kernels. Restore eager prefill
        # before serving so a cached teacher chunk uses invariant GEMMs too.
        batch_invariant_ops.disable_batch_invariant_mode()
        try:
            return capture(self)
        finally:
            batch_invariant_ops.enable_batch_invariant_mode()

    capture_decode._dynamopd_decode_capture = True
    ModelRunner.init_device_graphs = capture_decode


if __name__ in ("__main__", "__mp_main__"):
    install_decode_capture()

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
