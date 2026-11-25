from ..components import resolve


def executor_class(config, simulate=False):
    if simulate:
        from .simulation import Simulator
        return Simulator
    name = config.get("engine", "sglang_fsdp2")
    transport = config.get("cluster", {}).get("transport", "local")
    if name == "sglang_fsdp2":
        return resolve(transport, {"local": "DynaMOPD.backends.native:NativeExecutor",
                                   "ray": "DynaMOPD.cluster.ray:RayExecutor"})
    if name == "hf_reference" and transport != "local":
        raise ValueError("hf_reference requires local placement")
    return resolve(name, {"hf_reference": "DynaMOPD.backends.hf_reference.runtime:GPUExecutor"})
