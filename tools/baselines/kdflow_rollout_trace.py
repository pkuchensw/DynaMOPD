"""Per-server timings for the non-streaming requests used in the KDFlow benchmark."""
import os
from uuid import uuid4


def launch_server(server_args):
    from sglang.srt.entrypoints import http_server

    if os.environ.get("MOPD_TRACE_DIR"):
        from profile_trace import emit

        @http_server.app.middleware("http")
        async def trace_generate(request, call_next):
            if request.url.path != "/generate":
                return await call_next(request)
            event = dict(id=uuid4().hex, phase="rollout_worker_request",
                         method="SGLang.generate", host=server_args.host,
                         base_gpu_id=server_args.base_gpu_id, tp_size=server_args.tp_size)
            emit(dict(event, event="start"))
            ok = False
            try:
                response = await call_next(request)
                ok = response.status_code < 400
                return response
            finally:
                emit(dict(event, event="end", ok=ok))

    return http_server.launch_server(server_args)
