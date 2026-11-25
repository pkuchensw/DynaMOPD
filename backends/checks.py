"""Live SGLang checks for ragged input-logprob batches and resumable sampling."""
def check_server(server):
    rows = [[758,11,220,17,10,18,25,12], [785,25,17,17,16]]
    common = dict(sampling_params={"max_new_tokens":0}, return_logprob=True)
    reference = server.post("generate",dict(input_ids=rows, logprob_start_len=[0,0], **common))
    starts = [4,2]
    server.post("flush_cache?timeout=10", {})
    server.post("generate",dict(input_ids=[r[:s+1] for r,s in zip(rows,starts)], **common))
    warmed = server.post("generate",dict(input_ids=rows, logprob_start_len=[s-1 for s in starts], **common))
    cache_hits = [o["meta_info"]["cached_tokens"] for o in warmed]
    server.post("flush_cache?timeout=10", {})
    rebuilt = server.post("generate",dict(input_ids=rows, logprob_start_len=[s-1 for s in starts], **common))
    errors = []
    for batch in (warmed, rebuilt):
        for i,(r,start) in enumerate(zip(rows,starts)):
            full = reference[i]["meta_info"]["input_token_logprobs"]
            chunk = batch[i]["meta_info"]["input_token_logprobs"]
            if [q[1] for q in chunk]!=r[start-1:] or chunk[0][0] is not None:
                raise RuntimeError("chunk predecessor mismatch")
            errors.extend(abs(q[0]-f[0]) for q,f in zip(chunk[1:],full[start:],strict=True))
    if not all(cache_hits) or max(errors)>.25:
        raise RuntimeError(f"cache/first-token logprob check failed: {cache_hits}, {errors}")
    result = dict(max_logprob_error=max(errors), tested_positions=len(errors), cached_tokens=cache_hits)
    if server.name=="rollout":
        sampling = dict(temperature=1., top_p=1., top_k=-1, sampling_seed=731, ignore_eos=True)
        def generate(ids,n):
            return server.post("generate",dict(input_ids=ids,sampling_params=dict(sampling,max_new_tokens=n)))["output_ids"]
        full = generate(rows[0],8)
        prefix = generate(rows[0],3)
        server.post("flush_cache?timeout=10", {})
        continuation = generate(rows[0]+prefix,5)
        if full!=prefix+continuation:
            raise RuntimeError("seed/absolute-position continuation disagrees with uninterrupted generation")
        result["continuation_matches"] = True
    return result
