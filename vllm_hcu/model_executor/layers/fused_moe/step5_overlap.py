# SPDX-License-Identifier: Apache-2.0
"""Host-only phase guard for Step5 shared-expert overlap."""

def _metadata_is_short_uniform(metadata):
    starts = getattr(metadata, "query_start_loc_cpu", None)
    if starts is None or starts.device.type != "cpu":
        return False
    tokens = getattr(metadata, "num_actual_tokens", 0)
    if tokens <= 0 or getattr(metadata, "max_query_len", 0) > 4:
        return False
    requests = (getattr(metadata, "num_decodes", 0)
                + getattr(metadata, "num_prefills", 0))
    if requests:
        starts = starts[:requests + 1]
    values = starts.tolist()
    if len(values) < 2 or values[0] != 0 or values[-1] != tokens:
        return False
    lengths = [end - start for start, end in zip(values, values[1:])]
    if any(n < 0 or n > 4 for n in lengths):
        return False
    live = {n for n in lengths if n}
    return len(live) == 1

def is_short_uniform_decode(context):
    """Check host scheduler data, including FULL capture's missing descriptor."""
    if context is None:
        return False
    cached = getattr(context, "_step5_short_uniform_decode", None)
    if cached is not None:
        return cached
    desc = getattr(context, "batch_descriptor", None)
    ready = False
    if desc is not None:
        if desc.uniform:
            tokens, requests = desc.num_tokens, desc.num_reqs
            ready = bool(requests is not None and requests > 0 and tokens > 0
                         and tokens % requests == 0 and 1 <= tokens // requests <= 4)
    else:
        # Native MRv2 FULL capture uses runtime NONE and no BatchDescriptor.
        # Its CPU query starts are already built before entering the model.
        metadata = getattr(context, "attn_metadata", None)
        if isinstance(metadata, dict):
            seen = set()
            for value in metadata.values():
                if id(value) in seen:
                    continue
                seen.add(id(value))
                starts = getattr(value, "query_start_loc_cpu", None)
                if starts is None or starts.device.type != "cpu":
                    continue
                if not _metadata_is_short_uniform(value):
                    ready = False
                    break
                ready = True
    context._step5_short_uniform_decode = ready
    return ready
