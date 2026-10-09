# SPDX-License-Identifier: Apache-2.0
"""Shape profiles for Step5's gathered-head BF16 DCP2 Attention.

These profiles affect only short queries. They do not read device metadata,
change selected regions, or require host synchronization during Graph replay.
"""


def select_short_query_splits(tokens: int, max_query_len: int, window: int) -> int:
    if not 1 <= max_query_len <= 4:
        return 2
    if window == 0 and max_query_len == 1:
        return {8: 8, 16: 4}.get(tokens, 2)
    if window == 512 and tokens == 64:
        return 1
    return 2
