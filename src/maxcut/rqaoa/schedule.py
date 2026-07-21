from typing import Union


def select_depth(layer_count: Union[int, dict], n_current: int) -> int:
    current_p = layer_count
    if isinstance(layer_count, dict):
        sorted_keys = sorted(layer_count.keys(), reverse=True)
        for k in sorted_keys:
            if n_current <= k:
                current_p = layer_count[k]
    return current_p
