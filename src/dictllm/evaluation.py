"""Perplexity evaluation used by stage checkpoints."""

from __future__ import annotations

import torch
from tqdm import tqdm


def compute_perplexity(
    model,
    tokenized_data,
    context_length: int,
    stride: int,
    device: str,
    description: str = "WikiText-2 PPL",
) -> float:
    model.to(device)
    model.eval()
    sequence_length = tokenized_data.input_ids.shape[1]
    losses = []
    previous_end = 0
    for begin in tqdm(range(0, sequence_length, stride), desc=description):
        end = min(begin + context_length, sequence_length)
        target_length = end - previous_end
        input_ids = tokenized_data.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-target_length] = -100
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                labels=target_ids,
                use_cache=False,
            )
        losses.append(output.loss.detach().float().cpu())
        previous_end = end
        if end == sequence_length:
            break
    return torch.exp(torch.stack(losses).mean()).item()

