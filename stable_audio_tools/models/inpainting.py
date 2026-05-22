import random
import torch
from enum import Enum
from typing import List, Optional, Tuple, Dict, Union

class MaskType(Enum):
    RANDOM_SEGMENTS = 0
    FULL_MASK = 1
    CAUSAL_MASK = 2

def random_inpaint_mask(
    sequence: torch.Tensor,
    padding_masks: torch.Tensor,
    max_mask_segments: int = 10,
    mask_type_probabilities: Optional[List[float]] = None,
    fixed_mask_size: Optional[int] = None,
    outpainting_dropout_probs: Optional[Dict[str, float]] = None,
    use_context_shift: bool = False,
    return_dropout_lengths: bool = False,
    **kwargs,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Generates random inpainting masks for a batch of latent audio sequences.
    The output inpainting mask has 0 where data should be inpainted, and 1 where data is provided.

    Args:
        sequence: The input sequence tensor of shape (b, c, sequence_length).
        padding_masks: A tensor of shape (b, sequence_length)
                       where 1 indicates real data latents and 0 indicates latents encoding silence padding.
        max_mask_segments: The maximum number of segments for the RANDOM_SEGMENTS mask type.
        mask_type_probabilities: A list of probabilities for choosing each mask type.
                                 The order should correspond to:
                                 [P(RANDOM_SEGMENTS), P(FULL_MASK), P(CAUSAL_MASK)].
                                 If None, defaults to [0.1, 0.8, 0.1].
        use_context_shift: If True and outpainting_dropout_probs is set, shift the dropped audio
                           exactly like _apply_context_dropout: prepend zeros at the front and
                           shift the entire sequence right, truncating from the end. This means the
                           target window contains earlier audio (targets change). When
                           return_dropout_lengths=True, the pre-mask shifted sequence is returned so
                           the caller can update the diffusion input and recompute noising/targets
                           against the shifted audio.
        return_dropout_lengths: If True, also return a (b,) LongTensor of per-item dropout lengths
                                 (number of tokens prepended / shifted out). Useful for computing
                                 per-item RoPE position offsets downstream.
                                 When use_context_shift=True, also returns the full shifted sequence
                                 (before masking) as a 4th element so the caller can update
                                 diffusion_input for correct target computation.

    Returns:
        Depending on flags:
          return_dropout_lengths=False (default):
            (masked_sequence, inpaint_mask)
          return_dropout_lengths=True, use_context_shift=False:
            (masked_sequence, inpaint_mask, dropout_lengths)
          return_dropout_lengths=True, use_context_shift=True:
            (masked_sequence, inpaint_mask, dropout_lengths, shifted_sequence)
            where shifted_sequence is the full per-item shifted tensor (before masking) that
            the caller should use as the new diffusion_input for noising and target computation.
    """
    b, c, sequence_length = sequence.size()

    num_mask_types = len(MaskType)
    if mask_type_probabilities is None:
        mask_type_probabilities = [0.1, 0.8, 0.1]
    else:
        if len(mask_type_probabilities) != num_mask_types:
            raise ValueError(
                f"mask_type_probabilities must have {num_mask_types} elements, "
                f"one for each MaskType. Got {len(mask_type_probabilities)}."
            )
        if not torch.isclose(torch.tensor(float(sum(mask_type_probabilities))), torch.tensor(1.0)):
            raise ValueError(
                f"mask_type_probabilities must sum to 1.0. "
                f"Current sum: {sum(mask_type_probabilities)}"
            )

    output_masks_list = []

    # Per-item tracking for shift/zero path and dropout_lengths return
    drop_lengths_per_item: List[int] = []

    mask_types_to_sample = [mt.value for mt in MaskType]

    for i in range(b):
        padding_mask_single_item = padding_masks[i]
        real_sequence_length = (padding_mask_single_item == 1).sum().item()

        item_mask = torch.ones((1, 1, sequence_length), device=sequence.device, dtype=torch.float32)
        drop_length_i = 0

        chosen_mask_value = random.choices(mask_types_to_sample, weights=mask_type_probabilities, k=1)[0]
        current_mask_type = MaskType(chosen_mask_value)

        if current_mask_type == MaskType.FULL_MASK:
            item_mask = torch.zeros((1, 1, sequence_length), device=sequence.device, dtype=torch.float32)
        elif real_sequence_length == 0:
            pass  # item_mask remains all ones for RANDOM_SEGMENTS/CAUSAL_MASK on empty real data
        else:
            if current_mask_type == MaskType.RANDOM_SEGMENTS:
                num_segments = random.randint(1, max_mask_segments)
                max_len_per_segment_calc = max(1, real_sequence_length // num_segments)

                for _ in range(num_segments):
                    segment_length = random.randint(1, max_len_per_segment_calc)

                    if real_sequence_length - segment_length < 0:
                        continue
                    mask_start = random.randint(0, real_sequence_length - segment_length)
                    item_mask[:, :, mask_start : mask_start + segment_length] = 0

            elif current_mask_type == MaskType.CAUSAL_MASK:
                # Keep a prefix of real data, inpaint the suffix.
                if fixed_mask_size is None:
                    unmasked_prefix_len = random.randint(0, real_sequence_length)
                else:
                    unmasked_prefix_len = fixed_mask_size

                if unmasked_prefix_len < real_sequence_length:
                    item_mask[:, :, unmasked_prefix_len:real_sequence_length] = 0

                if outpainting_dropout_probs is not None:
                    p_uncond = outpainting_dropout_probs.get('p_uncond', 0.0)
                    p_partial = outpainting_dropout_probs.get('p_partial', 0.0)
                    rand_val = random.random()
                    if rand_val < p_uncond:
                        # Drop entire unmasked prefix
                        drop_length_i = unmasked_prefix_len
                    elif rand_val < p_uncond + p_partial and unmasked_prefix_len > 0:
                        # Drop some start chunks of the unmasked prefix
                        chunk_size = real_sequence_length - unmasked_prefix_len
                        if chunk_size > 0:
                            num_chunks = unmasked_prefix_len // chunk_size
                            chunks_to_drop = random.randint(1, num_chunks)
                            drop_length_i = chunks_to_drop * chunk_size

        drop_lengths_per_item.append(drop_length_i)
        output_masks_list.append(item_mask)

    final_inpaint_mask = torch.cat(output_masks_list, dim=0).to(sequence.device)

    if outpainting_dropout_probs is not None:
        # Per-item zero-context dropout: shift or in-place zero.
        sequence = sequence.clone()
        for i in range(b):
            dl = drop_lengths_per_item[i]
            if dl <= 0:
                continue

            if use_context_shift:
                # Exactly like _apply_context_dropout with use_context_shift=True:
                # prepend zeros at front, shift real audio right, truncate from end.
                # [zeros(0:dl), sequence(0:tl-dl)]
                real_portion = sequence[i, :, :sequence_length - dl]  # (c, tl - dl)
                filler = torch.zeros(c, dl, device=sequence.device, dtype=sequence.dtype)
                sequence[i] = torch.cat([filler, real_portion], dim=-1)
            else:
                # In-place: zero the dropped prefix positions (no audio shift).
                sequence[i, :, :dl] = 0

    # `sequence` is now the (possibly shifted) full sequence.
    # masked_sequence zeros out the to-be-inpainted regions.
    masked_sequence = sequence * final_inpaint_mask

    if return_dropout_lengths:
        dropout_lengths_tensor = torch.tensor(drop_lengths_per_item, dtype=torch.long)
        if use_context_shift:
            # Return the full shifted sequence so the caller can update diffusion_input
            # and recompute noising/targets against the shifted (earlier) audio.
            return masked_sequence, final_inpaint_mask, dropout_lengths_tensor, sequence
        return masked_sequence, final_inpaint_mask, dropout_lengths_tensor

    return masked_sequence, final_inpaint_mask


def apply_accomp_cutoff(
    accomp: torch.Tensor,
    cutoff: int,
    p_uncond: float = 0.0,
    p_partial: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a per-item cutoff to an accompaniment latent sequence for live-jamming conditioning.

    Args:
        accomp: (b, c, seq_len) accompaniment latents, time-aligned with target.
        cutoff: fixed end cutoff — frames at index >= cutoff are always zeroed
                (simulates the live-latency horizon). Must be <= seq_len.
        p_uncond: probability of dropping the entire accompaniment (start_cut := cutoff).
        p_partial: probability of randomly choosing a start_cut in [1, cutoff) — frames
                   at index < start_cut are zeroed (simulates loss of older context).

    Returns:
        accomp_masked: accomp with positions outside [start_cut, cutoff) zeroed.
        accomp_mask: (b, 1, seq_len) — 1 over [start_cut, cutoff), 0 elsewhere.
    """
    b, c, seq_len = accomp.shape
    assert cutoff <= seq_len, f"cutoff ({cutoff}) must be <= seq_len ({seq_len})"

    accomp_masked = accomp.clone()
    accomp_mask = torch.zeros(b, 1, seq_len, device=accomp.device, dtype=accomp.dtype)

    for i in range(b):
        rand_val = random.random()
        if rand_val < p_uncond:
            start_cut = cutoff
        elif rand_val < p_uncond + p_partial and cutoff > 1:
            start_cut = random.randint(1, cutoff - 1)
        else:
            start_cut = 0

        if start_cut > 0:
            accomp_masked[i, :, :start_cut] = 0
        if cutoff < seq_len:
            accomp_masked[i, :, cutoff:] = 0
        accomp_mask[i, :, :cutoff] = 1

    return accomp_masked, accomp_mask
