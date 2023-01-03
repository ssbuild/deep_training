# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 9:51
from typing import Union, Tuple, Optional, Any

import torch


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return type(self), (self.message,)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


class ExperimentalFeatureWarning(RuntimeWarning):
    """
    A warning that you are using an experimental feature
    that may change or be deleted.
    """

    pass

def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    # Parameters

    tensor : `torch.FloatTensor`, required.
        A tensor of arbitrary size.
    dim : `int`, optional (default = `-1`)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: `bool`, optional (default = `False`)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.

    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```

    # Parameters

    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    # Returns

    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices

def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)

def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/main/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.

    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    # Returns

    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def masked_index_fill(
    target: torch.Tensor, indices: torch.LongTensor, mask: torch.BoolTensor, fill_value: int = 1
) -> torch.Tensor:
    """
    The given `indices` in `target` will be will be filled with `fill_value` given a `mask`.


    # Parameters

    target : `torch.Tensor`, required.
        A 2 dimensional tensor of shape (batch_size, sequence_length).
        This is the tensor to be filled.
    indices : `torch.LongTensor`, required
        A 2 dimensional tensor of shape (batch_size, num_indices),
        These are the indices that will be filled in the original tensor.
    mask : `torch.Tensor`, required.
        A 2 dimensional tensor of shape (batch_size, num_indices), mask.sum() == `nonzero_indices`.
    fill_value : `int`, optional (default = `1`)
        The value we fill the tensor with.

    # Returns

    filled_target : `torch.Tensor`
        A tensor with shape (batch_size, sequence_length) where 'indices' are filled with `fill_value`
    """
    mask = mask.bool()
    prev_shape = target.size()
    # Shape: (batch_size * num_indices)
    flattened_indices = flatten_and_batch_shift_indices(indices * mask, target.size(1))
    # Shape: (batch_size * num_indices, 1)
    mask = mask.view(-1)
    # Shape: (batch_size * sequence_length, 1)
    flattened_target = target.view(-1, 1)
    # Shape: (nonzero_indices, 1)
    unmasked_indices = flattened_indices[mask].unsqueeze(-1)

    flattened_target = flattened_target.scatter(0, unmasked_indices, fill_value)

    filled_target = flattened_target.reshape(prev_shape)

    return filled_target


def masked_index_replace(
    target: torch.Tensor,
    indices: torch.LongTensor,
    mask: torch.BoolTensor,
    replace: torch.Tensor,
) -> torch.Tensor:
    """
    The given `indices` in `target` will be will be replaced with corresponding index
    from the `replace` tensor given a `mask`.


    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_dim).
        This is the tensor to be replaced into.
    indices : `torch.LongTensor`, required
        A 2 dimensional tensor of shape (batch_size, num_indices),
        These are the indices that will be replaced in the original tensor.
    mask : `torch.Tensor`, required.
        A 2 dimensional tensor of shape (batch_size, num_indices), mask.sum() == `nonzero_indices`.
    replace : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, num_indices, embedding_dim),
        The tensor to perform scatter from.

    # Returns

    replaced_target : `torch.Tensor`
        A tensor with shape (batch_size, sequence_length, embedding_dim) where 'indices'
        are replaced with the corrosponding vector from `replace`
    """
    target = target.clone()
    mask = mask.bool()
    prev_shape = target.size()
    # Shape: (batch_size * num_indices)
    flattened_indices = flatten_and_batch_shift_indices(indices * mask, target.size(1))
    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))
    # Shape: (nonzero_indices, 1)
    mask = mask.view(-1)
    flattened_target[flattened_indices[mask]] = replace.view(-1, replace.size(-1))[mask]
    # Shape: (batch_size, sequence_length, embedding_dim)
    replaced_target = flattened_target.reshape(prev_shape)
    return replaced_target


def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns segmented spans in the target with respect to the provided span indices.

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.

    # Returns

    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def flattened_index_select(target: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """
    The given `indices` of size `(set_size, subset_size)` specifies subsets of the `target`
    that each of the set_size rows should select. The `target` has size
    `(batch_size, sequence_length, embedding_size)`, and the resulting selected tensor has size
    `(batch_size, set_size, subset_size, embedding_size)`.

    # Parameters

    target : `torch.Tensor`, required.
        A Tensor of shape (batch_size, sequence_length, embedding_size).
    indices : `torch.LongTensor`, required.
        A LongTensor of shape (set_size, subset_size). All indices must be < sequence_length
        as this tensor is an index into the sequence_length dimension of the target.

    # Returns

    selected : `torch.Tensor`, required.
        A Tensor of shape (batch_size, set_size, subset_size, embedding_size).
    """
    if indices.dim() != 2:
        raise ConfigurationError(
            "Indices passed to flattened_index_select had shape {} but "
            "only 2 dimensional inputs are supported.".format(indices.size())
        )
    # Shape: (batch_size, set_size * subset_size, embedding_size)
    flattened_selected = target.index_select(1, indices.view(-1))

    # Shape: (batch_size, set_size, subset_size, embedding_size)
    selected = flattened_selected.view(target.size(0), indices.size(0), indices.size(1), -1)
    return selected