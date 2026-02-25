import phlower_tensor as pt
import torch


def threashold_sparse_tensor(
    sparse_tensor: pt.PhlowerTensor, threshold: float = 1e-8
) -> pt.PhlowerTensor:
    """
    Threshold a sparse tensor.

    Parameters
    ----------
    sparse_tensor: pt.PhlowerTensor
        Sparse tensor to threshold.
    threshold: float
        Threshold value.

    Returns
    -------
    pt.PhlowerTensor
        Thresholded sparse tensor.
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("input tensor is not sparse.")

    idx = sparse_tensor.indices()
    val = sparse_tensor.values()
    shape = sparse_tensor.shape
    dtype = sparse_tensor.dtype
    device = sparse_tensor.device

    mask = val.abs() > threshold
    tensor = torch.sparse_coo_tensor(
        idx[:, mask], torch.ones(mask.sum()), shape, dtype=dtype, device=device
    )
    return pt.phlower_tensor(
        tensor.coalesce(), dimension=sparse_tensor.dimension
    ).to(device=device)
