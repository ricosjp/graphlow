import phlower_tensor as pt


def phlower_repeat(a: pt.PhlowerTensor, *sizes: int) -> pt.PhlowerTensor:
    return pt.phlower_tensor(
        a.to_tensor().repeat(*sizes),
        dimension=a.dimension,
    ).to(device=a.device)
