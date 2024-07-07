
import torch


class GeometryProcessorMixin:
    """A mix-in class for geometry processing."""

    def compute_area(self) -> torch.Tensor:
        # Use self.points for point coordinates
        raise NotImplementedError

    def compute_volume(self) -> torch.Tensor:
        raise NotImplementedError
