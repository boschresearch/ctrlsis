import torch
import torch.utils.tensorboard


class SummaryWriter:
    def __init__(self, log_dir):
        if len(log_dir) > 0:
            self._writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        else:
            self._writer = None

    def add_scalar(self, name, value, step):
        if self._writer is not None:
            self._writer.add_scalar(name, value, step)
