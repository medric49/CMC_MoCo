import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, use_tb):
        self._log_dir = log_dir
        if use_tb:
            self._sw = SummaryWriter(log_dir / 'tb')
        else:
            self._sw = None

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)



