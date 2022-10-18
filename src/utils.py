import os
from pathlib import Path
import math


def project_root() -> Path:
    return Path(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')
        )
    )


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not math.isinf(val) or math.isnan(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
