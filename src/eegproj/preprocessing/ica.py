"""ICA convergence monitoring."""
import logging
import re


class ICAProgress(logging.Handler):
    """Read true iteration progress (Infomax n_iter can equal max on convergence)."""

    def __init__(self):
        super().__init__()
        self.step = 0
        self.change = float('inf')

    def emit(self, record):
        match = re.search(r'step (\d+) - .*wchange\s+([\d.eE+-]+)', record.getMessage())
        if match:
            self.step, self.change = int(match[1]), float(match[2])
