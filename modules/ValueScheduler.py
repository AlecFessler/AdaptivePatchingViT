# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch

class ValueScheduler:
    def __init__(self, start, end, steps, cosine=False):
        self.start = torch.tensor(start, dtype=torch.float32)
        self.end = torch.tensor(end, dtype=torch.float32)
        self.steps = steps
        self.cosine = cosine
        self.current_epoch = 0
        self.current_value = self.start.clone()

    def step(self):
        if self.current_epoch >= self.steps - 1:
            self.current_value = self.end
        else:
            progress = torch.tensor(self.current_epoch, dtype=torch.float32) / (self.steps - 1)
            if self.cosine:
                progress = 0.5 * (1 + torch.cos(torch.pi * progress))
            else:
                progress = 1 - progress

            self.current_value = self.end + (self.start - self.end) * progress

        self.current_epoch += 1

    def reset(self):
        self.current_epoch = 0
        self.current_value = self.start.clone()
