class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._lr(self.step_num)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _lr(self, s):
        return self.factor * (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup_steps ** -1.5)

    def get_lr(self):
        if self.step_num == 0:
            return 0.0
        return self._lr(self.step_num)
