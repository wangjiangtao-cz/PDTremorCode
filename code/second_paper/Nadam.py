import torch
import torch.optim as optim

class Nadam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, schedule_decay=0.004):
        defaults = dict(lr=lr, betas=betas, eps=eps, schedule_decay=schedule_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                eps = group['eps']
                schedule_decay = group['schedule_decay']

                state['step'] += 1
                t = state['step']

                # Adam update
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                p.data -= group['lr'] * (m_hat / (torch.sqrt(v_hat) + eps) + schedule_decay * p.data)

                state['m'], state['v'] = m, v

        return loss
