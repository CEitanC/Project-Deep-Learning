### equal layer contribution
### DL depends on gradient for each step
### non-linear loss


import math
import torch
from torch import Tensor
from typing import List, Optional
import numpy as np



def new_alg_comp(params: List[Tensor],
        d_p_list: List[Tensor],
        *,
        lr: float):
    
    g_norm = 0
    for i in d_p_list:
        g_norm += torch.norm(i, p='fro')**2

    for i, param in enumerate(params):

        d_p = d_p_list[i]

        alpha = -lr
        f_norm_sq = (torch.norm(param.grad, p='fro'))**2
        a_l = g_norm/f_norm_sq
        if f_norm_sq != 0:
            param.add_(a_l*d_p, alpha=alpha)



import torch
from torch import functional as F
from torch import optim
from torch.optim import Optimizer 
from torch.optim.optimizer import required


class new_alg(Optimizer):
    
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(new_alg, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(new_alg, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

#        for group in self.param_groups:
#            g_norm = 0

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)  #פרמטרים שהגדריאנט שלהם שונה מ-0
                    d_p_list.append(p.grad)     #הגרדיאנטים עצמם

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            
            new_alg_comp(params_with_grad, d_p_list, lr=lr)
            
            #F.sgd(params_with_grad,
            #      d_p_list,
            #      momentum_buffer_list,
            #      weight_decay=weight_decay,
            #      momentum=momentum,
            #      lr=lr,
            #      dampening=dampening,
            #      nesterov=nesterov,
            #      maximize=maximize,)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


