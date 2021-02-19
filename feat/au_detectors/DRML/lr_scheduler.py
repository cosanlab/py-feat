# https://github.com/ZhiwenShao/PyTorch-JAANet/blob/master/lr_schedule.py

def step_lr_scheduler(param_lr, optimizer, iter_num, gamma, stepsize, init_lr = 0.001):
    lr = init_lr * (gamma ** (iter_num // stepsize))

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * param_lr[i]
    
    return optimizer