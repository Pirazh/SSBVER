import numpy as np
from bisect import bisect_right

def get_params_groups(model, wd=None):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized, 'weight_decay': wd}, 
            {'params': not_regularized, 'weight_decay': wd}]

def gamma_scheduler(base_value,
                    warmup_epochs,
                    epochs,
                    warmup_factor,
                    gamma,
                    milestones):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(warmup_factor, 1., warmup_epochs) * base_value
    schedule = [base_value * gamma ** bisect_right(milestones, epoch) 
        for epoch in range(warmup_epochs, epochs)]
    schedule = np.concatenate((warmup_schedule, np.asarray(schedule)))
    assert len(schedule) == epochs
    return schedule


def cosine_scheduler(base_value, 
                    final_value, 
                    epochs, 
                    warmup_epochs, 
                    warmup_factor):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(warmup_factor, 1.0, warmup_epochs) * base_value

    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule