import math
import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square().sum(dim=-1)


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().sum(dim=-1).mean()


class Task:
    def __init__(self, n_dims, batch_size):
        self.n_dims = n_dims
        self.b_size = batch_size

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(task_name, n_dims, observed_dims, batch_size, **kwargs):
    task_names_to_classes = {
        "linear_dynamics": LinearDynamics,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        return lambda **args: task_cls(n_dims, observed_dims, batch_size, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearDynamics(Task):
    def __init__(self, n_dims, observed_dims, batch_size, radius=0.9, noise_std=0.1):
        super(LinearDynamics, self).__init__(n_dims, batch_size)
        self.radius = radius
        self.noise_std = noise_std
        self.observed_dims = observed_dims

        self.w_b = torch.randn(self.b_size, self.n_dims, self.n_dims)
        for i in range(self.b_size):
            self.w_b[i] *= self.radius/(torch.linalg.eigvals(self.w_b[i]).abs().max())

        self.c_b = torch.randn(self.b_size, self.n_dims, observed_dims) / math.sqrt(observed_dims)

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        c_b = self.c_b.to(xs_b.device)
        xs_b_next = (xs_b @ w_b)
        xs_b_next += torch.randn_like(xs_b_next) * self.noise_std
        ys_b  = xs_b_next @ c_b
        return xs_b_next, ys_b, (xs_b @ w_b) @ c_b 

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

