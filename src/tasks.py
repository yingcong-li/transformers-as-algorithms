import torch
from random import randint

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


class Task:
    def __init__(self, n_dims, batch_size, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.seeds = seeds

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(task_name, n_dims, batch_size, **kwargs):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "noisy_linear_regression": LinearRegression,
        "hetero_linear_regression": HeteroLinearRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        return lambda **args: task_cls(n_dims, batch_size, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(
        self, 
        n_dims, 
        batch_size, 
        seeds=None, 
        scale=1,
        noise_std=0,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, seeds)
        self.scale = scale
        self.noise_std = noise_std

        if seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        else:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        ys_b = ys_b + torch.randn_like(ys_b) * self.noise_std
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class HeteroLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        seeds=None,
        scale=1,
        base=2.,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(HeteroLinearRegression, self).__init__(
            n_dims, batch_size, seeds, scale
        )
        self.base = base
        Sigma = torch.tensor([i**(-base) for i in range(1, n_dims+1)])
        Sigma = torch.sqrt(torch.diag(Sigma))
        for i, w in enumerate(self.w_b):
            self.w_b[i] = Sigma.mm(w)

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

