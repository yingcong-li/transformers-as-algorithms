import torch


class Sampler:
    def __init__(self):
        super().__init__()

    def sample_xs(self, n_points, n_dims, task):
        b_size = task.b_size
        # set initial x to be zeros
        x = torch.zeros(b_size, 1, n_dims)
        xs_b = torch.zeros(b_size, n_points, n_dims)
        ys_b = torch.zeros(b_size, n_points, task.observed_dims)
        ys_b_noiseless = torch.zeros(b_size, n_points, task.observed_dims)
        for i in range(n_points):
            xs_b[ :, i, :] = x[ :, 0, :]
            x, y_b, y_b_noiseless = task.evaluate(x)
            ys_b[ :, i, :] = y_b[ :, 0, :]
            ys_b_noiseless[ :, i, :] = y_b_noiseless[ :, 0, :]
        return xs_b, ys_b, ys_b_noiseless
