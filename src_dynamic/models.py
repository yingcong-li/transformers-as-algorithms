import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from sklearn.linear_model import Ridge



def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            observed_dims=conf.observed_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": 
          [(LeastSquaresModel, {"H": h}) for h in [1,2,3,4]]
        + [(RidgeModel, {"H":1, "alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]]
        + [(RidgeModel, {"H":2, "alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]]
        + [(RidgeModel, {"H":3, "alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]]
        + [(RidgeModel, {"H":4, "alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, observed_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
            
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.observed_dims = observed_dims
        self._read_in = nn.Linear(observed_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, observed_dims)

    def forward(self, xs, inds=None):
        if inds is None:
            inds = torch.arange(xs.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= xs.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices that are not defined")
        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, inds, :]


class LeastSquaresModel:
    # H is the window size/number of previous points to use
    def __init__(self, H, driver):
        self.H = H
        self.driver= driver
        self.name = f"OLS_H={H}_driver={driver}"

    def __call__(self, xs, inds=None):
        xs = xs.cpu()
        if inds is None:
            inds = range(xs.shape[1])
        else:
            if max(inds) >= xs.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices that are not defined")

        preds = []
        for i in inds:
            if i <= self.H:
                preds.append(torch.zeros_like(xs[:, 0]))  # predict zero for first point
                continue
            train_zs = torch.zeros(xs.shape[0], i-self.H, int(xs.shape[2]*self.H))
            for j in range(i-self.H):
                train_zs[:,j,:] = xs[:,j:j+self.H,:].reshape(xs.shape[0],-1)
            train_ys = xs[:, self.H:i]
            test_z = xs[:, i - self.H + 1 : i + 1].reshape(xs.shape[0],1,-1)

            ws, _, _, _ = torch.linalg.lstsq(
                train_zs, train_ys, driver=self.driver
            )

            pred = test_z @ ws

            preds.append(pred[:, 0, :])

        return torch.stack(preds, dim=1)


class RidgeModel:
    # H is the window size/number of previous points to use
    # alpha is the regularization parameter
    def __init__(self, H, alpha, driver):
        self.H = H
        self.alpha = alpha
        self.driver = driver
        self.name = f"Ridge_H={H}_alpha={alpha}_driver={driver}"

    def __call__(self, xs, inds=None):
        xs = xs.cpu()
        if inds is None:
            inds = range(xs.shape[1])
        else:
            if max(inds) >= xs.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i <= self.H:
                preds.append(torch.zeros_like(xs[:, 0]))  # predict zero for first point
                continue
            train_zs = torch.zeros(xs.shape[0], i-self.H, int(xs.shape[2]*self.H))
            for j in range(i-self.H):
                train_zs[:,j,:] = xs[:,j:j+self.H,:].reshape(xs.shape[0],-1)
            train_ys = xs[:, self.H:i]
            test_z = xs[:, i - self.H + 1 : i + 1].reshape(xs.shape[0],1,-1)

            pred = torch.zeros_like(xs[:, 0])
            for bs in range(pred.shape[0]):
                clf = Ridge(
                    alpha=self.alpha, fit_intercept=False, max_iter=100000
                )

                clf.fit(train_zs[bs], train_ys[bs])
                pred[bs] = torch.from_numpy(clf.predict(test_z[bs]))

            preds.append(pred)
        return torch.stack(preds, dim=1)
