import json
import os
import sys

from munch import Munch
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler
from tasks import get_task_sampler, sample_seeds


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


def eval_batch(model, task, xs):
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2"]:
        device = "cuda"
    else:
        device = "cpu"

    ys = task.evaluate(xs)
    pred = model(xs.to(device), ys.to(device)).detach()
    metrics = task.get_metric()(pred.cpu(), ys)

    return metrics


def eval_batch_robust(model, task, xs, xs_):
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2"]:
        device = "cuda"
    else:
        device = "cpu"

    ys = task.evaluate(xs)
    ys_ = -task.evaluate(xs_)
    b_size, n_points, _ = xs.shape
    pred = model(xs.to(device), ys.to(device)).detach()
    metrics = task.get_metric()(pred.cpu(), ys)

    metrics_new = torch.zeros(b_size, n_points)

    xs_new = torch.cat((xs_[:, :1],xs[:, 1:]), dim=1)
    ys_new = torch.cat((ys_[:, :1],ys[:, 1:]), dim=1)
    pred_new = model(xs_new.to(device), ys_new.to(device)).detach()
    metrics_new = (task.get_metric()(pred_new.cpu(), ys_new) - metrics).abs()
    
    return (metrics_new)


def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    model,
    robust_eval,
    task_name,
    data_name,
    n_tasks,
    n_dims,
    n_points,
    batch_size,
    data_kwargs,
    task_kwargs,
    num_eval_batchs=100,
    mtl_eval=False,
):

    """
    Evaluate a model on a task with a variety of strategies.
    """        
    data_sampler = get_data_sampler(data_name, n_dims, **data_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_kwargs
    )

    all_metrics = []

    for i in range(num_eval_batchs):
        task_sampler_args = {}
        if mtl_eval:
            seeds = sample_seeds(n_tasks, batch_size)
            task_sampler_args["seeds"] = [s + 1 for s in seeds]
        task = task_sampler(**task_sampler_args)
        
        xs = gen_standard(data_sampler, n_points, batch_size)

        if not robust_eval:
            metrics = eval_batch(model, task, xs)
        else:
            xs_ = gen_standard(data_sampler, 1, batch_size)
            metrics = eval_batch_robust(model, task, xs, xs_)
        all_metrics.append(metrics)
    
    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data
    n_tasks = conf.training.n_sequences \
        if conf.training.n_sequences is not None \
            else conf.training.n_tasks

    task_kwargs = conf.training.task_kwargs
    data_kwargs = conf.training.data_kwargs

    kwargs = {
        "task_name": task_name,
        "task_kwargs": task_kwargs,
        "n_tasks": n_tasks,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "data_kwargs": data_kwargs,
    }
    return kwargs


def compute_evals(all_models, kwargs, save_path=None, robust_eval=False):
    metrics = {}

    for model in tqdm(all_models):
        metrics[model.name] = eval_model(model, robust_eval, **kwargs)
        if kwargs["n_tasks"] is not None:
            metrics["MTL_"+model.name] = eval_model(model, robust_eval, mtl_eval=True, **kwargs)

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(metrics, fp, indent=2)

    return metrics


def get_run_metrics(
    run_path, step=-1, cache=True, recompute_metrics=False, skip_baselines=False, robust_eval=False
):
    """
    Compute the metrics when recompute_metrics is True.
    Compute the metrics when recompute_metrics is False and cache is True and the metrics are not cached.
    """
    if not cache:
        save_path = None
    elif step == -1:
        if robust_eval:
            save_path = os.path.join(run_path, "metrics_robust.json")
        else:
            save_path = os.path.join(run_path, "metrics.json")
    else:
        if robust_eval:
            save_path = os.path.join(run_path, f"metrics_robust_{step}.json")
        else:
            save_path = os.path.join(run_path, f"metrics_{step}.json")
    
    if not recompute_metrics:
        try:
            with open(save_path) as fp:
                return json.load(fp)
        except:
            print("Metrics not found. Computing metrics...")

    model, conf = get_model_from_run(run_path, step)
    model = model.cuda().eval()
    all_models = [model]
    if not skip_baselines:
        all_models += models.get_relevant_baselines(conf.training.task)
    evaluation_kwargs = build_evals(conf)

    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, robust_eval)
    return all_metrics


def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if "ridge" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Ridge (alpha={alpha})"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            n_tasks = (
                conf.training.n_sequences 
                if conf.training.n_sequences is not None 
                else conf.training.n_tasks
            )
            params["n_tasks"] = n_tasks if n_tasks is not None else -1
            n_sequences = (
                conf.training.n_sequences
                if conf.training.n_sequences is not None
                else None
            )
            params["n_sequences"] = n_sequences if n_sequences is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name
            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    return df


if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)
