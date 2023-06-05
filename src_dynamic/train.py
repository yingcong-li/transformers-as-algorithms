import os
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import Sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(ys)
    loss = loss_func(output[:,:-1,:], ys[:,1:,:])
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach().cpu()


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    observed_dims = model.observed_dims
    bsize = args.training.batch_size
    data_sampler = Sampler()
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        observed_dims,
        bsize,
        **args.training.task_kwargs
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))


    for i in pbar:
        task_sampler_args = {}

        task = task_sampler(**task_sampler_args)
        xs, ys, _ = data_sampler.sample_xs(
            curriculum.n_points,
            curriculum.n_dims_truncated,
            task,
        )

        loss_func = task.get_training_metric()

        loss, output = train_step(model, ys.cuda(), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points-1))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output[:,:-1,:], ys[:,1:,:]).mean(dim=0)


        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, list(point_wise_loss.numpy()))
                    ),
                    "n_points": curriculum.n_points-1,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
