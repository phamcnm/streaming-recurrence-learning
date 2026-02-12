from importlib import import_module
import os
import pickle
import sys
from recurrent_wrappers.create_models import create_model
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class Agent(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        rnn_name="lru",
        arch="none",
        rnn_mode="act",
        input_is_one_hot=False,
    ):
        super().__init__()
        if input_is_one_hot:
            if embed_dim != input_dim:
                raise ValueError("For one-hot inputs, embed_dim must equal input_dim.")
            self.embedding = nn.Identity()
        else:
            self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn, rnn_out_dim = create_model(
            name=rnn_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            arch=arch,
            rnn_mode=rnn_mode,
            use_layernorm=False
        )
        self.readout = nn.Linear(rnn_out_dim, output_dim)

    def forward(self, x, hidden=None, **kwargs):
        x = self.embedding(x)
        x, hidden, aux = self.rnn(x, hidden, **kwargs)
        x = self.readout(x)
        return x, hidden, aux

def run_experiment(
        train, evaluate, input_dim, output_dim, num_seeds=1,
        seq_lens=[40], rnn_list=['lru', 'glru'], rnn_mode='sequential',
        train_kwargs=None, eval_kwargs=None, task_name=None, vocab_size=None,
        arch="none",
        exp_id=None, exp_desc=None, input_is_one_hot=False
    ):
    seeds = list(range(num_seeds))

    embed_dim = input_dim if input_is_one_hot else 64
    hidden_dim = 128
    train_kwargs = train_kwargs or {}
    eval_kwargs = eval_kwargs or {}

    results = {name: [] for name in rnn_list}
    for seq_len in seq_lens:
        print(f"\n=== Sequence length: {seq_len} ===")
        for rnn in rnn_list:
            scores = []
            for seed in seeds:
                print(f"Model: {rnn} | Seed: {seed}")
                set_seed(seed)
                model = Agent(
                    input_dim=input_dim,
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    rnn_name=rnn,
                    arch=arch,
                    rnn_mode=rnn_mode,
                    input_is_one_hot=input_is_one_hot,
                )
                loss_history = None
                try:
                    train_out = train(
                        model,
                        seq_len=seq_len,
                        return_loss_history=True,
                        **train_kwargs,
                    )
                except TypeError:
                    train_out = train(model, seq_len=seq_len, **train_kwargs)
                if isinstance(train_out, tuple):
                    model, loss_history = train_out
                else:
                    model = train_out
                score = evaluate(model, seq_len, show_results=seed==0, **eval_kwargs)
                scores.append(score)
                if loss_history is not None:
                    base_dir = None
                    exp_suffix = f"__exp_id_{exp_id}" if exp_id is not None else ""
                    if task_name is not None and vocab_size is not None:
                        base_dir = f"data_sl_{task_name}_vocab_{vocab_size}{exp_suffix}"
                    elif task_name is not None:
                        base_dir = f"data_sl_{task_name}{exp_suffix}"
                    else:
                        base_dir = f"data_sl_unknown_task{exp_suffix}"
                    run_dir = os.path.join(base_dir, 'seq-len__'+str(seq_len), rnn)
                    if not os.path.exists(run_dir):
                        try:
                            os.makedirs(run_dir)
                        except FileExistsError:
                            pass
                    seed_path = os.path.join(run_dir, f"seed_{seed}.pkl")
                    with open(seed_path, "wb") as f:
                        pickle.dump(
                            {
                                "losses": loss_history,
                                "eval_score": score,
                            },
                            f,
                        )

            mean_score = np.mean(scores)
            results[rnn].append(mean_score)
            print(f"Eval Mean Success: {mean_score:.2f}")

    # plot_results(
    #     seq_lens,
    #     results,
    #     task_name=task_name,
    #     vocab_size=vocab_size,
    #     exp_id=exp_id,
    #     exp_desc=exp_desc,
    # )

def plot_results(seq_lens, results, task_name=None, vocab_size=None, exp_id=None, exp_desc=None):
    plt.figure(figsize=(7, 5))
    if results:
        num_models = len(results)
    else:
        num_models = 1
    jitter_scale = 0.1
    jitter_center = (num_models - 1) / 2.0
    for idx, (model_name, scores) in enumerate(results.items()):
        offset = (idx - jitter_center) * jitter_scale
        seq_lens_jittered = [s + offset for s in seq_lens]
        plt.plot(seq_lens_jittered, scores, marker="o", label=model_name)
    plt.xlabel("Sequence length")
    plt.ylabel("Evaluation score")
    if task_name is not None:
        plt.title(f"{task_name}: Sequence Length Sweep")
    else:
        plt.title("Sequence Length Sweep")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if task_name is not None and vocab_size is not None:
        exp_suffix = f"__exp_id_{exp_id}" if exp_id is not None else ""
        folder_name = f"data_sl_{task_name}_vocab_{vocab_size}{exp_suffix}"
        if not os.path.exists(folder_name):
            try:
                os.makedirs(folder_name)
            except FileExistsError:
                pass
        plot_name = f"plot_{exp_id}.jpg" if exp_id is not None else "plot.jpg"
        plot_path = os.path.join(folder_name, plot_name)
        plt.savefig(plot_path, dpi=200)
        desc_name = f"desc_{exp_id}.txt" if exp_id is not None else "desc.txt"
        desc_path = os.path.join(folder_name, desc_name)
        with open(desc_path, "w", encoding="utf-8") as desc_file:
            desc_file.write((exp_desc or "") + "\n")
    # plt.show()

def set_seed(seed):
    import random, numpy, torch
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

def load_task(task_name, vocab_size=None):
    module_name = f"sl_tasks.{task_name}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unknown task '{task_name}'. Expected module {module_name}.") from exc

    if hasattr(module, "build_task"):
        if vocab_size is None:
            if hasattr(module, "vocab_size"):
                vocab_size = module.vocab_size
            else:
                raise ValueError(f"Task '{task_name}' requires vocab_size.")
        task = module.build_task(vocab_size)
    elif hasattr(module, "TASK"):
        task = module.TASK
    else:
        required = ["train", "evaluate", "input_dim", "output_dim"]
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise ValueError(
                f"Task '{task_name}' is missing required attributes: {', '.join(missing)}"
            )
        task = {
            "name": task_name,
            "train": module.train,
            "evaluate": module.evaluate,
            "input_dim": module.input_dim,
            "output_dim": module.output_dim,
        }
    return task

def run_task(
    task_name,
    vocab_size=None,
    num_seeds=1,
    seq_lens=None,
    rnn_list=None,
    rnn_mode="sequential",
    arch="none",
    exp_id=None,
    exp_desc=None,
    num_epochs=None,
    num_samples=None,
    one_hot=False,
    variable_k=False,
    cli_args=None,
):
    task = load_task(task_name, vocab_size=vocab_size)
    train_kwargs = dict(task.get("train_kwargs") or {})
    eval_kwargs = dict(task.get("eval_kwargs") or {})
    if num_epochs is not None:
        train_kwargs["num_epochs"] = num_epochs
    if num_samples is not None:
        train_kwargs["num_samples"] = num_samples
    if arch is None:
        arch = "none"
    if one_hot:
        train_kwargs["one_hot"] = True
        eval_kwargs["one_hot"] = True
    if variable_k:
        train_kwargs["variable_k"] = True
        eval_kwargs["variable_k"] = True
    task_label = task_name
    if task_name == "copy_task":
        one_hot = train_kwargs.get("one_hot")
        variable_k = train_kwargs.get("variable_k")
        if one_hot is True:
            task_label = f"{task_label}_one_hot"
        elif one_hot is False:
            task_label = f"{task_label}_token"
        if variable_k is True:
            task_label = f"{task_label}_variable_k"
    exp_suffix = f"__exp_id_{exp_id}" if exp_id is not None else ""
    if task_label is not None and vocab_size is not None:
        base_dir = f"data_sl_{task_label}_vocab_{vocab_size}{exp_suffix}"
    elif task_label is not None:
        base_dir = f"data_sl_{task_label}{exp_suffix}"
    else:
        base_dir = f"data_sl_unknown_task{exp_suffix}"
    if not os.path.exists(base_dir):
        try:
            os.makedirs(base_dir)
        except FileExistsError:
            pass
    notes_name = f"notes_{exp_id}.txt" if exp_id is not None else "notes.txt"
    notes_path = os.path.join(base_dir, notes_name)
    try:
        fd = os.open(notes_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            plot_cmd = f"python utils/plot_sl.py --dir {base_dir}"
            f.write(f"{plot_cmd}\n\n")
            if exp_desc:
                f.write(f"exp_desc: {exp_desc}\n")
            args_dict = vars(cli_args) if cli_args is not None else {
                "task": task_name,
                "vocab_size": vocab_size,
                "num_seeds": num_seeds,
                "seq_lens": seq_lens,
                "rnn_list": rnn_list,
                "rnn_mode": rnn_mode,
                "exp_id": exp_id,
                "exp_desc": exp_desc,
                "num_epochs": num_epochs,
                "num_samples": num_samples,
                "one_hot": one_hot,
                "variable_k": variable_k,
            }
            for key, value in args_dict.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n{' '.join(sys.argv)}\n")
    except FileExistsError:
        pass
    run_experiment(
        task["train"],
        task["evaluate"],
        input_dim=task["input_dim"],
        output_dim=task["output_dim"],
        num_seeds=num_seeds,
        seq_lens=seq_lens,
        rnn_list=rnn_list,
        rnn_mode=rnn_mode,
        train_kwargs=train_kwargs,
        eval_kwargs=eval_kwargs,
        task_name=task_label,
        vocab_size=vocab_size,
        arch=arch,
        exp_id=exp_id,
        exp_desc=exp_desc,
        input_is_one_hot=task.get("input_is_one_hot", False),
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="copy_task")
    parser.add_argument("--vocab-size", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="*", default=[7, 10])
    parser.add_argument("--rnn-list", type=str, nargs="*", default=['gru', 'glru'])
    parser.add_argument("--rnn-mode", type=str, default="sequential")
    parser.add_argument("--arch", type=str, default="none")
    parser.add_argument("--exp-id", type=int, default=0)
    parser.add_argument("--exp-desc", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--one-hot", action="store_true")
    parser.add_argument("--variable-k", action="store_true")
    
    args = parser.parse_args()
    run_task(
        args.task,
        vocab_size=args.vocab_size,
        num_seeds=args.num_seeds,
        seq_lens=args.seq_lens,
        rnn_list=args.rnn_list,
        rnn_mode=args.rnn_mode,
        arch=args.arch,
        exp_id=args.exp_id,
        exp_desc=args.exp_desc,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        one_hot=args.one_hot,
        variable_k=args.variable_k,
        cli_args=args,
    )
