from importlib import import_module
import os
from recurrent_wrappers.create_models import create_model
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class Agent(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, rnn_name="lru", rnn_mode="act"):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn, rnn_out_dim = create_model(
            name=rnn_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            arch='memora',
            rnn_mode=rnn_mode,
            layernorm=False
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
        exp_id=None, exp_desc=None
    ):
    seeds = list(range(num_seeds))

    embed_dim = 64
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
                    rnn_mode=rnn_mode,
                )
                train(model, seq_len=seq_len, **train_kwargs)
                score = evaluate(model, seq_len, show_results=seed==0, **eval_kwargs)
                scores.append(score)

            mean_score = np.mean(scores)
            results[rnn].append(mean_score)
            print(f"Eval Mean Success: {mean_score:.2f}")

    plot_results(
        seq_lens,
        results,
        task_name=task_name,
        vocab_size=vocab_size,
        exp_id=exp_id,
        exp_desc=exp_desc,
    )

def plot_results(seq_lens, results, task_name=None, vocab_size=None, exp_id=None, exp_desc=None):
    plt.figure(figsize=(7, 5))
    for model_name, scores in results.items():
        plt.plot(seq_lens, scores, marker="o", label=model_name)
    plt.xlabel("Sequence length")
    plt.ylabel("Success rate (closer to 0 is better)")
    plt.title("Repeat-First: Memory vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if exp_id is not None and task_name is not None and vocab_size is not None:
        folder_name = f"data_sl/exp_{exp_id}_{task_name}_{vocab_size}"
        os.makedirs(folder_name, exist_ok=True)
        plot_path = os.path.join(folder_name, f"plot_{exp_id}.jpg")
        plt.savefig(plot_path, dpi=200)
        desc_path = os.path.join(folder_name, f"desc_{exp_id}.txt")
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
    exp_id=None,
    exp_desc=None,
    num_epochs=None,
    num_samples=None,
):
    task = load_task(task_name, vocab_size=vocab_size)
    train_kwargs = dict(task.get("train_kwargs") or {})
    if num_epochs is not None:
        train_kwargs["num_epochs"] = num_epochs
    if num_samples is not None:
        train_kwargs["num_samples"] = num_samples
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
        eval_kwargs=task.get("eval_kwargs"),
        task_name=task_name,
        vocab_size=vocab_size,
        exp_id=exp_id,
        exp_desc=exp_desc,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="repeat_previous")
    parser.add_argument("--vocab-size", type=int, default=3)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="*", default=[10])
    parser.add_argument("--rnn-list", type=str, nargs="*", default=['glru', 'lru'])
    parser.add_argument("--rnn-mode", type=str, default="sequential")
    parser.add_argument("--exp-id", type=int, default=0)
    parser.add_argument("--exp-desc", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=20000)
    
    args = parser.parse_args()
    run_task(
        args.task,
        vocab_size=args.vocab_size,
        num_seeds=args.num_seeds,
        seq_lens=args.seq_lens,
        rnn_list=args.rnn_list,
        rnn_mode=args.rnn_mode,
        exp_id=args.exp_id,
        exp_desc=args.exp_desc,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
    )
