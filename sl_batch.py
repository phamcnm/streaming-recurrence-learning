from importlib import import_module
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
        train_kwargs=None, eval_kwargs=None
    ):
    seeds = list(range(num_seeds))

    embed_dim = 64
    hidden_dim = 256
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

    plot_results(seq_lens, results)

def plot_results(seq_lens, results):
    plt.figure(figsize=(7, 5))
    for model_name, scores in results.items():
        plt.plot(seq_lens, scores, marker="o", label=model_name)
    plt.xlabel("Sequence length")
    plt.ylabel("Success rate (closer to 0 is better)")
    plt.title("Repeat-First: Memory vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

def run_task(task_name, vocab_size=None, num_seeds=1, seq_lens=None, rnn_list=None, rnn_mode="sequential"):
    task = load_task(task_name, vocab_size=vocab_size)
    run_experiment(
        task["train"],
        task["evaluate"],
        input_dim=task["input_dim"],
        output_dim=task["output_dim"],
        num_seeds=num_seeds,
        seq_lens=seq_lens,
        rnn_list=rnn_list,
        rnn_mode=rnn_mode,
        train_kwargs=task.get("train_kwargs"),
        eval_kwargs=task.get("eval_kwargs"),
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="repeat_first_one")
    parser.add_argument("--vocab-size", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seq-lens", type=int, nargs="*", default=[5])
    parser.add_argument("--rnn-list", type=str, nargs="*", default=["glru", "lru", "gru", "rnn"])
    parser.add_argument("--rnn-mode", type=str, default="act")
    
    args = parser.parse_args()
    run_task(
        args.task,
        vocab_size=args.vocab_size,
        num_seeds=args.num_seeds,
        seq_lens=args.seq_lens,
        rnn_list=args.rnn_list,
        rnn_mode=args.rnn_mode,
    )
