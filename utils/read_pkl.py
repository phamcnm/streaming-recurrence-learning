import pickle

# Input/output paths
pickle_file = "data_rnn_variants_ppo_discrete.py__exp_id_0/RepeatFirst_5/arch__mynet/seed_8.pkl"
output_file = "utils/pkl_out.txt"


def _format_list(values, precision=3):
    if values is None:
        return "None"
    if not values:
        return "[]"
    return "[" + ", ".join(f"{v:.{precision}f}" for v in values) + "]"


def _format_grad_norms(grad_norms):
    if not grad_norms:
        return "None"
    items = [f"{k}={v:.2e}" for k, v in sorted(grad_norms.items())]
    return ", ".join(items)


def _format_seq_grad_norms(seq_grad_norms):
    if not seq_grad_norms:
        return "None"
    items = []
    for name, stats in sorted(seq_grad_norms.items()):
        items.append(
            f"{name}(min={stats['min']:.2e},mean={stats['mean']:.2e},max={stats['max']:.2e})"
        )
    return ", ".join(items)


with open(pickle_file, "rb") as f:
    data = pickle.load(f)

returns = []
timesteps = []
tracking = None
extra = None

if isinstance(data, dict):
    returns = data.get("returns", [])
    timesteps = data.get("timesteps", [])
    tracking = data.get("tracking")
elif isinstance(data, (list, tuple)):
    if len(data) >= 2:
        returns, timesteps = data[0], data[1]
    if len(data) >= 3:
        extra = data[2]
else:
    raise ValueError(f"Unsupported pickle format: {type(data)}")

lines = []
lines.append(f"pickle_file: {pickle_file}")
lines.append(f"num_entries: {len(returns)}")
lines.append("")

if tracking is None:
    for idx in range(len(returns)):
        r = returns[idx]
        t = timesteps[idx] if idx < len(timesteps) else None
        extra_val = extra[idx] if isinstance(extra, (list, tuple)) and idx < len(extra) else None
        lines.append(f"{idx:04d} | step={t} | return={r} | extra={extra_val}")
else:
    rnn_dorm = tracking.get("actor_rnn_dormancy", [])
    mlp_dorm = tracking.get("actor_mlp_dormancy", [])
    grad_norms = tracking.get("grad_norms", [])
    seq_grad_norms = tracking.get("seq_grad_norms", [])
    for idx in range(len(returns)):
        r = returns[idx]
        t = timesteps[idx] if idx < len(timesteps) else None
        rnn = rnn_dorm[idx] if idx < len(rnn_dorm) else None
        mlp = mlp_dorm[idx] if idx < len(mlp_dorm) else None
        grads = grad_norms[idx] if idx < len(grad_norms) else None
        seq_grads = seq_grad_norms[idx] if idx < len(seq_grad_norms) else None
        lines.append(f"{idx:04d} | step={t} | return={r}")
        lines.append(f"  rnn_dormancy={_format_list(rnn)} | mlp_dormancy={mlp}")
        lines.append(f"  grad_norms={_format_grad_norms(grads)}")
        lines.append(f"  seq_grad_norms={_format_seq_grad_norms(seq_grads)}")
        lines.append("")

with open(output_file, "w") as f:
    f.write("\n".join(lines))
