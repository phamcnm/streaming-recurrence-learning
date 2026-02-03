import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def avg_return_curve(x, y, stride, total_steps):
    """
    :param x: A list of list of termination steps for each episode. len(x) == total number of runs
    :param y: A list of list of episodic return. len(y) == total number of runs
    :param stride: The timestep interval between two aggregate datapoints to be calculated
    :param total_steps: The total number of time steps to be considered
    :return: time steps for calculated data points, average returns for each data points, std-errs
    """
    assert len(x) == len(y)
    num_runs = len(x)
    avg_ret = np.zeros(total_steps // stride)
    stderr_ret = np.zeros(total_steps // stride)
    steps = np.arange(stride, total_steps+1, stride)
    for i in range(0, total_steps // stride):
        rets = []
        avg_rets_per_run = []
        for run in range(num_runs):
            xa = np.array(x[run])
            ya = np.array(y[run])
            rets.append(ya[np.logical_and(i * stride < xa, xa <= (i + 1) * stride)].tolist())
            avg_rets_per_run.append(np.mean(rets[-1]))
        avg_ret[i] = np.mean(avg_rets_per_run)
        stderr_ret[i] = np.std(avg_rets_per_run) / np.sqrt(num_runs)
    return steps, avg_ret, stderr_ret

def main(parent_dir, int_space, total_steps, exp_name, notes, metric):
    plotted_any = False
    layer1_dirs = [d for d in os.listdir(parent_dir) 
                   if os.path.isdir(os.path.join(parent_dir, d))]
    layer1_dirs.sort()  # Sort for consistent ordering
    
    # Check if there's a layer 3 by examining the first layer2 dir of the first layer1 dir
    sample_layer1 = os.path.join(parent_dir, layer1_dirs[0])
    layer2_dirs = [d for d in os.listdir(sample_layer1) 
                   if os.path.isdir(os.path.join(sample_layer1, d))]
    layer2_dirs.sort()
    
    sample_layer2 = os.path.join(sample_layer1, layer2_dirs[0])
    layer3_dirs = [d for d in os.listdir(sample_layer2) 
                   if os.path.isdir(os.path.join(sample_layer2, d))]
    has_layer3 = len(layer3_dirs) > 0
    
    if has_layer3:
        # 2D grid: rows = layer2 dirs, columns = layer1 dirs
        layer3_dirs.sort()
        num_rows = len(layer2_dirs)
        num_cols = len(layer1_dirs)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for col_idx, layer1_dir in enumerate(layer1_dirs):
            layer1_path = os.path.join(parent_dir, layer1_dir)
            layer2_list = [d for d in os.listdir(layer1_path) 
                          if os.path.isdir(os.path.join(layer1_path, d))]
            layer2_list.sort()
            
            for row_idx, layer2_dir in enumerate(layer2_list):
                ax = axes[row_idx, col_idx]
                layer2_path = os.path.join(layer1_path, layer2_dir)
                
                # Get all layer3+ directories (could be more layers beyond 3)
                leaf_dirs = get_all_leaf_paths(layer2_path)
                
                for leaf_path in leaf_dirs:
                    if metric == "returns":
                        all_termination_time_steps, all_episodic_returns, all_critic_accuracy = [], [], []
                        read_dir(leaf_path, all_termination_time_steps, all_episodic_returns, all_critic_accuracy)
                        if len(all_termination_time_steps) > 0:
                            steps, avg_ret, stderr_ret = avg_return_curve(
                                all_termination_time_steps,
                                all_episodic_returns,
                                all_termination_time_steps[0][0] if int_space == 0 else int_space,
                                total_steps,
                            )
                            label = os.path.relpath(leaf_path, layer2_path).replace(os.sep, '_')
                            ax.plot(steps, avg_ret, label=label)
                            ax.fill_between(steps, avg_ret - stderr_ret, avg_ret + stderr_ret, alpha=0.3)
                            plotted_any = True
                    elif metric == "grad_norms":
                        series_list = []
                        timestep_list = []
                        for file in os.listdir(leaf_path):
                            if file.endswith(".pkl"):
                                run = load_run_data(os.path.join(leaf_path, file))
                                if run is None:
                                    continue
                                series = _mean_grad_norm_series(run.get("tracking"))
                                if series:
                                    series_list.append(series)
                                    timestep_list.append(run.get("timesteps", []))
                        mean, stderr = _aggregate_scalar_series(series_list)
                        if mean.size > 0:
                            if timestep_list and timestep_list[0]:
                                min_len = min(len(s) for s in series_list)
                                x = np.array(timestep_list[0][:min_len])
                            else:
                                x = np.arange(1, len(mean) + 1)
                            label = os.path.relpath(leaf_path, layer2_path).replace(os.sep, '_')
                            ax.plot(x, mean, label=label)
                            ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.3)
                            plotted_any = True
                    elif metric == "dormancy_actor_rnn":
                        series_list = []
                        timestep_list = []
                        for file in os.listdir(leaf_path):
                            if file.endswith(".pkl"):
                                run = load_run_data(os.path.join(leaf_path, file))
                                if run is None:
                                    continue
                                series = _mean_rnn_dormancy_series(run.get("tracking"))
                                if series:
                                    series_list.append(series)
                                    timestep_list.append(run.get("timesteps", []))
                        mean, stderr = _aggregate_scalar_series(series_list)
                        if mean.size > 0:
                            if timestep_list and timestep_list[0]:
                                min_len = min(len(s) for s in series_list)
                                x = np.array(timestep_list[0][:min_len])
                            else:
                                x = np.arange(1, len(mean) + 1)
                            label = os.path.relpath(leaf_path, layer2_path).replace(os.sep, '_')
                            ax.plot(x, mean, label=label)
                            ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.3)
                            plotted_any = True
                    elif metric == "dormancy_actor_mlp":
                        series_list = []
                        timestep_list = []
                        for file in os.listdir(leaf_path):
                            if file.endswith(".pkl"):
                                run = load_run_data(os.path.join(leaf_path, file))
                                if run is None:
                                    continue
                                series = _mean_mlp_dormancy_series(run.get("tracking"))
                                if series:
                                    series_list.append(series)
                                    timestep_list.append(run.get("timesteps", []))
                        mean, stderr = _aggregate_scalar_series(series_list)
                        if mean.size > 0:
                            if timestep_list and timestep_list[0]:
                                min_len = min(len(s) for s in series_list)
                                x = np.array(timestep_list[0][:min_len])
                            else:
                                x = np.arange(1, len(mean) + 1)
                            label = os.path.relpath(leaf_path, layer2_path).replace(os.sep, '_')
                            ax.plot(x, mean, label=label)
                            ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.3)
                            plotted_any = True
                    elif metric == "seq_grad_norms":
                        series_min = []
                        series_mean = []
                        series_max = []
                        timestep_list = []
                        for file in os.listdir(leaf_path):
                            if file.endswith(".pkl"):
                                run = load_run_data(os.path.join(leaf_path, file))
                                if run is None:
                                    continue
                                series = _mean_seq_grad_series(run.get("tracking"))
                                if series:
                                    series_min.append(series.get("min", []))
                                    series_mean.append(series.get("mean", []))
                                    series_max.append(series.get("max", []))
                                    timestep_list.append(run.get("timesteps", []))
                        label = os.path.relpath(leaf_path, layer2_path).replace(os.sep, '_')
                        color = ax._get_lines.get_next_color()
                        for name, series_list, linestyle, show_label in (
                            ("min", series_min, "--", False),
                            ("mean", series_mean, "-", True),
                            ("max", series_max, ":", False),
                        ):
                            mean, stderr = _aggregate_scalar_series(series_list)
                            if mean.size == 0:
                                continue
                            if timestep_list and timestep_list[0]:
                                min_len = min(len(s) for s in series_list)
                                x = np.array(timestep_list[0][:min_len])
                            else:
                                x = np.arange(1, len(mean) + 1)
                            ax.plot(x, mean, label=label if show_label else None, linestyle=linestyle, color=color)
                            ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.2, color=color)
                            plotted_any = True
                
                # ax.set_xlabel('Steps')
                # ax.set_ylabel('Average Return')
                ax.set_title(f'{layer1_dir} - {layer2_dir}', fontsize=12)
                ax.tick_params(axis='x', labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                ax.xaxis.get_offset_text().set_fontsize(6)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
    else:
        # 1D grid: one plot per layer1 dir, curves are layer2 dirs
        num_plots = len(layer1_dirs)
        fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
        if num_plots == 1:
            axes = [axes]
        
        for idx, layer1_dir in enumerate(layer1_dirs):
            ax = axes[idx]
            layer1_path = os.path.join(parent_dir, layer1_dir)
            layer2_list = [d for d in os.listdir(layer1_path) 
                          if os.path.isdir(os.path.join(layer1_path, d))]
            layer2_list.sort()
            
            for layer2_dir in layer2_list:
                layer2_path = os.path.join(layer1_path, layer2_dir)
                if metric == "returns":
                    all_termination_time_steps, all_episodic_returns, all_critic_accuracy = [], [], []
                    read_dir(layer2_path, all_termination_time_steps, all_episodic_returns, all_critic_accuracy)
                    if len(all_termination_time_steps) > 0:
                        steps, avg_ret, stderr_ret = avg_return_curve(
                            all_termination_time_steps,
                            all_episodic_returns,
                            all_termination_time_steps[0][0] if int_space == 0 else int_space,
                            total_steps,
                        )
                        ax.plot(steps, avg_ret, label=layer2_dir)
                        ax.fill_between(steps, avg_ret - stderr_ret, avg_ret + stderr_ret, alpha=0.3)
                        plotted_any = True
                elif metric == "grad_norms":
                    series_list = []
                    timestep_list = []
                    for file in os.listdir(layer2_path):
                        if file.endswith(".pkl"):
                            run = load_run_data(os.path.join(layer2_path, file))
                            if run is None:
                                continue
                            series = _mean_grad_norm_series(run.get("tracking"))
                            if series:
                                series_list.append(series)
                                timestep_list.append(run.get("timesteps", []))
                    mean, stderr = _aggregate_scalar_series(series_list)
                    if mean.size > 0:
                        if timestep_list and timestep_list[0]:
                            min_len = min(len(s) for s in series_list)
                            x = np.array(timestep_list[0][:min_len])
                        else:
                            x = np.arange(1, len(mean) + 1)
                        ax.plot(x, mean, label=layer2_dir)
                        ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.3)
                        plotted_any = True
                elif metric == "dormancy_actor_rnn":
                    series_list = []
                    timestep_list = []
                    for file in os.listdir(layer2_path):
                        if file.endswith(".pkl"):
                            run = load_run_data(os.path.join(layer2_path, file))
                            if run is None:
                                continue
                            series = _mean_rnn_dormancy_series(run.get("tracking"))
                            if series:
                                series_list.append(series)
                                timestep_list.append(run.get("timesteps", []))
                    mean, stderr = _aggregate_scalar_series(series_list)
                    if mean.size > 0:
                        if timestep_list and timestep_list[0]:
                            min_len = min(len(s) for s in series_list)
                            x = np.array(timestep_list[0][:min_len])
                        else:
                            x = np.arange(1, len(mean) + 1)
                        ax.plot(x, mean, label=layer2_dir)
                        ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.3)
                        plotted_any = True
                elif metric == "dormancy_actor_mlp":
                    series_list = []
                    timestep_list = []
                    for file in os.listdir(layer2_path):
                        if file.endswith(".pkl"):
                            run = load_run_data(os.path.join(layer2_path, file))
                            if run is None:
                                continue
                            series = _mean_mlp_dormancy_series(run.get("tracking"))
                            if series:
                                series_list.append(series)
                                timestep_list.append(run.get("timesteps", []))
                    mean, stderr = _aggregate_scalar_series(series_list)
                    if mean.size > 0:
                        if timestep_list and timestep_list[0]:
                            min_len = min(len(s) for s in series_list)
                            x = np.array(timestep_list[0][:min_len])
                        else:
                            x = np.arange(1, len(mean) + 1)
                        ax.plot(x, mean, label=layer2_dir)
                        ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.3)
                        plotted_any = True
                elif metric == "seq_grad_norms":
                    series_min = []
                    series_mean = []
                    series_max = []
                    timestep_list = []
                    for file in os.listdir(layer2_path):
                        if file.endswith(".pkl"):
                            run = load_run_data(os.path.join(layer2_path, file))
                            if run is None:
                                continue
                            series = _mean_seq_grad_series(run.get("tracking"))
                            if series:
                                series_min.append(series.get("min", []))
                                series_mean.append(series.get("mean", []))
                                series_max.append(series.get("max", []))
                                timestep_list.append(run.get("timesteps", []))
                    color = ax._get_lines.get_next_color()
                    for name, series_list, linestyle, show_label in (
                        ("min", series_min, "--", False),
                        ("mean", series_mean, "-", True),
                        ("max", series_max, ":", False),
                    ):
                        mean, stderr = _aggregate_scalar_series(series_list)
                        if mean.size == 0:
                            continue
                        if timestep_list and timestep_list[0]:
                            min_len = min(len(s) for s in series_list)
                            x = np.array(timestep_list[0][:min_len])
                        else:
                            x = np.arange(1, len(mean) + 1)
                        ax.plot(
                            x,
                            mean,
                            label=layer2_dir if show_label else None,
                            linestyle=linestyle,
                            color=color,
                        )
                        ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.2, color=color)
                        plotted_any = True
            
            # ax.set_xlabel('Steps')
            # ax.set_ylabel('Average Return')
            ax.set_title(layer1_dir)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.xaxis.get_offset_text().set_fontsize(6)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    space = 1 - 0.02*(exp_name!='') - 0.02*(notes!='')
    plt.tight_layout(rect=[0, 0, 1, space])
    if exp_name != '':
        fig.suptitle(exp_name, fontsize=14, y=0.99)  # main title
    if notes != '':
        fig.text(0.5, 0.95, notes, ha='center', fontsize=12)
    # fig.set_constrained_layout_pads(w_pad=10.0, h_pad=10.0, hspace=10.0)
    if not plotted_any:
        plt.close(fig)
        return False
    out_name = "learning_curves.png" if metric == "returns" else f"{metric}.png"
    plt.savefig(os.path.join(parent_dir, out_name), dpi=150, bbox_inches='tight')
    # plt.show()
    return True

def get_all_leaf_paths(root_dir):
    """
    Recursively find all leaf directories (directories containing only .pkl files, no subdirs)
    """
    leaf_paths = []
    
    def explore(current_dir):
        subdirs = [d for d in os.listdir(current_dir) 
                  if os.path.isdir(os.path.join(current_dir, d))]
        
        if len(subdirs) == 0:
            # This is a leaf directory
            leaf_paths.append(current_dir)
        else:
            # Recursively explore subdirectories
            for subdir in subdirs:
                explore(os.path.join(current_dir, subdir))
    
    explore(root_dir)
    return sorted(leaf_paths)

def load_run_data(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        return {
            "returns": payload.get("returns", []),
            "timesteps": payload.get("timesteps", []),
            "tracking": payload.get("tracking"),
        }
    if isinstance(payload, (list, tuple)):
        if len(payload) == 2:
            episodic_returns, termination_time_steps = payload
            return {
                "returns": episodic_returns,
                "timesteps": termination_time_steps,
                "tracking": None,
            }
        if len(payload) == 3:
            episodic_returns, termination_time_steps, _ = payload
            return {
                "returns": episodic_returns,
                "timesteps": termination_time_steps,
                "tracking": None,
            }
    return None


def read_dir(data_dir, all_termination_time_steps, all_episodic_returns, all_critic_accuracy):
    """
    Read all .pkl files in a directory and append data to the lists
    """
    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            run = load_run_data(os.path.join(data_dir, file))
            if run is None:
                continue
            episodic_returns = run["returns"]
            termination_time_steps = run["timesteps"]
            if len(episodic_returns) == 0 or len(termination_time_steps) == 0:
                continue
            all_termination_time_steps.append(termination_time_steps)
            all_episodic_returns.append(episodic_returns)
            all_critic_accuracy.append(None)


def _mean_grad_norm_series(tracking):
    if not tracking:
        return []
    grad_series = tracking.get("grad_norms")
    if not grad_series:
        return []
    series = []
    for entry in grad_series:
        if not entry:
            series.append(None)
            continue
        values = [v for v in entry.values() if v is not None]
        series.append(float(np.mean(values)) if values else None)
    return series


def _mean_rnn_dormancy_series(tracking):
    if not tracking:
        return []
    series = tracking.get("actor_rnn_dormancy")
    if not series:
        return []
    values = []
    for entry in series:
        if entry is None:
            values.append(None)
            continue
        if len(entry) == 0:
            values.append(None)
            continue
        values.append(float(np.mean(entry)))
    return values


def _mean_mlp_dormancy_series(tracking):
    if not tracking:
        return []
    series = tracking.get("actor_mlp_dormancy")
    if series is None:
        return []
    values = []
    for entry in series:
        if entry is None:
            values.append(None)
        else:
            values.append(float(entry))
    return values


def _mean_seq_grad_series(tracking):
    if not tracking:
        return {}
    series = tracking.get("seq_grad_norms")
    if not series:
        return {}
    out = {"min": [], "mean": [], "max": []}
    for entry in series:
        if not entry:
            for key in out:
                out[key].append(None)
            continue
        mins = []
        means = []
        maxs = []
        for stats in entry.values():
            if stats is None:
                continue
            mins.append(stats.get("min"))
            means.append(stats.get("mean"))
            maxs.append(stats.get("max"))
        out["min"].append(float(np.mean(mins)) if mins else None)
        out["mean"].append(float(np.mean(means)) if means else None)
        out["max"].append(float(np.mean(maxs)) if maxs else None)
    return out


def _aggregate_scalar_series(series_list):
    if not series_list:
        return np.array([]), np.array([])
    min_len = min(len(s) for s in series_list)
    if min_len == 0:
        return np.array([]), np.array([])
    arr = np.full((len(series_list), min_len), np.nan)
    for i, s in enumerate(series_list):
        for j in range(min_len):
            val = s[j]
            if val is None:
                continue
            arr[i, j] = val
    mean = np.nanmean(arr, axis=0)
    counts = np.sum(~np.isnan(arr), axis=0)
    stderr = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(counts, 1))
    return mean, stderr

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data_stream_ac__exp_id_45')
    parser.add_argument('--int_space', type=int, default=0) # if 0 then plot every point
    parser.add_argument('--total_steps', type=int, default=4_000_000)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--metrics', type=str, default='all')
    args = parser.parse_args()
    supported = {
        'returns',
        'grad_norms',
        'dormancy_actor_rnn',
        'dormancy_actor_mlp',
        'seq_grad_norms',
    }
    metrics_arg = args.metrics.strip()
    if metrics_arg == 'all':
        metrics = sorted(supported)
    else:
        if metrics_arg not in supported:
            raise ValueError(
                "Only metrics in {'returns','grad_norms','dormancy_actor_rnn','dormancy_actor_mlp','seq_grad_norms','all'} are supported right now."
            )
        metrics = [metrics_arg]
    for metric in metrics:
        main(args.dir, args.int_space, args.total_steps, args.exp_name, args.notes, metric)
