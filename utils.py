import torch

def format_aux(aux, decimals=1):
    if isinstance(aux, dict):
        summary = aux.get("summary")
        ponder_cost = aux.get("ponder_cost")
        if torch.is_tensor(ponder_cost):
            ponder_cost = ponder_cost.item()
        return {
            "summary": format_aux(summary, decimals),
            "ponder_cost": None if ponder_cost is None else f"{ponder_cost:.{decimals}f}",
        }
    if isinstance(aux, list):
        return [float(f"{v:.{decimals}f}") for v in aux]
    if torch.is_tensor(aux):
        return f"{aux.item():.{decimals}f}"
    if isinstance(aux, float):
        return f"{aux:.{decimals}f}"
    return aux