import torch
from termcolor import colored

def ndcg_recall_precision_batch(user_preds:torch.Tensor,
                                user_gts:torch.Tensor,
                                top_ks:list[int]) -> dict[str, torch.Tensor]:
    metric_values = {}

    device = user_preds.device
    batch_size = user_preds.size(0)

    for N in top_ks:
        user_preds_k = user_preds[:, :N]
        preds_exp = user_preds_k.unsqueeze(2)           
        gts_exp = user_gts.unsqueeze(1)                  
        relevance = (preds_exp == gts_exp).any(dim=2).float()
        
        # recall, precision
        gt_counts = (user_gts != -1).sum(dim=1)
        recall = torch.where(gt_counts > 0, relevance.sum(dim=1) / gt_counts, torch.zeros_like(gt_counts))
        precision =  relevance.sum(dim=1)/N

        positions = torch.arange(1, N + 1, device=device).float() 
        discount = 1.0 / torch.log2(positions + 1)
        dcg = (relevance * discount).sum(dim=1)
        gt_counts = (user_gts != -1).sum(dim=1).clamp(max=N)      
        ideal_discount = discount.unsqueeze(0).expand(batch_size, -1)   
        ideal_mask = torch.arange(N, device=device).unsqueeze(0) < gt_counts.unsqueeze(1)  
        idcg = (ideal_discount * ideal_mask).sum(dim=1)   
        # NDCG             
        ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))

        # store metrics at top N
        metric_values[f"ndcg_at_{N}"] = ndcg.mean()
        metric_values[f"recall_at_{N}"] = recall.mean()
        metric_values[f"precision_at_{N}"] = precision.mean()

    return metric_values

# prety print util
def print_colored_table(data, headers, colors):
    if isinstance(data, dict):
        rows = [
            (k, *v) if isinstance(v, tuple) else (k, v)
            for k, v in data.items()
        ]
    else:
        rows = data
    str_rows = [[str(v) for v in row] for row in rows]
    widths = []
    for j, h in enumerate(headers):
        col_width = max(len(h), max(len(r[j]) for r in str_rows))
        widths.append(col_width)
    header = '\t'.join(
        colored(h.ljust(widths[i]), colors[i], attrs=['bold'])
        for i, h in enumerate(headers)
    )
    print(header)
    for row in str_rows:
        line = '\t'.join(
            colored(v.ljust(widths[i]), colors[i])
            for i, v in enumerate(row)
        )
        print(line)