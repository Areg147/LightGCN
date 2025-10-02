import sys
sys.path.append("./")

from src.engine import InferenceEngine
from src.config import load_config
from src.utils import print_colored_table
import argparse
import torch
import numpy as np
from scipy import sparse

def validate_arguments(args):
    assert not (args.user_cold_start_mode and args.user_ids), "Cannot provide user_ids in cold start mode."
    if args.user_cold_start_mode:
        assert len(args.positive_interactions) > 0, \
            "In user_cold_start_mode, positive_interactions must be provided"
    else:
        if args.positive_interactions:
            raise ValueError("positive_interactions should not be provided in cold start mode.")

def recomend(user_ids, train_data, engine, args):

    scores, recommendations = engine.recommend(user_ids, k=args.top_k + 100)
    recommendations = np.array(recommendations)
    filtered_recommendations = []
    for i, user_id in enumerate(user_ids):
        seen = train_data[user_id].indices
        mask = ~np.isin(recommendations[i], seen)
        filtered = recommendations[i][mask][:args.top_k]
        filtered_recommendations.append(filtered.tolist())

    users_pred = {uid: (pred, train_data[uid].indices) for uid, pred in zip(user_ids, filtered_recommendations)}
    print_colored_table(
        users_pred,
        headers=["User ID", f"Top-{args.top_k} Recommendations", "GT"],
        colors=["cyan", "green", "blue"]
    )
    return users_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--user_cold_start_mode", action="store_true") 
    parser.add_argument("--user_ids", type=int, nargs='+', required=False) 
    parser.add_argument("--user_ids_file", type=str, required=False)
    parser.add_argument("--positive_interactions", type=int, nargs='+', required=False)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dump_predictions", action="store_true")

    args = parser.parse_args()
    validate_arguments(args)
    config_dir = f"src/models_configs/{args.model_name}_config.yaml"
    model_input_dir = "data/dump/{}"
    if args.model_name in ["lightgcn", "mgdcf"]:
        model_inp_pth = model_input_dir.format("BipartiteGraphDataset_model_input.pt")
    elif args.model_name == "kgin":
        model_inp_pth = model_input_dir.format("KnowledgeGraphDataset_model_input.pt")
    else:
        raise ValueError(f"Model {args.model_name} not supported.")
    
    model_see_data_path = model_input_dir.format(f"{args.dataset_name}.pt")
    
    config = load_config(config_dir, **torch.load(model_inp_pth))
    engine = InferenceEngine(config)
    train_data = torch.load(model_see_data_path)
    items = train_data.shape[1]
    # Load user IDs
    if args.user_ids:
        user_ids = args.user_ids
    elif args.user_ids_file:
        with open(args.user_ids_file, "r") as f:
            user_ids = [int(line.strip()) for line in f if line.strip()]


    if args.user_cold_start_mode:
        user_preference_vector = sparse.coo_matrix(
            (np.ones(len(args.positive_interactions), dtype=np.float64),
             (args.positive_interactions, [0] * len(args.positive_interactions))),
            shape=(items, 1)
        ).tocsr()
        # do train data and user_preference_vector dot product, in sparce mode
        scores = train_data @ user_preference_vector
        scores = scores / train_data.sum(axis=1)  # normalize by user activity
        sorted_indices = np.argsort(scores.toarray().flatten())[-1:]
        recs = recomend(sorted_indices.tolist(), train_data, engine, args)

    else:
        recs = recomend(user_ids, train_data, engine, args)
    if args.dump_predictions:
        if args.user_cold_start_mode:
            key = tuple(sorted(args.positive_interactions)+ [-1])
        else:
            key = tuple(sorted(recs.keys()))
        hash_val = str(abs(hash(key)))
        out_path = f"data/{args.model_name}_{args.dataset_name}_{hash_val}.txt"
        with open(out_path, "w") as f:
            for uid, (pred, gt) in recs.items():
                pred_str = ", ".join(map(str, pred))
                gt_str = ", ".join(map(str, gt))
                if args.user_cold_start_mode:
                    user_scores = scores.toarray().ravel()
                    f.write(f"{uid}: {pred_str} | GT: {gt_str} | Cold-start matched: {user_scores[uid]}\n")
                else:
                    f.write(f"{uid}: {pred_str} | GT: {gt_str}\n")
