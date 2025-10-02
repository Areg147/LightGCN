import sys
sys.path.append("./")
from src.engine import InferenceEngine
from src.config import load_config
from src.utils import print_colored_table
import argparse
import torch
import numpy as np
# import scipy.sparse as sp ## REMOVED
from termcolor import colored
from scipy import sparse

def validate_arguments(args):
    assert not (args.user_cold_start_mode and args.user_ids), "Cannot provide user_ids in cold start mode."
    # assert not (args.user_cold_start_mode and args.positive_interactions), "Cannot provide user_ids_file in cold start mode."
    if args.user_cold_start_mode:
        assert args.positive_interactions and len(args.positive_interactions) > 0, \
            "In user_cold_start_mode, positive_interactions must be provided and non-empty."
        

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
    parser.add_argument("--model_config_pth", type=str, required=True)
    parser.add_argument("--model_inp_pth", type=str, required=True)
    parser.add_argument("--model_see_data_path", type=str, required=True)
    parser.add_argument("--user_cold_start_mode", action="store_true") 
    parser.add_argument("--user_ids", type=int, nargs='+', required=False) 
    parser.add_argument("--user_ids_file", type=str, required=False)
    parser.add_argument("--positive_interactions", type=int, nargs='+', required=False)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()
    validate_arguments(args)
    
    config = load_config(args.model_config_pth, **torch.load(args.model_inp_pth))
    engine = InferenceEngine(config)
    train_data = torch.load(args.model_see_data_path)
    items = train_data.shape[1]
    # Load user IDs
    if args.user_ids:
        user_ids = args.user_ids
    elif args.user_ids_file:
        with open(args.user_ids_file, "r") as f:
            user_ids = [int(line.strip()) for line in f if line.strip()]
    elif not args.user_cold_start_mode:
        raise ValueError("Either --user_ids or --user_ids_file must be provided.")
    else:
        user_ids = []


    if args.user_cold_start_mode:
        user_preference_vector = sparse.coo_matrix(
            (np.ones(len(args.positive_interactions), dtype=np.float64),
             (args.positive_interactions, [0] * len(args.positive_interactions))),
            shape=(items, 1)
        ).tocsr()
        # do train data and user_preference_vector dot product, in sparce mode
        scores = train_data @ user_preference_vector
        # sort scores
        sorted_indices = np.argsort(scores.toarray().flatten())[-3:]
        recomend(sorted_indices.tolist(), train_data, engine, args)
    else:
        recomend(user_ids, train_data, engine, args)
