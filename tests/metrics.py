import torch
import numpy as np
import tqdm
import sys
sys.path.append("./")
import unittest
from src.utils import ndcg_recall_precision_batch

ITEMS_COUNT = 5000

def generate_data(items_count, batch_size, top_k, max_interaction_count):
    # Generate ground truth with unique items per row
    test_gt = torch.stack([torch.randperm(items_count)[:max_interaction_count] for _ in range(batch_size)])
    
    # Generate predictions with unique items per row
    test_pred = torch.stack([torch.randperm(items_count)[:top_k] for _ in range(batch_size)])
    
    # Pad each row differently - ensure at least one non-negative element per row
    for i in range(batch_size):
        # Random padding size (at least 1 less than max_interaction_count to keep 1+ real elements)
        pad_size = torch.randint(0, max_interaction_count - 2, (1,)).item()
        if pad_size > 0:
            test_gt[i, -pad_size:] = -1
    
    return test_gt, test_pred

def calcualte_recall(test_gt:torch.Tensor,
                     test_pred:torch.Tensor):
    numerators = torch.tensor([len(np.intersect1d(test_gt[i], test_pred[i])) for i in range(len(test_gt))])
    denominator = (test_gt !=-1).sum(axis=1)
    recall = numerators / denominator
    return recall.mean()

def calcualte_precision(test_gt:torch.Tensor,
                        test_pred:torch.Tensor,
                        top_k:int):

    numerators = torch.tensor([len(np.intersect1d(test_gt[i], test_pred[i])) for i in range(len(test_gt))])
    numerators = numerators/ top_k
    return numerators.mean()

def calcualte_ndcg(test_gt:torch.Tensor,
                   test_pred:torch.Tensor,
                   tok_k:int):

    k = []
    for test_1_gt_,test_1_pred_ in zip(test_gt, test_pred):
        dcg = 0.0
        for i, pred in enumerate(test_1_pred_, start=1):
            if pred in test_1_gt_:
                dcg += 1.0 / torch.log2(torch.tensor(i + 1.0)).item()
        ideal_hits = min(len(test_1_gt_[test_1_gt_!=-1]), tok_k)
        idcg = sum(1.0 / torch.log2(torch.tensor(i + 1.0)).item() for i in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        k.append(ndcg)
    ndcg_test = np.mean(k)
    return ndcg_test

class TestPRMetrics(unittest.TestCase):
    top_ks = [5, 10, 20, 50]
    batch_sizes = [32, 64, 128]
    max_interaction_counts = [20, 50, 100]

    def test_for_metrics(self):
        torch.manual_seed(42)
        np.random.seed(42)

        for top_k in self.top_ks:
            for batch_size in self.batch_sizes:
                for max_interaction_count in self.max_interaction_counts:
                    test_gt, test_pred = generate_data(ITEMS_COUNT, batch_size, top_k, max_interaction_count)

                    recall_ref = calcualte_recall(test_gt, test_pred)
                    precision_ref = calcualte_precision(test_gt, test_pred, top_k)
                    ndcg_ref = calcualte_ndcg(test_gt, test_pred, top_k)

                    metrics = ndcg_recall_precision_batch(test_pred, test_gt, [top_k])
                    ndcg_batch = metrics[f"ndcg_at_{top_k}"]
                    recall_batch = metrics[f"recall_at_{top_k}"]
                    precision_batch = metrics[f"precision_at_{top_k}"]

                    self.assertAlmostEqual(recall_ref.item(), recall_batch.item(), places=6)
                    self.assertAlmostEqual(precision_ref.item(), precision_batch.item(), places=6)
                    self.assertAlmostEqual(ndcg_ref, ndcg_batch.item(), places=6)

class TestPRMetrics(unittest.TestCase):
    top_ks = [5, 10, 20, 50]
    batch_sizes = [32, 64, 128]
    max_interaction_counts = [20, 50, 100]

    def test_for_metrics(self):
        torch.manual_seed(42)
        np.random.seed(42)

        for top_k in self.top_ks:
            for batch_size in self.batch_sizes:
                for max_interaction_count in self.max_interaction_counts:
                    test_gt, test_pred = generate_data(ITEMS_COUNT, batch_size, top_k, max_interaction_count)

                    recall_ref = calcualte_recall(test_gt, test_pred)
                    precision_ref = calcualte_precision(test_gt, test_pred, top_k)
                    ndcg_ref = calcualte_ndcg(test_gt, test_pred, top_k)

                    metrics = ndcg_recall_precision_batch(test_pred, test_gt, [top_k])
                    ndcg_batch = metrics[f"ndcg_at_{top_k}"]
                    recall_batch = metrics[f"recall_at_{top_k}"]
                    precision_batch = metrics[f"precision_at_{top_k}"]

                    self.assertAlmostEqual(recall_ref.item(), recall_batch.item(), places=6)
                    self.assertAlmostEqual(precision_ref.item(), precision_batch.item(), places=6)
                    self.assertAlmostEqual(ndcg_ref, ndcg_batch.item(), places=6)

    #  Edge case: empty ground truth
    def test_empty_gt(self):
        test_gt = torch.full((2, 5), -1)  # all padded
        test_pred = torch.randint(0, ITEMS_COUNT, (2, 5))
        metrics = ndcg_recall_precision_batch(test_pred, test_gt, [5])
        self.assertEqual(metrics["recall_at_5"].item(), 0.0)
        self.assertEqual(metrics["precision_at_5"].item(), 0.0)
        self.assertEqual(metrics["ndcg_at_5"].item(), 0.0)

    #  Edge case: empty predictions
    def test_empty_preds(self):
        test_gt = torch.tensor([[1, 2, 3], [4, 5, -1]])
        test_pred = torch.empty((2, 0), dtype=torch.long)
        metrics = ndcg_recall_precision_batch(test_pred, test_gt, [0])
        self.assertEqual(metrics["recall_at_0"].item(), 0.0)
        self.assertEqual(metrics["ndcg_at_0"].item(), 0.0)

    #  Edge case: perfect prediction
    def test_perfect_prediction(self):
        test_gt = torch.tensor([[1, 2, 3], [4, 5, 6]])
        test_pred = torch.tensor([[1, 2, 3], [4, 5, 6]])
        metrics = ndcg_recall_precision_batch(test_pred, test_gt, [3])
        self.assertAlmostEqual(metrics["recall_at_3"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["precision_at_3"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["ndcg_at_3"].item(), 1.0, places=6)

    #  Edge case: no overlap
    def test_no_overlap(self):
        test_gt = torch.tensor([[1, 2, 3], [4, 5, -1]])
        test_pred = torch.tensor([[10, 11, 12], [20, 21, 22]])
        metrics = ndcg_recall_precision_batch(test_pred, test_gt, [3])
        self.assertEqual(metrics["recall_at_3"].item(), 0.0)
        self.assertEqual(metrics["precision_at_3"].item(), 0.0)
        self.assertEqual(metrics["ndcg_at_3"].item(), 0.0)

    # #  Edge case: more predictions than ground truth
    def test_gt_smaller_than_topk(self):
        test_gt = torch.tensor([[1, 2, -1]])  # only 2 positives
        test_pred = torch.tensor([[1, 2, 3, 4, 5]])
        metrics = ndcg_recall_precision_batch(test_pred, test_gt, [5])
        self.assertLessEqual(metrics["recall_at_5"].item(),1.0)  
        self.assertLess(metrics["precision_at_5"].item(), 1.0)             
        self.assertLessEqual(metrics["ndcg_at_5"].item(), 1.0)                   


if __name__ == "__main__":
    unittest.main()