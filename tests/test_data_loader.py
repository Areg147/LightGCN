import unittest
from tqdm import tqdm
import numpy as np
from typing import Callable, Dict, List, Tuple
import sys
sys.path.append("./")
from src.data_loader import uniform_sampler

def create_data(users_count: int,
                items_count: int) -> Tuple[Dict[int, List[int]], int]:
    data_size = 0
    user_pos_items_dict_train = {}

    for user in range(users_count):
        n_items = np.random.randint(2, 100)
        data_size += n_items
        user_pos_items_dict_train[user] = np.random.choice(items_count, n_items, replace=False).tolist()

        all_items = set(np.arange(0, items_count).tolist())

        posible_neg = set(all_items).difference(user_pos_items_dict_train[user])
        posible_neg = list(posible_neg)

    return user_pos_items_dict_train, data_size

class TestSampler(unittest.TestCase):

    def validate_uniform_sample_output(self,
                               sampler: Callable,
                               user_pos_items_dict_train: Dict[int, List[int]],
                               data_size: int,
                               users_count: int,
                               items_count: int,
                               neg_per_pos: int = 1):
        
        result = sampler(
            data_size=data_size,
            users_count=users_count,
            user_pos_items_dict_train=user_pos_items_dict_train,
            items_count=items_count,
            neg_per_pos=neg_per_pos
        )

        self.assertEqual(result.shape[0], data_size, f"Expected {data_size} rows, got {result.shape[0]}")
        self.assertEqual(result.shape[1], neg_per_pos+2, f"Expected {neg_per_pos+2} columns, got {result.shape[1]}")

        for i in result:
            user, pos_item, neg_items = i[0], i[1], i[2:]
            for neg_item in neg_items:
                self.assertIn(pos_item, user_pos_items_dict_train[user], f"Positive item {pos_item} not associated with user {user}")
                self.assertNotIn(neg_item, user_pos_items_dict_train[user], f"Negative item {neg_item} is actually a positive for user {user}")
                self.assertNotEqual(pos_item, neg_item, f"Positive and negative items are the same:for user {user}")
                self.assertLess(user, users_count, f"User index {user} out of bounds")
                self.assertLess(pos_item, items_count, f"Positive item index {pos_item} out of bounds")
                self.assertLess(neg_item, items_count, f"Negative item index {neg_item} out of bounds")


    def test_uniform_sampler(self):
        users_count = 300
        items_count = 500
        neg_per_poss = [1,2,3,5]
        n_trials = 20
        for neg_per_pos in neg_per_poss:
            for _ in tqdm(range(n_trials),desc=f"Testing uniform and explicit samplers {neg_per_pos} neg per pos"):
                pos, data_size = create_data(users_count, items_count)
                self.validate_uniform_sample_output(uniform_sampler, pos, data_size, users_count, items_count,neg_per_pos)

if __name__ == "__main__":
    unittest.main()