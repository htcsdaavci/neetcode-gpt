import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        T = 0
        full_list = positive + negative
        voc = sorted({word for sentence in full_list for word in sentence.split()})
        words_to_ids = {word : idx + 1 for idx, word in enumerate(voc)}
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        encodings = [torch.tensor([words_to_ids[word] for word in s.split()]) for s in full_list]
        
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return nn.utils.rnn.pad_sequence(encodings, batch_first=True)