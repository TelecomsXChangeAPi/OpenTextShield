from utils.bert import MLXBertModel
from transformers import BertModel, AutoTokenizer
from mlx.utils import tree_unflatten
import mlx.core as mx
import mlx.nn as nn
import numpy as np

og_model = BertModel.from_pretrained("bert-base-uncased") 
og_model.eval()

config = og_model.config
og_state = og_model.state_dict()

converted_weights = {k: mx.array(v.numpy()) for k, v in og_state.items() if k is not None}
print(converted_weights.keys())

np.savez("converted_bert.npz", **converted_weights)
