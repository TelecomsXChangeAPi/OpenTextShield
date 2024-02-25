import time
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from utils.bert import MLXBertModel
from transformers import AutoTokenizer, BertConfig

pretrained_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

## LOAD MOEL
config = BertConfig.from_pretrained(pretrained_model_name)
start = time.time()
model = MLXBertModel(config)
print("Loaded MLXBert: {:.3f} sec".format(time.time()-start))

## Load Weight
loaded_weights = mx.load("converted_bert.npz")
start = time.time()
model.update(tree_unflatten(list(loaded_weights.items())))

model.eval()
mx.eval(model.parameters())
print("Loaded Weight: {:.3f} sec".format(time.time()-start))

## Inference Sample
encoded = tokenizer("hello", return_tensors="np")
print(encoded)

input_ids = mx.array(encoded["input_ids"])
token_type_ids = mx.array(encoded["token_type_ids"])
attention_mask = mx.array(encoded["attention_mask"])

## Inference Test
model_outputs = model(
	input_ids,
	attention_mask = attention_mask,
	token_type_ids = token_type_ids
)

sequence_output, pooled_output = model_outputs
print(sequence_output.shape)
print(pooled_output.shape)

print(sequence_output[0,0,:10])
print(pooled_output[:,0])