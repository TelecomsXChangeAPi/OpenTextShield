# Code Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L733
import math

import mlx.core as mx
import mlx.nn as nn

class BertEmbeddings(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
		
		self.position_ids = mx.expand_dims(mx.arange(0, config.max_position_embeddings), axis = 0)
		self.token_type_ids = mx.zeros((1, config.max_position_embeddings))
		
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

	def __call__(
		self,
		input_ids,
		token_type_ids,
		position_ids,
		inputs_embeds,
		past_key_values_length: int = 0,
	):
		if input_ids is not None:
			input_shape = input_ids.shape
		else:
			input_shape = inputs_embeds.shape[:-1]

		seq_length = input_shape[1]

		if position_ids is None:
			position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

		# Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
		# when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
		# issue #5664
		if token_type_ids is None:
			if hasattr(self, "token_type_ids"):
				buffered_token_type_ids = self.token_type_ids[:, :seq_length]
				buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
				token_type_ids = buffered_token_type_ids_expanded
			else:
				token_type_ids = mx.zeros(input_shape, dtype=mx.int64, device=self.position_ids.device)

		if inputs_embeds is None:
			inputs_embeds = self.word_embeddings(input_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		embeddings = inputs_embeds + token_type_embeddings
		if self.position_embedding_type == "absolute":
			position_embeddings = self.position_embeddings(position_ids)
			embeddings += position_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings

class BertSelfAttention(nn.Module):
	def __init__(self, config, position_embedding_type=None):
		super().__init__()
		if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
			raise ValueError(
				f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
				f"heads ({config.num_attention_heads})"
			)

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
		self.position_embedding_type = position_embedding_type or getattr(
			config, "position_embedding_type", "absolute"
		)
		if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
			self.max_position_embeddings = config.max_position_embeddings
			self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

		self.is_decoder = config.is_decoder

	def transpose_for_scores(self, x):
		new_x_shape = tuple(x.shape[:-1]) + (self.num_attention_heads, self.attention_head_size)
		x = mx.reshape(x, new_x_shape)
		return x.transpose([0, 2, 1, 3])

	def __call__(
		self,
		hidden_states,
		attention_mask = None,
		head_mask = None,
		encoder_hidden_states = None,
		encoder_attention_mask = None,
		past_key_value = None,
		output_attentions = False,
	):
		mixed_query_layer = self.query(hidden_states)

		# If this is instantiated as a cross-attention module, the keys
		# and values come from an encoder; the attention mask needs to be
		# such that the encoder's padding tokens are not attended to.
		is_cross_attention = encoder_hidden_states is not None

		if is_cross_attention and past_key_value is not None:
			# reuse k,v, cross_attentions
			key_layer = past_key_value[0]
			value_layer = past_key_value[1]
			attention_mask = encoder_attention_mask
		elif is_cross_attention:
			key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
			value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
			attention_mask = encoder_attention_mask
		elif past_key_value is not None:
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))
			key_layer = mx.concatenate([past_key_value[0], key_layer], axis=2)
			value_layer = mx.concatenate([past_key_value[1], value_layer], axis=2)
		else:
			key_layer = self.transpose_for_scores(self.key(hidden_states))
			value_layer = self.transpose_for_scores(self.value(hidden_states))

		query_layer = self.transpose_for_scores(mixed_query_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = mx.matmul(query_layer, key_layer.transpose([0, 1, -1, -2]))

		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		if attention_mask is not None:
			# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
			attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = mx.softmax(attention_scores, axis=-1)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		# Mask heads if we want to
		if head_mask is not None:
			attention_probs = attention_probs * head_mask

		context_layer = mx.matmul(attention_probs, value_layer)

		context_layer = context_layer.transpose([0, 2, 1, 3])
		new_context_layer_shape = tuple(context_layer.shape[:-2]) + (self.all_head_size,)
		context_layer = context_layer.reshape(new_context_layer_shape)

		outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
		return outputs

class BertSelfOutput(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def __call__(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states

class BertAttention(nn.Module):
	def __init__(self, config, position_embedding_type=None):
		super().__init__()
		self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
		self.output = BertSelfOutput(config)
		self.pruned_heads = set()

	def __call__(
		self,
		hidden_states,
		attention_mask = None,
		head_mask = None,
		encoder_hidden_states = None,
		encoder_attention_mask = None,
		past_key_value = None,
		output_attentions = False,
	):
		self_outputs = self.self(
			hidden_states,
			attention_mask,
			head_mask,
			encoder_hidden_states,
			encoder_attention_mask,
			past_key_value,
			output_attentions,
		)
		attention_output = self.output(self_outputs[0], hidden_states)
		outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
		return outputs

class BertIntermediate(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		self.intermediate_act_fn = nn.GELU()

	def __call__(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class BertOutput(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def __call__(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states

def apply_chunking_to_forward(
	forward_fn,
	chunk_size,
	chunk_dim,
	*input_tensors
):
	if chunk_size > 0:
		tensor_shape = input_tensors[0].shape[chunk_dim]
		for input_tensor in input_tensors:
			if input_tensor.shape[chunk_dim] != tensor_shape:
				raise ValueError(
					f"All input tenors have to be of the same shape: {tensor_shape}, "
					f"found shape {input_tensor.shape[chunk_dim]}"
				)

		if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
			raise ValueError(
				f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
				f"size {chunk_size}"
			)

		num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

		# chunk input tensor into tuples
		input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
		# apply forward fn to every tuple
		output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
		# concatenate output at same dimension
		return mx.concatenate(output_chunks, axis=chunk_dim)

	return forward_fn(*input_tensors)

class BertLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.chunk_size_feed_forward = config.chunk_size_feed_forward
		self.seq_len_dim = 1
		self.attention = BertAttention(config)
		self.is_decoder = config.is_decoder
		self.add_cross_attention = config.add_cross_attention
		if self.add_cross_attention:
			if not self.is_decoder:
				raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
			self.crossattention = BertAttention(config, position_embedding_type="absolute")
		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def __call__(
		self,
		hidden_states,
		attention_mask = None,
		head_mask = None,
		encoder_hidden_states = None,
		encoder_attention_mask = None,
		past_key_value = None,
		output_attentions = False,
	):
		# decoder uni-directional self-attention cached key/values tuple is at positions 1,2
		self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
		self_attention_outputs = self.attention(
			hidden_states,
			attention_mask,
			head_mask,
			output_attentions=output_attentions,
			past_key_value=self_attn_past_key_value,
		)
		attention_output = self_attention_outputs[0]
		outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

		cross_attn_present_key_value = None
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		outputs = (layer_output,) + outputs
		return outputs

	def feed_forward_chunk(self, attention_output):
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output

class BertEncoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.layer = [
			BertLayer(config) for _ in range(config.num_hidden_layers)
		]
		self.gradient_checkpointing = False
	
	def __call__(
		self,
		hidden_states,
		attention_mask,
		head_mask = None,
		encoder_hidden_states = None,
		encoder_attention_mask = None,
		past_key_values = None,
		output_attentions = False,
		output_hidden_states = False,
		return_dict = True,
		use_cache = False
	):
		all_hidden_states = () if output_hidden_states else None
		all_self_attentions = () if output_attentions else None
		all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

		next_decoder_cache = None
		for i, layer_module in enumerate(self.layer):
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			layer_head_mask = head_mask[i] if head_mask is not None else None
			past_key_value = past_key_values[i] if past_key_values is not None else None

			layer_outputs = layer_module(
				hidden_states,
				attention_mask,
				layer_head_mask,
				encoder_hidden_states,
				encoder_attention_mask,
				past_key_value,
				output_attentions,
			)

			hidden_states = layer_outputs[0]
			if output_attentions:
				all_self_attentions = all_self_attentions + (layer_outputs[1],)
				if self.config.add_cross_attention:
					all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		return tuple(
			v
			for v in [
				hidden_states,
				next_decoder_cache,
				all_hidden_states,
				all_self_attentions,
				all_cross_attentions,
			]
			if v is not None
		)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = mx.tanh

    def __call__(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MLXBertModel(nn.Module):
	def __init__(
		self, 
		config
	):
		super().__init__()
		self.config = config
		self.embeddings = BertEmbeddings(config)
		self.encoder = BertEncoder(config)

		self.pooler = BertPooler(config)

	def get_head_mask(
		self, 
		head_mask, 
		num_hidden_layers, 
		is_attention_chunked = False
	):
		"""
		Prepare the head mask if needed.
		"""
		if head_mask is not None:
			head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
			if is_attention_chunked is True:
				head_mask = head_mask.unsqueeze(-1)
		else:
			head_mask = [None] * num_hidden_layers

		return head_mask
	
	def __call__(
		self,
		input_ids,
		attention_mask = None,
		token_type_ids = None,
		position_ids = None,
		head_mask = None,
		inputs_embeds = None,
		encoder_hidden_states = None,
		encoder_attention_mask = None,
		past_key_values = None,
		use_cache = None,
		output_attentions = None,
		output_hidden_states = None,
		return_dict = None
	):
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		use_cache = False

		input_shape = input_ids.shape
		batch_size, seq_length = input_shape

		# past_key_values_length
		past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
		head_mask = [None] * self.config.num_hidden_layers

		embedding_output = self.embeddings(
			input_ids=input_ids,
			position_ids=position_ids,
			token_type_ids=token_type_ids,
			inputs_embeds=inputs_embeds,
			past_key_values_length=past_key_values_length,
		)
		encoder_outputs = self.encoder(
			embedding_output,
			attention_mask=attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=None,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

		return (sequence_output, pooled_output) + encoder_outputs[1:]