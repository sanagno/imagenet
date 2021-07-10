import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def dequantize_verts(verts, n_bits, add_noise=False):
    """Quantizes vertices and outputs integers with specified n_bits."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2 ** n_bits - 1
    if torch.is_tensor(verts):
        verts = verts.float()
    else:
        verts = verts.astype(float)
    verts = verts * (max_range - min_range) / range_quantize + min_range
    if add_noise:
        if torch.is_tensor(verts):
            verts = verts + torch.randn(verts.shape * (1 / float(range_quantize)))
        else:
            verts = verts + np.randn(verts.shape * (1 / float(range_quantize)))
    return verts


def quantize_vets(verts, n_bits):
    """Dequantizes integer vertices to floats."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2 ** n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (max_range - min_range)
    if torch.is_tensor(verts_quantize):
        return torch.round(verts_quantize).int()
    else:
        return np.around(verts_quantize).astype(int)


def top_k_logits(logits, k):
    """Masks logits such that logits not in top-k are small."""
    if k == 0:
        return logits
    else:
        values, _ = torch.topk(logits, k=k)
        k_largest = torch.min(values, axis=2)[0]
        logits[logits < torch.unsqueeze(k_largest, 2)] = -1e9
        return logits


def top_p_logits(logits, p):
    """Masks logits using nucleus (top-p) sampling."""
    if p == 1:
        return logits
    else:
        logit_shape = logits.shape
        seq, dim = logit_shape[1], logit_shape[2]
        logits = logits.view([-1, dim])
        sort_indices = torch.argsort(logits, axis=1, descending=True)
        probs = torch.gather(torch.nn.Softmax(dim=1)(logits), 1, sort_indices)
        cumprobs = torch.cumsum(
            probs,
            dim=-1,
        )
        # The top 1 candidate always will not be masked.
        # This way ensures at least 1 indices will be selected.
        sort_mask = cumprobs > p
        sort_mask[:, 0] = False
        top_p_mask = torch.zeros_like(sort_mask)
        for i in range(sort_mask.shape[0]):
            for j in range(sort_mask.shape[1]):
                top_p_mask[i, sort_indices[i, j]] = sort_mask[i, j]
        logits -= top_p_mask * 1e9
        return logits.view([-1, seq, dim])


def scaled_dot_product(q, k, v, bias=None, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if bias is not None:
        attn_logits = attn_logits + torch.squeeze(bias, 1)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


# TODO no DROPOUT HERE!!
class MultiheadAttention(nn.Module):
    def __init__(
        self, input_dim, embed_dim, num_heads, add_context=False, context_size=None
    ):
        super(MultiheadAttention, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."
        assert not add_context or context_size is not None

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.add_context = add_context
        self.context_size = context_size

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(input_dim, embed_dim)

        if self.add_context:
            self.k_proj = nn.Linear(context_size, embed_dim)
            self.v_proj = nn.Linear(context_size, embed_dim)
        else:
            self.k_proj = nn.Linear(input_dim, embed_dim)
            self.v_proj = nn.Linear(input_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.kaiming_normal_(self.q_proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.q_proj.bias, std=1e-6)
        nn.init.kaiming_normal_(self.k_proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.k_proj.bias, std=1e-6)
        nn.init.kaiming_normal_(self.v_proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.v_proj.bias, std=1e-6)
        nn.init.kaiming_normal_(self.o_proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.o_proj.bias, std=1e-6)

    def forward(self, x, context=None, bias=None, mask=None, return_attention=False):
        assert not self.add_context or context is not None
        batch_size, seq_length, embed_dim = x.size()
        context_seq_length = seq_length

        q = self.q_proj(x)

        if not self.add_context:
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            context_seq_length = context.shape[1]
            k = self.k_proj(context)
            v = self.v_proj(context)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        k = k.view(
            batch_size, context_seq_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = v.view(
            batch_size, context_seq_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, bias=bias, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class TransformerEncoder(nn.Module):
    """Transformer encoder.
    Sonnet Transformer encoder module as described in Vaswani et al. 2017. Uses
    the Tensor2Tensor multihead_attention function for full self attention
    (no masking). Layer norm is applied inside the residual path as in sparse
    transformers (Child 2019).
    This module expects inputs to be already embedded, and does not add position
    embeddings.
    """

    def __init__(
        self,
        hidden_size=256,
        fc_size=1024,
        num_heads=4,
        layer_norm=True,
        num_layers=8,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=False,
    ):
        """Initializes TransformerEncoder.
        Args:
          hidden_size: Size of embedding vectors.
          fc_size: Size of fully connected layer.
          num_heads: Number of attention heads.
          layer_norm: If True, apply layer normalization
          num_layers: Number of Transformer blocks, where each block contains a
            multi-head attention layer and a MLP.
          dropout_rate: Dropout rate applied immediately after the GELU in each
            fully-connected layer.
          re_zero: If True, alpha scale residuals with zero init.
          memory_efficient: If True, recompute gradients for memory savings.
          name: Name of variable scope
        """
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.fc_size = fc_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero

        self.dropout = nn.Dropout(self.dropout_rate)

        self.layer_norm_op = nn.LayerNorm(self.hidden_size)

        self.attention_layers = nn.ModuleList([])
        self.fc1s = nn.ModuleList([])
        self.fc2s = nn.ModuleList([])

        for layer_num in range(self.num_layers):
            self.attention_layers.append(
                MultiheadAttention(self.hidden_size, self.hidden_size, self.num_heads)
            )

            self.fc1s.append(nn.Linear(self.hidden_size, self.fc_size))
            self.fc2s.append(nn.Linear(self.fc_size, self.hidden_size))

        if self.re_zero:
            self.context_scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for fc in self.fc1s:
            nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
            nn.init.normal_(fc.bias, std=1e-6)

        for fc in self.fc2s:
            nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
            nn.init.normal_(fc.bias, std=1e-6)

    def forward(self, inputs):
        """Passes inputs through Transformer encoder network.
        Args:
          inputs: Tensor of shape [batch_size, sequence_length, embed_size]. Zero
            embeddings are masked in self-attention.
          is_training: If True, dropout is applied.
        Returns:
          output: Tensor of shape [batch_size, sequence_length, embed_size].
        """
        # Identify elements with all zeros as padding, and create bias to mask
        # out padding elements in self attention.

        encoder_padding = (inputs == 0).all(dim=-1).float()
        encoder_self_attention_bias = (
            (encoder_padding * (-1e9)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        )

        x = inputs
        for layer_num in range(self.num_layers):

            # Multihead self-attention from Tensor2Tensor.
            res = x
            if self.layer_norm:
                res = self.layer_norm_op(res)

            res = self.attention_layers[layer_num](
                res, bias=encoder_self_attention_bias
            )

            if self.re_zero:
                res = res * self.context_scale
            res = self.dropout(res)
            x = x + res

            # MLP
            res = x
            if self.layer_norm:
                res = self.layer_norm_op(res)

            res = self.fc1s[layer_num](res)
            res = nn.GELU()(res)
            res = self.fc2s[layer_num](res)
            if self.re_zero:
                res = res * self.context_scale
            res = self.dropout(res)
            x = x + res

        if self.layer_norm:
            output = self.layer_norm_op(x)
        else:
            output = x
        return output


class TransformerDecoder(nn.Module):
    """Transformer decoder.
    Sonnet Transformer decoder module as described in Vaswani et al. 2017. Uses
    the Tensor2Tensor multihead_attention function for masked self attention, and
    non-masked cross attention attention. Layer norm is applied inside the
    residual path as in sparse transformers (Child 2019).
    This module expects inputs to be already embedded, and does not
    add position embeddings.
    """

    def __init__(
        self,
        hidden_size=256,
        fc_size=1024,
        num_heads=4,
        layer_norm=True,
        num_layers=8,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=False,
        context_embedding_size=None,
    ):
        """Initializes TransformerDecoder.
        Args:
          hidden_size: Size of embedding vectors.
          fc_size: Size of fully connected layer.
          num_heads: Number of attention heads.
          layer_norm: If True, apply layer normalization. If mem_efficient_attention
            is True, then layer norm is always applied.
          num_layers: Number of Transformer blocks, where each block contains a
            multi-head attention layer and a MLP.
          dropout_rate: Dropout rate applied immediately after the GELU in each
            fully-connected layer.
          re_zero: If True, alpha scale residuals with zero init.
          memory_efficient: If True, recompute gradients for memory savings.
          name: Name of variable scope
        """
        super(TransformerDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.fc_size = fc_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.memory_efficient = memory_efficient

        self.dropout = nn.Dropout(self.dropout_rate)

        self.layer_norm_op = nn.LayerNorm(self.hidden_size)

        self.attention_layers = nn.ModuleList([])
        self.fc1s = nn.ModuleList([])
        self.fc2s = nn.ModuleList([])

        for layer_num in range(self.num_layers):
            self.attention_layers.append(
                MultiheadAttention(self.hidden_size, self.hidden_size, self.num_heads)
            )

            self.fc1s.append(nn.Linear(self.hidden_size, self.fc_size))
            self.fc2s.append(nn.Linear(self.fc_size, self.hidden_size))

        self.context_embedding = False
        if context_embedding_size is not None and context_embedding_size[0] is not None:

            # could attend to multiple sequences...
            self.num_contexs = len(context_embedding_size)

            self.context_embedding = True
            self.context_attention_layers = nn.ModuleList([])

            for layer_num in range(self.num_layers):
                for i in range(self.num_contexs):
                    self.context_attention_layers.append(
                        MultiheadAttention(
                            self.hidden_size,
                            self.hidden_size,
                            self.num_heads,
                            add_context=True,
                            context_size=context_embedding_size[i],
                        )
                    )

        if self.re_zero:
            self.context_scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for fc in self.fc1s:
            nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
            nn.init.normal_(fc.bias, std=1e-6)
        for fc in self.fc2s:
            nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
            nn.init.normal_(fc.bias, std=1e-6)

    def forward(self, inputs, sequential_context_embeddings=None, cache=None):
        """Passes inputs through Transformer decoder network.
        Args:
          inputs: Tensor of shape [batch_size, sequence_length, embed_size]. Zero
            embeddings are masked in self-attention.
          sequential_context_embeddings: Optional tensor with global context
            (e.g image embeddings) of shape
            [batch_size, context_seq_length, context_embed_size].
          cache: Optional dict containing tensors which are the results of previous
            attentions, used for fast decoding. Expects the dict to contain two
            keys ('k' and 'v'), for the initial call the values for these keys
            should be empty Tensors of the appropriate shape.
            'k' [batch_size, 0, key_channels] 'v' [batch_size, 0, value_channels]
        Returns:
          output: Tensor of shape [batch_size, sequence_length, embed_size].
        """

        # create bias to mask future elements for causal self-attention.
        seq_length = inputs.shape[1]
        decoder_self_attention_bias = torch.triu(
            torch.ones((1, 1, seq_length, seq_length), device=inputs.device) * (-1e9),
            diagonal=1,
        )

        # If using sequential_context, identify elements with all zeros as padding,
        # and create bias to mask out padding elements in self attention.
        if sequential_context_embeddings is not None:

            assert (
                len(sequential_context_embeddings) == self.num_contexs
            ), "Make sure the appropriate number of contexs is provided in the decoder"

            encoder_decoder_attention_biases = []

            for i in range(self.num_contexs):
                encoder_padding = (
                    (sequential_context_embeddings[i] == 0).all(dim=-1).float()
                )
                encoder_decoder_attention_bias = (
                    (encoder_padding * (-1e9)).unsqueeze(1).unsqueeze(1)
                )

                encoder_decoder_attention_biases.append(
                    torch.unsqueeze(encoder_decoder_attention_bias, 1)
                )

        x = inputs
        for layer_num in range(self.num_layers):

            # If using cached decoding, access cache for current layer, and create
            # bias that enables un-masked attention into the cache
            if cache is not None:
                raise NotImplementedError
                layer_decoder_bias = torch.zeros([1, 1, 1, 1])

            layer_decoder_bias = decoder_self_attention_bias

            # Multihead self-attention from Tensor2Tensor.
            res = x
            if self.layer_norm:
                res = self.layer_norm_op(res)

            res = self.attention_layers[layer_num](res, bias=layer_decoder_bias)

            if self.re_zero:
                res = res * self.context_scale

            res = self.dropout(res)
            x = x + res

            # Optional cross attention into sequential context
            if sequential_context_embeddings is not None:
                assert self.context_embedding
                for i in range(self.num_contexs):
                    res = x
                    if self.layer_norm:
                        res = self.layer_norm_op(res)
                    res = self.context_attention_layers[
                        layer_num * self.num_contexs + i
                    ](
                        res,
                        context=sequential_context_embeddings[i],
                        bias=encoder_decoder_attention_biases[i],
                    )

                    if self.re_zero:
                        res = res * self.context_scale
                    res = self.dropout(res)
                    x = x + res

            # FC layers
            res = x
            if self.layer_norm:
                res = self.layer_norm_op(res)

            res = self.fc1s[layer_num](res)
            res = nn.GELU()(res)
            res = self.fc2s[layer_num](res)
            if self.re_zero:
                res = res * self.context_scale
            res = self.dropout(res)
            x = x + res

        if self.layer_norm:
            output = self.layer_norm_op(x)
        else:
            output = x
        return output


class MyEmbed(nn.Module):
    def __init__(self, classes, embedding_size):
        super(MyEmbed, self).__init__()
        self.embedding = nn.Embedding(classes, embedding_size)
        nn.init.kaiming_normal_(
            self.embedding.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):
        return self.embedding(x)
