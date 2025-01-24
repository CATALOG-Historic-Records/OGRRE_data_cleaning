import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    
    def __init__(
            self, in_features, embed_dim, ff_dim, num_heads, dropout, norm_eps, device):
        super().__init__()

        self.device = device
        self.num_heads = num_heads

        # Also known as key dimension, in Attention is all you need paper they
        # refer to this as d_key = d_value = d_model / num heads
        self.head_dim = embed_dim//num_heads

        # We can compute query, key and value in single batch mat multiplication
        self.qkv = nn.Linear(in_features, embed_dim*3)

        # The multi-headed attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            batch_first=True
            )

        # Dropout and linear layers after attention
        self.drop_out_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps)
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim),
            nn.ReLU()
        )
        self.drop_out_2 = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """
        Arguments:
            x (torch.Tensor) : The attention layer input, should have dimensions
                of (batch x sequence x embedding)
            padding_mask (torch.Tensor) : The padding mask, should have
                dimensions of (batch x sequence)

        Returns:
            block_out (torch.Tensor) : The Transformer (encoder in this case) 
                layer output.
        """

        # The linear layer returns (batch, sequence, 3*embedding)
        # 3*embedding= 3*key_dim*num_heads
        qkv = self.qkv(x)

        # Now separate out q, k and v from the 3*embedding dimension each of 
        # these 3 tensors final shape should be (batch size, sequence length, 
        # embedding dim) the final dimension has many names, in Attention is all 
        # you need they just call it the d_key = d_value = d_model/num heads, 
        # but in torch all the heads are in one dimension
        q, k, v = qkv.chunk(3, -1)

        # Mask such that words cannot place attention on words that occur after
        # they do in the sequence, the mask should be a (sequence x sequence)
        # matrix
        # mask = nn.Transformer.generate_square_subsequent_mask(q.shape[-1])
        # mask = nn.Transformer.generate_square_subsequent_mask(q.shape[-2], self.device)

        # Calculate attention
        attention, _ = self.attention(
            q, k, v,
            need_weights=False,
            key_padding_mask=padding_mask,
            is_causal=False
            )
        
        # Apply dropout to attention
        attention_out = self.drop_out_1(attention)
        # Add and normalize per Attention is all you need
        attention_out = self.layer_norm_1(x + attention_out)

        # Pass attention with residual to linear stack
        feed_forward_out = self.linear_stack(attention_out)
        feed_forward_out = self.drop_out_2(feed_forward_out)

        # Add and Normalize
        block_out = self.layer_norm_2(attention_out + feed_forward_out)
        
        return block_out


class Encoder(nn.Module):
    def __init__(
            self, vocab_dim, sequence_dim, embed_dim, ff_dim, 
            num_heads, num_blocks, dropout, norm_eps, device
            ) -> None:
        super().__init__()
        """
        Arguments:
            vocab_dim (int) : The number of unique tokens.
            sequence_dim (int) : The max sequence length.
            embed_dim (int) : The model dimension (number of features per token)
            ff_dim (int) : The number of features between the dense feed forward
                layers. Specifically the output dimension of the first FF layer
                and the input dimension of the second FF layer.
            num_heads (int) : The number of attention heads.
            num_blocks (int) : The number of transformer blocks.
            dropout (float) : The dropout rate inside the transformer block,
                see TransformerBlock.
            norm_eps (float) : Added to denominator of layer norm for numerical 
                stability
            device (str) : The device to store the model ("cpu" or "cuda").
        """

        # The position input
        self.position = torch.arange(sequence_dim, device=device)
        
        self.word_embedding = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(num_embeddings=sequence_dim, embedding_dim=embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(
                in_features=embed_dim, 
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                num_heads=num_heads,
                dropout=dropout,
                norm_eps=norm_eps,
                device=device
                ) for _ in range(num_blocks)]
            )
        
        # Transform from (batch, sequence, embedding) to (batch, sequence x 
        # embedding) since we want to use the entire sequence to produce the
        # final logits. But better to do this outside of the Encoder
        # self.linear = nn.Linear(in_features=embed_dim, out_features=vocab_dim)

        # torch cross entropy calculates the softmax internally, we only need
        # to use softmax during inference time
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        # Set padding mask to a bool tensor, True where padding token exists
        padding_mask = x==0
        x = self.word_embedding(x) + self.position_embedding(self.position)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        # Transform from (batch, sequence, embedding) to (batch, sequence x 
        # embedding) since we want to use the entire sequence to produce the
        # final logits. But better to do this outside of the Encoder
        # x = self.linear(x)
        return x

class Linear(nn.Module):
    
    def __init__(self, in_features, num_classes) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features, out_features=num_classes
            )
    
    def forward(self, x):
        return self.linear(x)

class Classifier(nn.Module):
    
    def __init__(self, encoder, sequence_dim, 
                 embedding_dim, num_classes) -> None:
        super().__init__()
        self.encoder = encoder
        self.output_layer_features = sequence_dim*embedding_dim
        self.output_layer = Linear(
            in_features=self.output_layer_features, num_classes=num_classes
            )
    
    def forward(self, x):
        x = self.encoder(x)
        # Flatten inner dimensions (sequence and embedding dimensions)
        x = torch.flatten(x, start_dim=1)
        logits = self.output_layer(x)
        return logits
        