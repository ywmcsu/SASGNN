import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.attn import ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class adj_emb(nn.Module):
    def __init__(self, enc_in,
                factor=7, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(adj_emb, self).__init__()
        self.output_attention = output_attention
        self.test_projection = nn.Linear(512, 12, bias=True)
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        Attn = ProbAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
    def forward(self, x):
            enc_out = self.enc_embedding(x)
            enc_out = self.encoder(enc_out)
            return enc_out