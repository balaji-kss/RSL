import torch
import math
from torch import nn, Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 36):
        
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)  #(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) #(div_term) 

        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term) #(max_len, div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term) #(max_len, div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :] # (1, T, 50)

        return self.dropout(x)

class TransformerEncoder(nn.Module):
    
    def __init__(self, embed_dim=25*2, embed_proj_dim=None, ff_dim=256, num_heads=5, num_layers=6, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_proj_dim = embed_proj_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads 
        self.dropout = dropout
        self.activation = 'relu'
        self.num_layers = num_layers
     
        if self.embed_proj_dim:
            self.input_layer = nn.Linear(self.embed_dim, self.embed_proj_dim)
            self.output_layer = nn.Linear(self.embed_proj_dim, self.embed_dim)
        else:
            self.embed_proj_dim = self.embed_dim
            self.input_layer = None
            self.output_layer = None

        self.pos_encoder = PositionalEncoding(self.embed_proj_dim, dropout=self.dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_proj_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, activation=self.activation, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.print_params()
        
    def print_params(self):
        print('embed_dim: ', self.embed_dim)
        print('embed_proj_dim: ', self.embed_proj_dim)
        print('ff_dim: ', self.ff_dim)
        print('num_heads: ', self.num_heads)
        print('num_layers: ', self.num_layers)
        print('dropout: ', self.dropout)

    def forward(self, x, padding_masks):
        
        if self.input_layer:
            x = self.input_layer(x)

        pe_out = self.pos_encoder(x)

        tenc_out = self.transformer_encoder(pe_out, src_key_padding_mask=~padding_masks)
        tenc_out = self.act(tenc_out)
        tenc_out = self.dropout1(tenc_out)

        if self.output_layer:
            tenc_out = self.output_layer(tenc_out)

        return tenc_out


class TransformerDecoder(nn.Module):
    
    def __init__(self, embed_dim=25*2, embed_proj_dim=None, ff_dim=256, num_heads=5, num_layers=6, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_proj_dim = embed_proj_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads 
        self.dropout = dropout
        self.activation = 'relu'
        self.num_layers = num_layers
     
        if self.embed_proj_dim:
            self.input_layer = nn.Linear(self.embed_dim, self.embed_proj_dim)
            self.output_layer = nn.Linear(self.embed_proj_dim, self.embed_dim)
        else:
            self.embed_proj_dim = self.embed_dim
            self.input_layer = None
            self.output_layer = None

        self.pos_decoder = PositionalEncoding(self.embed_proj_dim, self.dropout)        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_proj_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, activation=self.activation, dropout=self.dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)

        self.print_params()
        
    def print_params(self):
        print('embed_dim: ', self.embed_dim)
        print('embed_proj_dim: ', self.embed_proj_dim)
        print('ff_dim: ', self.ff_dim)
        print('num_heads: ', self.num_heads)
        print('num_layers: ', self.num_layers)
        print('dropout: ', self.dropout)

    def forward(self, x, tgt_mask=None):
        
        if self.input_layer:
            x = self.input_layer(x)

        pd_out = self.pos_decoder(x)
        tdec_out = self.transformer_decoder(pd_out, pd_out, tgt_mask=tgt_mask)

        if self.output_layer:
            tdec_out = self.output_layer(tdec_out)

        return tdec_out

if __name__ == "__main__":

    transformer_encoder = TransformerEncoder(embed_dim=25*2, embed_proj_dim=None, ff_dim=2048, num_heads=5, num_layers=8, dropout=0.1)
    x = torch.rand(2, 20, 50)
    out = transformer_encoder(x)