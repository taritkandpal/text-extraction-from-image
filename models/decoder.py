import math

import torch
import torch.nn as nn

import config
from utils import check_model_params_and_size


class PositionalEncoding(nn.Module):
    """
    Create positional encoding for the transformer for sequential tracking
    """

    # d_model : dimension of each positional encoding
    # max_len : maximum length of input sequences
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # initialize positional encodings as 0
        pe = torch.zeros(max_len, d_model)
        # tensor representing position indices of input sequence. Resized into [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # division term used in sine and cosine calculations of positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # sine function for pe is applied to even indices and cosine to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # resize pe to make it suitable for addition to the embeddings. Also, the input to transformer model is
        # [sequence_len, batch_size, features]. Thus, pe is reshaped to [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        # register pe as a buffer. Buffers in PyTorch are not considered during backprop (not considered as model params)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pe added to input tensor. Sliced to match the sequence length of x.
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Decoder(nn.Module):
    """
    Transformer Model for Text Generation
    """

    def __init__(self, vocab_len, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super(Decoder, self).__init__()

        # Create a default PyTorch transformer
        # nheads : number of heads in MHSA mechanism
        # hidden_dim : size of embeddings in the transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # Output positional encodings (object queries). Input is already a latent representation
        self.embedding = nn.Embedding(vocab_len, hidden_dim)  # embedding for each word/character in the vocabulary
        self.query_pos = PositionalEncoding(hidden_dim, .2)  # initialize positional encoding module

        # Spatial positional encodings needed apart from the usual one since OCR involves interpreting
        # images of texts. Thus, each characters row and column (spatial) position is important
        # There are learnable parameters, randomly initialized
        self.row_embed = nn.Parameter(torch.rand(config.ENCODER_OP_DIM, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(config.ENCODER_OP_DIM, hidden_dim // 2))
        # mask is used to prevent the model from seeing certain parts of the input data.
        # Used to prevent seeing the future tokens for seq2seq generation
        self.trg_mask = None

    """
    We need our model to predict the current character based on previous characters and not future ones.
    Even though seq2seq generation is not really the objective of OCR. It helps if the model
    understands the sequence of text for better accuracy and prediction. Mimic seq2seq

    mask = [[0, -inf, -inf], this means that the first word can see itself and nothing beyond it
            [0, 0, -inf], second word can see the first word and nothing beyond this
            [0, 0, 0]]

    This is called look-ahead mask or upper triangular mask
    """

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), 1)  # upper triangular matrix of size sz x sz
        mask = mask.masked_fill(mask == 1, float('-inf'))  # 1s converted to -inf so that softmax can make it 0
        return mask

    """
    This mask identifies where padding is present in the input sequence, since the input sequence
    may be padded to ensure uniform length. It is important to prevent the model from using
    padding as a part of the input sequence.
    """

    @staticmethod
    def make_len_mask(inp):
        # Returns True where input sequence is 0 (padded)
        return (inp == 0).transpose(0, 1)  # transpose: make the shape of mask as [seq_len, batch_size]

    def inference(self, h, trg):
        # construct positional encodings
        bs, _, H, W = h.shape
        pos = torch.cat([
            self.col_embed.unsqueeze(0).repeat(H, 1, 1),
            self.row_embed.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # Getting postional encoding for target
        trg = self.embedding(trg)
        trg = self.query_pos(trg)

        output = self.transformer(pos + h.flatten(2).permute(2, 0, 1), trg.permute(1, 0, 2))
        return output.transpose(0, 1)

    def forward(self, h, trg):
        # construct positional encodings
        bs, _, H, W = h.shape
        pos = torch.cat([
            self.col_embed.unsqueeze(0).repeat(H, 1, 1),
            self.row_embed.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # generating subsequent mask for target
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(trg.shape[1]).to(trg.device)

        # Padding mask
        trg_pad_mask = self.make_len_mask(trg)

        # Getting postional encoding for target
        trg = self.embedding(trg)
        trg = self.query_pos(trg)

        # 1st parameter -> adding spatial positional encoding for source and reshaping it to [height*width, batch_size,
        # channels] .permute changes the order of the axis or dimensions. 0.1 is multiplied to scale the feature map
        # 2nd parameter -> reshaping target sequence to [seq_length, batch_size, features]
        # 3rd parameter -> adding mask to target sequence
        # 4th parameter -> adding the padding mask
        output = self.transformer(pos + h.flatten(2).permute(2, 0, 1), trg.permute(1, 0, 2),
                                  tgt_mask=self.trg_mask,
                                  tgt_key_padding_mask=trg_pad_mask.permute(1, 0))

        return output.transpose(0, 1)


if __name__ == '__main__':
    model = Decoder(vocab_len=6, hidden_dim=512, nheads=8, num_encoder_layers=4, num_decoder_layers=4)
    h = torch.rand(1, 512, 16, 16)
    trg = torch.arange(1, 6).reshape(1, 5)
    tmp = torch.tensor([0, 0, 0]).reshape(1, 3)
    trg = torch.cat([trg, tmp], dim=-1)
    op = model(h, trg)
    print(f"Output Shape: {op.shape}")
    check_model_params_and_size(model)
