import torch
import torch.nn as nn

import config
from models.encoder import resnet50Encoder as combinedEncoder
from models.decoder import Decoder
from utils import check_model_params_and_size


class TextExtractor(nn.Module):
    """
    Transformer Model for Text Generation
    """

    def __init__(self, wd_in_channels, wd_out_channels, wd_wide_features, wd_deep_features, vocab_len, tr_hidden_dim,
                 tr_nheads, tr_num_encoder_layers, tr_num_decoder_layers):
        super(TextExtractor, self).__init__()
        self.encoder = combinedEncoder(wd_in_channels, wd_out_channels, wd_wide_features, wd_deep_features)
        self.decoder = Decoder(vocab_len, tr_hidden_dim, tr_nheads, tr_num_encoder_layers, tr_num_decoder_layers)
        # Prediction heads with length of vocab
        self.prediction_head = nn.Linear(tr_hidden_dim, vocab_len)

    def inference(self, img, txt):
        h = self.encoder(img)
        while txt.shape[-1] < config.MAX_TOKENS and txt[0][-1] != 3:
            letter = self.decoder.inference(h, txt)
            output = self.prediction_head(letter)
            amax = torch.argmax(output, dim=-1)[0][-1].reshape(1, 1)
            txt = torch.cat([txt, amax], dim=-1)
        return txt

    def forward(self, img, trg):
        h = self.encoder(img)
        letter = self.decoder(h, trg)
        output = self.prediction_head(letter)
        return output


if __name__ == '__main__':
    model = TextExtractor(wd_in_channels=1, wd_out_channels=256, wd_wide_features=32, wd_deep_features=8, vocab_len=6,
                          tr_hidden_dim=512, tr_nheads=8, tr_num_encoder_layers=4, tr_num_decoder_layers=4)
    ip = torch.rand(1, 3, 512, 512)
    trg = torch.arange(1, 6).reshape(1, 5)
    tmp = torch.tensor([0, 0, 0]).reshape(1, 3)
    trg = torch.cat([trg, tmp], dim=-1)
    op = model(ip, trg)
    inf = model.inference(ip, trg)
    print(f"Output Shape: {op.shape}")
    print(f"Inference Output Shape: {inf.shape}")
    check_model_params_and_size(model)
