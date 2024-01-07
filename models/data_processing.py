import os
import random
import string
import unicodedata
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import editdistance

from torch.utils.data import Dataset
import config


def string_metrics(string1, string2, lowercase=True, normalize=True):
    """
    Function for calculation of string metrics.
    Character Error Rate: Number of characters to edit to convert one string to another.
    Word Error Rate: Number of words to edit to convert one string to another.
    """
    # preprocessing
    if lowercase:
        string1 = string1.lower()
        string2 = string2.lower()
    if normalize:
        string1 = unicodedata.normalize("NFKD", string1).encode("ASCII", "ignore").decode("ASCII")
        string2 = unicodedata.normalize("NFKD", string2).encode("ASCII", "ignore").decode("ASCII")
    # character error rate
    c_dist = editdistance.eval(string1, string2)
    cer = c_dist / max(len(string1), len(string2))
    # word error rate
    w1 = string1.split()
    w2 = string2.split()
    w_dist = editdistance.eval(w1, w2)
    wer = w_dist / max(len(w1), len(w2))
    return cer, wer


class OcrDataset(Dataset):
    def __init__(self, data_path, transform=None, shuffle=True):
        if type(data_path) == str:
            data_path = Path(data_path)
        self.image_path = data_path / "images"
        self.text_path = data_path / "text"
        self.transform = transform
        self.fnames = [".".join(fname.split(".")[:-1]) for fname in os.listdir(self.image_path)]
        if shuffle:
            random.shuffle(self.fnames)
        self.tok = Tokenizer(max_text_length=config.MAX_TOKENS)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        im_path = self.image_path / (self.fnames[idx] + ".png")
        image = Image.open(im_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        txt_path = self.text_path / (self.fnames[idx] + ".txt")
        with open(txt_path, "r") as fh:
            text = fh.read()
        ohc, lec = self.tok.encode(text)
        return image, torch.from_numpy(lec), torch.from_numpy(ohc)


class Tokenizer:
    def __init__(self, max_text_length=1024):
        # Add padding token, token for unknown character, BOS and EOS token
        self.PAD_TK, self.UNK_TK, self.BOS, self.EOS = "¶", "¤", "BOS", "EOS"
        # Add these tokens to the character dictionary
        self.chars = [self.PAD_TK] + [self.UNK_TK] + [self.BOS] + [self.EOS] + list(string.printable[:-5])
        # Padding value is 0 since index of padding token is 0
        self.PAD = self.chars.index(self.PAD_TK)
        # Unknown value is 1 since index of unkown token is 1
        self.UNK = self.chars.index(self.UNK_TK)
        # vocab size and max token length
        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def one_hot_encode(self, encoded_text):
        num_classes = len(self.chars)  # Total number of unique characters/tokens
        # Create a matrix of zeros with shape [len(encoded_text), num_classes]
        one_hot = np.zeros((len(encoded_text), num_classes), dtype=np.float32)
        # Set the appropriate element to 1 for each one-hot vector
        for i, idx in enumerate(encoded_text):
            one_hot[i, idx] = 1
        return one_hot

    def encode(self, text):
        """Encode text to vector"""
        # Unicode Normalization is done to standardize different forms of the same character.
        # Normalized text is encoded to ASCII while ignoring all non-ASCII characters
        # and then decoded back to string.
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        # Text is broken down into wrods and then joined back with a single whitespace.
        # This is done to remove extra whitespace like new lines or tabs
        text = " ".join(text.split())
        # to store sequences of numerical indices representing the text - label encoding
        encoded = []
        # add BOS and EOS token at start and end of text
        text = ['BOS'] + list(text) + ['EOS']
        # For each character in text, it finds it's corresponding index in self.chars and
        # appends it to the encoded list. If index is not found, then it appends the index
        # of UNK (unknown character)
        for item in text:
            index = self.chars.index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)
        encoded = np.asarray(encoded, dtype=np.int64)
        encoded = np.pad(encoded, (0, config.MAX_TOKENS - len(encoded)))
        # Convert to one-hot encoded vectors
        encoded_one_hot = self.one_hot_encode(encoded)
        return encoded_one_hot, encoded

    def decode(self, text):
        """Decode vector to text"""
        # Decode text by taking index numbers and converting them into characters
        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        # To remove any special tokens defined during the encoding process
        decoded = self.remove_tokens(decoded)
        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD and UNK) from text"""
        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "").replace("BOS", "").replace("EOS", "")


if __name__ == '__main__':
    dataset = OcrDataset(r"D:\Projects\text-extraction-from-image\data\test")
    print(len(dataset))
