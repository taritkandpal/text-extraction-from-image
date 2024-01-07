import os
import random
import warnings

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_value_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
import numpy as np

import config
from models.data_processing import OcrDataset, string_metrics
from models.combined_model import TextExtractor
from utils import save_text, save_curve, save_checkpoint

warnings.filterwarnings("ignore")
torch.manual_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)
os.chdir(config.PROJECT_PATH)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print(f"Device: {device}")


def train_model(model, train_loader, optimizer, loss_func, scheduler):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, desc="Training")
    for x_img_batch, x_text_batch, y_batch in loop:
        # x_img_batch => batches of input images
        # x_text_batch => batches of label-encoded texts
        # y_batch => batches of one-hot-encoded texts
        x_img_batch = x_img_batch.to(device)
        x_text_batch = x_text_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_img_batch, x_text_batch[:, :-1])
        loss = loss_func(y_pred, y_batch[:, 1:, :])
        running_loss = running_loss + loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if config.GRAD_CLIP:
            clip_grad_value_(model.parameters(), config.GRAD_CLIP)
        loop.set_postfix(loss=loss.item())
    return running_loss / len(train_loader)


def eval_model(model, test_loader, loss_func, tok, epoch, generation):
    model.eval()
    bos_input = torch.tensor([[config.BOS_TOKEN]]).to(device)
    running_loss = 0
    running_cer = 0
    running_wer = 0
    for x_img_batch, x_text_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
        x_img_batch = x_img_batch.to(device)
        x_text_batch = x_text_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_img_batch, x_text_batch[:, :-1])
        loss = loss_func(y_pred, y_batch[:, 1:, :])
        running_loss = running_loss + loss.item()
        if generation:
            text_inference = model.inference(x_img_batch, bos_input)
            text_inference = tok.decode(text_inference[0])
            text_orig = tok.decode(x_text_batch[0])
            cer, wer = string_metrics(text_orig, text_inference)
            running_cer = running_cer + cer
            running_wer = running_wer + wer
            save_text(text_orig, text_inference, config.OP_PATH / f"{epoch}_eval.txt")

    n = len(test_loader)
    return running_loss / n, running_cer / n, running_wer / n


def main():
    # transforms and create dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                inplace=True,
            ),
        ]
    )
    dataset = OcrDataset(config.DATA_PATH, transform=transform, shuffle=True)

    # split into training and testing data
    train_size = int(config.TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # make dataloader objects for train and test data
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # initialize model
    model = TextExtractor(config.IN_CHANNELS, config.OUT_CHANNELS, config.WIDE_FEATURES, config.DEEP_FEATURES,
                          config.VOCAB_LEN, config.TR_HIDDEN_DIM, config.TR_NHEADS, config.TR_NUM_ENCODER_LAYERS,
                          config.TR_NUM_DECODER_LAYERS)
    model.to(device)

    # initialize optimizer
    optimizer = AdamW(
        model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.NUM_EPOCHS)

    # initialize loss
    loss_func = CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # train model
    train_losses, eval_losses, cers, wers = [], [], [], []
    for epoch in range(config.NUM_EPOCHS):
        print(f"Starting epoch {epoch + 1}.")
        train_loss = train_model(model, train_loader, optimizer, loss_func, scheduler)
        print(f"Epoch {epoch + 1} train loss: {train_loss}")
        train_losses.append(train_loss)
        generation = True if (epoch + 1) % config.NUM_VAL_EPOCHS == 0 else False
        eval_loss, cer, wer = eval_model(model, test_loader, loss_func, dataset.tok, epoch + 1, generation)
        print(f"Epoch {epoch + 1} eval loss: {eval_loss}")
        eval_losses.append(eval_loss)
        if generation:
            print(f"Epoch {epoch + 1} CER: {cer}")
            print(f"Epoch {epoch + 1} WER: {wer}")
            cers.append(cer)
            wers.append(wer)
            save_curve(cers, wers, "Character Error Rate", "Word Error Rate", "Error Rates",
                       config.OP_PATH / "error_rates.png")
        # save model
        save_checkpoint(model, optimizer, config.OP_PATH / "model.pth.tar")
        # show graphs
        save_curve(train_losses, eval_losses, "train loss", "eval loss", "Loss Curves",
                   config.OP_PATH / "loss_curve.png")


if __name__ == '__main__':
    main()
