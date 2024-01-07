import torch
import matplotlib.pyplot as plt


def check_model_params_and_size(model):
    print(f"Model Params: {sum(p.numel() for p in model.parameters())}")
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))


def save_checkpoint(model, optimizer, filename):
    """
    Function to save checkpoint.
    """
    print("**SAVING CHECKPOINT**")
    checkpoint = {
        "model": model.state_dict(),  # saves the models current weights and biases
        "optimizer": optimizer.state_dict(),  # saves the optimizers state and hyperparameters
    }
    torch.save(checkpoint, filename)


def save_curve(set1, set2, label1, label2, title, fpath):
    ax = plt.axes()
    if len(set1) != 0:
        (p1,) = ax.plot(list(range(len(set1))), set1)
        p1.set_label(label1)
    if len(set2) != 0:
        (p2,) = ax.plot(list(range(len(set2))), set2, color="r")
        p2.set_label(label2)
    plt.legend()
    plt.title(title)
    plt.savefig(fpath)
    plt.cla()
    plt.clf()


def save_text(string1, string2, fpath):
    with open(fpath, "a") as fh:
        fh.write("\n".join([string1, string2, "\n"]))
