import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def sample_batch(dataloader, device="cpu"):
    """
    Get one batch from a DataLoader and move to device.
    """
    for batch in dataloader:
        return batch.to(device)
    raise ValueError("Dataloader returned no batches.")


def run_inference(model, batch):
    """
    Run model inference on a batch.

    Args:
        model (nn.Module): PyTorch model
        batch (torch.Tensor): Input batch [B, C, H, W]

    Returns:
        torch.Tensor: Model outputs [B, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    return outputs


def show_input_output_pairs(
    model: torch.nn.Module, dataloader: DataLoader, nrows=3, ncols=3
):
    # 1. Get batch
    batch = sample_batch(dataloader, device="cpu")

    # 2. Run model
    outputs = run_inference(model, batch)

    # 3. Visualize
    _show_input_output_pairs(batch, outputs, nrows=nrows, ncols=ncols)


def _show_input_output_pairs(inputs, outputs, nrows=4, ncols=4, normalize=False):
    """
    Display input-output image pairs in a grid.

    Args:
        inputs (torch.Tensor): [B, C, H, W] input images
        outputs (torch.Tensor): [B, C, H, W] output images
        nrows (int): Number of pairs per column
        ncols (int): Number of pairs per row
        normalize (bool): If True, assumes float inputs [0,1] and rescales to [0,255]
    """
    batch_size = min(inputs.size(0), outputs.size(0))
    num_pairs = min(batch_size, nrows * ncols)

    inputs = inputs[:num_pairs].detach().cpu()
    outputs = outputs[:num_pairs].detach().cpu()

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 4, nrows * 2))
    axes = axes.reshape(nrows, ncols * 2)

    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            if idx >= num_pairs:
                axes[row, col * 2].axis("off")
                axes[row, col * 2 + 1].axis("off")
                continue

            inp = inputs[idx]
            out = outputs[idx]

            if normalize and inp.dtype.is_floating_point:
                inp = (inp * 255).clamp(0, 255).byte()
                out = (out * 255).clamp(0, 255).byte()

            # Reformat for imshow
            if inp.shape[0] == 1:
                inp = inp.squeeze(0)
                out = out.squeeze(0)
                axes[row, col * 2].imshow(inp, cmap="gray")
                axes[row, col * 2 + 1].imshow(out, cmap="gray")
            else:
                inp = inp.permute(1, 2, 0)
                out = out.permute(1, 2, 0)
                axes[row, col * 2].imshow(inp)
                axes[row, col * 2 + 1].imshow(out)

            axes[row, col * 2].set_title("Input")
            axes[row, col * 2 + 1].set_title("Output")
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].axis("off")

            idx += 1

    plt.tight_layout()
    plt.show()
