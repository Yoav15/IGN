import torch
from torch.utils.data import Dataset, DataLoader
from model import DCGANLikeModel
from visual import show_input_output_pairs


class RandomImageDataset(Dataset):
    def __init__(
        self,
        num_samples,
        image_height,
        image_width,
        channels=3,
        normalized=True,
        seed=None,
        device="cpu",
    ):
        """
        Random Image Dataset for testing pipelines.

        Args:
            num_samples (int): Total number of samples in dataset.
            image_height (int): Height of images.
            image_width (int): Width of images.
            channels (int): Channels (1=grayscale, 3=RGB).
            normalized (bool): If True, values in [0,1], else [0,255].
            seed (int, optional): Random seed for reproducibility.
            device (str): 'cpu' or 'cuda'.
        """
        self.num_samples = num_samples
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.normalized = normalized
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.normalized:
            img = torch.rand(
                self.channels, self.image_height, self.image_width, device=self.device
            )
        else:
            img = torch.randint(
                0,
                256,
                (self.channels, self.image_height, self.image_width),
                dtype=torch.uint8,
                device=self.device,
            )
        return img


if __name__ == "__main__":
    # Create dataset with 1000 random RGB 64x64 images
    dataset = RandomImageDataset(
        num_samples=32, image_height=28, image_width=28, channels=1, normalized=True
    )

    # Wrap in DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DCGANLikeModel()

    show_input_output_pairs(model, dataloader, 4, 4)
