from copy import deepcopy
import torch
from torch.optim import Adam
from torch.nn import L1Loss
import pytorch_lightning as pl
from visual import _show_input_output_pairs
import wandb


class IdempotentNetwork(pl.LightningModule):
    def __init__(
        self,
        prior,
        model,
        lr=1e-4,
        criterion=L1Loss(),
        lrec_w=20.0,
        lidem_w=20.0,
        ltight_w=2.5,
    ):
        super(IdempotentNetwork, self).__init__()
        self.prior = prior
        self.model = model
        self.model_copy = deepcopy(model)
        self.lr = lr
        self.criterion = criterion
        self.lrec_w = lrec_w
        self.lidem_w = lidem_w
        self.ltight_w = ltight_w
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim = Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return optim

    def get_losses(self, x):
        # Prior samples
        z = self.prior.sample((x.shape[0],)).to(x.device)

        # Updating the copy
        self.model_copy.load_state_dict(self.model.state_dict())

        # Forward passes
        fx = self(x)
        fz = self(z)
        fzd = fz.detach()

        l_rec = self.lrec_w * self.criterion(fx, x)
        l_idem = self.lidem_w * self.criterion(self.model_copy(fz), fz)
        l_tight = -self.ltight_w * self.criterion(self(fzd), fzd)

        return fx, l_rec, l_idem, l_tight

    def training_step(self, batch, batch_idx):
        _, loss = self.inference_step(batch=batch, type="train")
        return loss

    def validation_step(self, batch, batch_idx):
        fx, _ = self.inference_step(batch=batch, type="val")
        self.validation_step_outputs.append({"input": batch, "output": fx})

    def on_validation_epoch_end(self):
        inputs = torch.cat(
            [out["input"] for out in self.validation_step_outputs], dim=0
        )
        outputs = torch.cat(
            [out["output"] for out in self.validation_step_outputs], dim=0
        )
        fig = _show_input_output_pairs(inputs=inputs, outputs=outputs, nrows=4, ncols=4)

        logger = getattr(self, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log({f"val/generated_samples": wandb.Image(fig)})
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        self.inference_step(batch=batch, type="test")

    def inference_step(self, batch, type="val"):
        fx, l_rec, l_idem, l_tight = self.get_losses(batch)
        loss = l_rec + l_idem + l_tight

        self.log_dict(
            {
                f"{type}/loss_rec": l_rec,
                f"{type}/loss_idem": l_idem,
                f"{type}/loss_tight": l_tight,
                f"{type}/loss": loss,
            },
            sync_dist=True,
        )

        return fx, loss

    def generate_n(self, n, device=None):
        z = self.prior.sample((n,))

        if device is not None:
            z = z.to(device)

        return self(z)
