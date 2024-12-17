import os
import numpy as np
import torch
import lightning as L
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.cli import LightningCLI

from datasets import DataModule
from utils import (
    compute_metrics,
    save_val_image,
    save_preds,
    save_eval_images,
    to_complex
)


class Runner(L.LightningModule):
    def __init__(
        self,
        mode,
        network,
        lr,
        lr_min,
        optim_betas,
        use_eval_mask,
        multiscale_loss,
        multiscale_loss_weight
    ):
        super().__init__()
        self.save_hyperparameters(ignore="network")

        self.mode = mode.lower()
        self.lr = lr
        self.lr_min = lr_min
        self.optim_betas = optim_betas
        self.use_eval_mask = use_eval_mask
        self.multiscale_loss = multiscale_loss
        self.loss_weight = multiscale_loss_weight

        # Networks
        self.net = network

    def training_step(self, batch):
        method = getattr(self, f"training_step_{self.mode}", None)
        return method(batch)

    def validation_step(self, batch, batch_idx):
        method = getattr(self, f"validation_step_{self.mode}", None)
        return method(batch, batch_idx)

    def on_test_start(self):
        self.test_samples = []
        self.psnrs = []
        self.ssims = []
        self.eval_mask = None

        # Load mask for evaluation
        if self.use_eval_mask:
            self.eval_mask = self.trainer.datamodule.test_dataset._load_mask()

    def test_step(self, batch, batch_idx):
        method = getattr(self, f"test_step_{self.mode}", None)
        x_recon = method(batch, batch_idx)

        x_fs, *_, slice_idx = batch

        # Gather pred images across all ranks
        all_pred = self.all_gather(x_recon.abs())
        slice_indices = self.all_gather(slice_idx)
        
        if self.global_rank == 0:
            h, w = x_fs.shape[-2:]
            self.test_samples.extend(list(zip(
                slice_indices.flatten().tolist(),
                all_pred.reshape(-1, h, w).cpu().numpy())))

    def training_step_mri(self, batch):
        """ Supervised MRI reconstruction training step """
        x_fs, x_us, mask, coilmap, _ = batch

        # Supervised training
        x_fs = torch.cat((x_fs.real, x_fs.imag), dim=1)
        x_us_cc = (torch.conj(coilmap) * x_us).sum(axis=1, keepdim=True)

        x_recon = self.net(
            x=torch.cat([x_us_cc.real, x_us_cc.imag], dim=1),
            target=x_us,
            mask=mask,
            coilmap=coilmap
        )
        
        # Compute loss
        if self.multiscale_loss:
            loss = F.mse_loss(x_recon[0], x_fs)
            loss += self.loss_weight * F.mse_loss(
                x_recon[1],
                x_fs.repeat(1, x_recon[1].shape[1]//x_fs.shape[1], 1, 1)
            )
        else:
            loss = F.mse_loss(x_recon, x_fs)

        # Logging
        self.log("loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step_mri(self, batch, batch_idx):
        """ Supervised MRI reconstruction validation step """
        x_fs, x_us, mask, coilmap, _ = batch

        # Compute coil-combined image
        x_us_cc = (torch.conj(coilmap) * x_us).sum(axis=1, keepdim=True)
        
        # Perform reconstruction: input and output are complex tensors
        x_recon = self.net(
            x=torch.cat([x_us_cc.real, x_us_cc.imag], dim=1),
            target=x_us,
            mask=mask,
            coilmap=coilmap
        )

        if self.multiscale_loss:
            x_recon = x_recon[0]

        # Convert to complex
        x_recon = to_complex(x_recon)
        
        # Compute metrics
        metrics = compute_metrics(x_fs.abs(), x_recon.abs())

        # Log metrics
        self.log("val_psnr", metrics["psnr_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics["ssim_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)

        # Log sample images
        if batch_idx == 0 and self.global_rank == 0:
            path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
            save_val_image(x_fs.abs(), x_us_cc.abs(), x_recon.abs(), metrics, path)

    def test_step_mri(self, batch, batch_idx):
        """ Supervised MRI reconstruction test step """
        x_fs, x_us, mask, coilmap, _ = batch

        # Compute coil-combined image
        x_us_cc = (torch.conj(coilmap) * x_us).sum(axis=1, keepdim=True)

        # Perform reconstruction: input and output are complex tensors
        x_recon = self.net(
            x=torch.cat([x_us_cc.real, x_us_cc.imag], dim=1),
            target=x_us,
            mask=mask,
            coilmap=coilmap
        )

        if self.multiscale_loss:
            x_recon = x_recon[0]

        # Convert to complex
        x_recon = to_complex(x_recon)

        return x_recon

    def training_step_ct(self, batch):
        """ Supervised CT reconstruction training step """
        x_fs, x_us, s_us, theta, us_factor, _ = batch
        
        # Prediction
        x_recon = self.net(x_us, s_us, theta, us_factor[0])
        
        # Compute loss
        if self.multiscale_loss:
            loss = F.mse_loss(x_recon[0], x_fs)
            loss += self.loss_weight * F.mse_loss(
                x_recon[1],
                x_fs.repeat(1, x_recon[1].shape[1]//x_fs.shape[1], 1, 1)
            )
        else:
            loss = F.mse_loss(x_recon, x_fs)

        # Logging
        self.log("loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step_ct(self, batch, batch_idx):
        """ Supervised CT reconstruction validation step """
        x_fs, x_us, s_us, theta, us_factor, _ = batch

        # Prediction
        x_recon = self.net(x_us, s_us, theta, us_factor[0])

        if self.multiscale_loss:
            x_recon = x_recon[0]

        # Compute metrics
        metrics = compute_metrics(x_fs, x_recon)

        # Log metrics
        self.log("val_psnr", metrics["psnr_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics["ssim_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)

        # Log sample images
        if batch_idx == 0 and self.global_rank == 0:
            path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
            save_val_image(x_fs.abs(), x_us.abs(), x_recon.abs(), metrics, path)

    def test_step_ct(self, batch, batch_idx):
        """ Supervised CT reconstruction test step """
        x_fs, x_us, s_us, theta, us_factor, _ = batch
        x_recon = self.net(x_us, s_us, theta, us_factor[0])

        if self.multiscale_loss:
            x_recon = x_recon[0]
            
        return x_recon

    def on_test_end(self):
        # Save predicted images
        if self.global_rank == 0:
            # Sort samples by slice index
            self.test_samples.sort(key=lambda x: x[0])
            
            # Extract pred images
            pred = np.array([x[1] for x in self.test_samples])
            slice_indices = np.array([x[0] for x in self.test_samples])

            # Remove repeated slices that can occur in multi-GPU setting
            _, locs = np.unique(slice_indices, return_index=True)
            pred = pred[locs]

            # Get source and target images
            dataset = self.trainer.datamodule.test_dataset
            source = dataset.image_us
            target = np.abs(dataset.image_fs)


            # Save predictions
            path = os.path.join(self.logger.log_dir, "test_samples", "pred.npy")
            save_preds(pred, path)

            # Compute metrics and save report
            metrics = compute_metrics(
                gt_images=target,
                pred_images=pred,
                mask=self.eval_mask,
                subject_ids=dataset.subject_ids,
                report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt")
            )

            # Print metrics
            print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"SSIM: {metrics['ssim_mean']:.2f} ± {metrics['ssim_std']:.2f}")

            # Save sample images
            indices = np.random.choice(len(dataset), 10)
            path = os.path.join(self.logger.log_dir, "test_samples")

            if hasattr(dataset, "coilmaps"):
                coilmaps = dataset.coilmaps
                source_images = (source[indices] * np.conj(coilmaps[indices])).sum(axis=1, keepdims=True)
                source_images = np.abs(source_images)
            else:
                source_images = source[indices]
        
            save_eval_images(
                source_images=source_images,
                target_images=target[indices],
                pred_images=pred[indices],
                psnrs=metrics["psnrs"][indices],
                ssims=metrics["ssims"][indices],
                save_path=os.path.join(self.logger.log_dir, "test_samples")
            )

    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=self.lr, betas=self.optim_betas)
        
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr_min)

        return [optimizer], [scheduler]


class _LightningCLI(LightningCLI):
    def instantiate_classes(self):
        # Log to checkpoint directory when testing
        if 'test' in self.parser.args and 'CSVLogger' in self.config.test.trainer.logger[0].class_path:
            exp_dir = os.path.dirname(os.path.dirname(self.config.test.ckpt_path))
            logger = self.config.test.trainer.logger[0]
            logger.init_args.save_dir = os.path.dirname(exp_dir)
            logger.init_args.name = os.path.basename(exp_dir)
            logger.init_args.version = "test"

        super().instantiate_classes()

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--model_configs", type=dict)

 
def cli_main():
    cli = _LightningCLI(
        Runner,
        DataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main()
