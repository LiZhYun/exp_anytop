import functools
import torch
import matplotlib.pyplot as plt
from diffusion.resample import LossAwareSampler
from utils import dist_util
from train.training_loop import TrainLoop, log_loss_dict
from train.diagnostics import (
    fsq_stats, slot_diversity, null_z_divergence,
    collect_z_embeddings, z_pca_figure, slot_attn_figure,
)


class TrainLoopConditioned(TrainLoop):
    """Extends TrainLoop to encode source motion into z before each diffusion step."""

    def forward_backward(self, batch, cond, epoch=-1):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            micro      = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]

            # Encode source motion → z_embed
            z_embed = self.ddp_model.encoder(
                micro_cond['y']['source_motion'],
                micro_cond['y']['source_offsets'],
                micro_cond['y']['source_joints_mask'],
            )  # [B, N', K, D']
            micro_cond['y']['z'] = z_embed

            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

        # Encoder diagnostics at log_interval
        if self.total_step() % self.log_interval == 0:
            with torch.no_grad():
                z_diag, inter = self.ddp_model.encoder(
                    micro_cond['y']['source_motion'],
                    micro_cond['y']['source_offsets'],
                    micro_cond['y']['source_joints_mask'],
                    return_intermediates=True,
                )
            stats = fsq_stats(inter['z_codes'])
            self.train_platform.report_scalar(
                'fsq_utilization', stats['fsq_utilization'], self.total_step(), 'Encoder')
            self.train_platform.report_scalar(
                'fsq_entropy_norm', stats['fsq_entropy_norm'], self.total_step(), 'Encoder')
            self.train_platform.report_scalar(
                'slot_diversity', slot_diversity(z_diag), self.total_step(), 'Encoder')
            self.train_platform.report_scalar(
                'null_z_divergence', null_z_divergence(self.model.null_z, z_diag),
                self.total_step(), 'Encoder')

    def evaluate(self):
        # WandB's internal step is already at total_step+1 by the time evaluate() is called
        # (scalar logs exhaust the partial-history buffer for total_step).
        # Log images at total_step+1 so WandB accepts them as a new history record.
        step = self.total_step() + 1

        # PCA of z embeddings over ~30 batches
        z_flat, labels = collect_z_embeddings(self.model, self.data, self.device, n_batches=30)
        if z_flat is not None:
            fig = z_pca_figure(z_flat, labels)
            if fig is not None:
                self.train_platform.report_figure('z_pca', fig, step)
                plt.close(fig)

        # Slot attention heatmap for one sample
        with torch.no_grad():
            for _, cond in self.data:
                cond_y = {k: v.to(self.device) if torch.is_tensor(v) else v
                          for k, v in cond['y'].items()}
                _, inter = self.model.encoder(
                    cond_y['source_motion'][:1],
                    cond_y['source_offsets'][:1],
                    cond_y['source_joints_mask'][:1],
                    return_intermediates=True,
                )
                # attn_weights: [1, T, K, J] → [T, K, J]
                attn = inter['attn_weights'][0]
                n_joints = int(cond_y['source_joints_mask'][0].sum().item())
                joint_labels = [str(j) for j in range(n_joints)]
                fig = slot_attn_figure(attn, joint_labels)
                self.train_platform.report_figure('slot_attn', fig, step)
                plt.close(fig)
                break
