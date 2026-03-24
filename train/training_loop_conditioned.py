import functools
import torch
from diffusion.resample import LossAwareSampler
from utils import dist_util
from train.training_loop import TrainLoop, log_loss_dict


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
            # (source_* tensors are already on device, moved by run_loop)
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
