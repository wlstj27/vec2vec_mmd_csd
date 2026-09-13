import dataclasses
import types

import accelerate
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class VanillaGAN:
    cfg: types.SimpleNamespace
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    discriminator_opt: torch.optim.Optimizer
    discriminator_scheduler: torch.optim.lr_scheduler._LRScheduler
    accelerator: accelerate.Accelerator

    @property
    def _batch_size(self) -> int:
        return self.cfg.bs
    
    def compute_gradient_penalty(self, d_out: torch.Tensor, d_in: torch.Tensor) -> torch.Tensor:
        gradients = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=d_in,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return gradients.pow(2).sum().mean()
    
    def set_discriminator_requires_grad(self, rg: bool) -> None:
        for module in self.discriminator.parameters():
            module.requires_grad = rg

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = F.binary_cross_entropy_with_logits(d_real_logits, real_labels)
        disc_loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = (d_real_logits.sigmoid() < 0.5).float().mean().item()
        disc_acc_fake = (d_fake_logits.sigmoid() > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
        r2_penalty = self.compute_gradient_penalty(d_out=d_fake_logits, d_in=fake_data)

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            (
                disc_loss + 
                ((r1_penalty + r2_penalty) * self.cfg.loss_coefficient_r1_penalty)
            ) * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return (r1_penalty + r2_penalty).detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.zeros((batch_size, 1), device=device)
        gen_loss = F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)
        gen_acc = (d_fake_logits.sigmoid() < 0.5).float().mean().item()
        return gen_loss, gen_acc

    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        if self.cfg.loss_coefficient_disc > 0:
            return self._step_discriminator(real_data, fake_data)
        else:
            return torch.tensor(0.0), 0.0, 0.0

    def step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.cfg.loss_coefficient_gen > 0:
            return self._step_generator(real_data=real_data, fake_data=fake_data)
        else:
            return torch.tensor(0.0), 0.0

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        self.generator.eval()
        self.discriminator.train()
        self.set_discriminator_requires_grad(True)
        r1_penalty, disc_loss, disc_acc_real, disc_acc_fake = self.step_discriminator(
            real_data=real_data.detach(),
            fake_data=fake_data.detach()
        )
        self.generator.train()
        self.discriminator.eval()
        self.set_discriminator_requires_grad(False)
        gen_loss, gen_acc = self.step_generator(
            real_data=real_data,
            fake_data=fake_data
        )

        return r1_penalty, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc



class LeastSquaresGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = (d_real_logits ** 2).mean()
        disc_loss_fake = ((d_fake_logits - 1) ** 2).mean()
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = ((d_real_logits ** 2) < 0.5).float().mean().item()
        disc_acc_fake = ((d_fake_logits ** 2) > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
        r2_penalty = self.compute_gradient_penalty(d_out=d_fake_logits, d_in=fake_data)
        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            (disc_loss + ((r1_penalty + r2_penalty) * self.cfg.loss_coefficient_r1_penalty)) * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return (r1_penalty + r2_penalty).detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        gen_loss = ((d_fake_logits) ** 2).mean()
        gen_acc = ((d_fake_logits ** 2) < 0.5).float().mean().item()
        return gen_loss * 0.5, gen_acc


class RelativisticGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.generator.eval()
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)

        disc_loss = F.binary_cross_entropy_with_logits(d_fake_logits - d_real_logits, torch.ones_like(d_real_logits))
        disc_acc_real = (d_real_logits > d_fake_logits).float().mean().item()
        disc_acc_fake = 1.0 - disc_acc_real

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(disc_loss * self.cfg.loss_coefficient_disc)
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()

        return disc_loss, disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.discriminator.eval()

        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        gen_loss = F.binary_cross_entropy_with_logits(d_real_logits - d_fake_logits, torch.ones_like(d_real_logits))

        gen_acc = (d_real_logits > d_fake_logits).float().mean().item()
        self.discriminator.train()
        return gen_loss, gen_acc

#new_line
# =====================================================================
# MMD / CSD distributional-alignment losses (no discriminator trained)
# Paper ref: UniBioTranslate §3.2.1 — "D may be chosen from: (1) GAN,
# (2) MMD, (3) CSD, (4) Wasserstein" — this implements (2) and (3).
# UniBioTranslate names MMD/CSD as candidate alignment divergences but
# does not fix an estimator or kernel — the choices below (biased vs.
# unbiased MMD, mixture-kernel CSD, pooled median-heuristic bandwidth)
# are this project's own design decisions and should be stated as such
# in Methods, not attributed to the paper.
#
# v3: incorporates a second round of review —
#   - biased MMD^2 is described as "non-negative in exact arithmetic"
#     (not an unconditional ">= 0"): fp round-off can still push a
#     mathematically-nonnegative quantity a hair below 0.
#   - corrected the biased-vs-unbiased gradient comment: it is NOT a
#     uniform m/(m-1) scaling of the whole gradient. The RBF diagonal
#     (k(y_i,y_i)=1) is constant in y, so only the *self*-kernel term's
#     normalization differs between estimators (by a (m-1)/m factor);
#     the cross term -2*Kxy.mean() is identical either way. The net
#     effect is smaller than a flat m/(m-1) on the total gradient.
#   - the n,m>1 guard for the unbiased estimator now returns a
#     graph-connected zero (`y.sum() * 0.0`) instead of a detached
#     `torch.tensor(0.0, ...)`, so a degenerate batch never produces a
#     loss term with no grad_fn at all (defensive; this branch should
#     essentially never trigger at bs=256, but matters for e.g. DDP
#     gradient-sync edge cases with an uneven last batch).
#   - _median_sigma_xy's docstring no longer calls itself "the standard"
#     median heuristic — there are multiple conventions (this project's:
#     sigma = sqrt(median(d^2)/2), pooled over {x,y}, diagonal excluded).
#   - MMDGAN's default mmd_sigma_scales changed from a 5-point multi-
#     bandwidth mixture to a SINGLE scale [1.0], matching CSD's existing
#     single-scale default. The two were asymmetric before (MMD defaulted
#     to an ensemble of 5 bandwidths, CSD to 1), which would confound
#     "MMD vs CSD" with "multi- vs single-bandwidth" in the first Phase 2
#     comparison. Multi-bandwidth MMD is still available by explicitly
#     setting mmd_sigma_scales — treat it as a separate ablation axis
#     applied AFTER a divergence winner is picked, not baked into the
#     default first comparison.
#
# IMPORTANT (carries forward to train.py wiring / evaluation code, not
# a gan.py change): the bandwidth here is recomputed every step from the
# CURRENT batch (adaptive median heuristic). That means the logged loss
# value is computed under a DIFFERENT kernel at every step — an MMD loss
# curve going down does not by itself mean "more aligned than before" in
# a fixed-metric sense, since the yardstick itself is moving. This does
# NOT affect Phase 2's actual comparison metrics (cos/T-1/Rank), which
# are independent of training-time bandwidth. It DOES matter for two
# things to do later: (1) log the per-step sigma (e.g. train/mmd_sigma)
# alongside the loss so the curve is interpretable, and (2) when Phase
# 4/5 report an alignment-divergence number as an evaluation metric,
# compute it with a FIXED bandwidth shared across every model (GAN-,
# MMD-, CSD-trained alike) — never reuse each model's own adaptive
# training-time bandwidth for evaluation, or the comparison is circular.
# =====================================================================

def _pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared L2 distances. Clamped >=0 to guard fp round-off."""
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).transpose(0, 1)
    d2 = x2 + y2 - 2.0 * (x @ y.transpose(0, 1))
    return d2.clamp_min(0.0)


def _median_sigma_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Median-heuristic bandwidth computed from pooled pairwise distances:
    median of ALL pairwise squared distances in the POOLED {x, y} sample
    (diagonal excluded), not just cross-distances. Cross-only bandwidth
    over-smooths early in training when the two domains are still far
    apart. Detached: bandwidth selection is not part of the optimization.
    (One of several median-heuristic conventions in use elsewhere; this
    project's specific choice is sigma = sqrt(median(d^2) / 2).)"""
    with torch.no_grad():
        z = torch.cat([x, y], dim=0).float()
        d2 = _pdist2(z, z)
        mask = ~torch.eye(d2.size(0), dtype=torch.bool, device=d2.device)
        med = d2[mask].median()
        return (med / 2.0).clamp_min(1e-12).sqrt()


def _rbf_mix(d2: torch.Tensor, sigmas) -> torch.Tensor:
    """Average of RBF kernels over a list of bandwidths (mix-then-use,
    never mix logs of separate single-bandwidth divergences)."""
    K = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * s * s + 1e-12)
        K = K + torch.exp(-gamma * d2)
    return K / len(sigmas)


@dataclasses.dataclass
class MMDGAN:
    """
    Distribution-alignment loss via RBF-MMD^2. No discriminator is
    trained. .step() keeps VanillaGAN's exact return signature so it
    drops into train.py without touching the training loop or the
    wandb logging code.

    cfg (all optional; sensible defaults apply if absent):
      - mmd_sigma_scales: list[float] = [1.0]
            multipliers applied to the per-step (pooled) median-heuristic
            bandwidth. Single-scale by default to match CauchySchwarzGAN's
            default and keep the first GAN-vs-MMD-vs-CSD comparison from
            being confounded by an ensemble-vs-single-bandwidth difference.
            Pass e.g. [0.25, 0.5, 1.0, 2.0, 4.0] to opt into a multi-
            bandwidth mixture, as a deliberate, separate ablation.
      - mmd_sigmas: list[float]
            if set, OVERRIDES the median heuristic with fixed absolute
            sigmas. Only use this after you've swept and know the right
            scale for this specific embedding space.
      - mmd_biased: bool = True
            True  -> V-statistic MMD^2 (Kxx.mean()+Kyy.mean()-2*Kxy.mean()),
                     non-negative in exact arithmetic, exactly 0 for
                     identical samples. Default, for readable/monotonic
                     training-stability curves.
            False -> unbiased U-statistic (diagonal excluded from Kxx/Kyy),
                     can be slightly negative for finite samples — a
                     correct property of the estimator, not a bug.
            For an RBF kernel, diagonal self-kernel terms (k(y_i,y_i)=1)
            have zero gradient. So biased vs. unbiased mainly changes the
            normalization of the within-sample (self-kernel) gradient by
            a (m-1)/m factor; the cross term -2*Kxy.mean() is identical
            either way. At bs~256 the net difference is small, but it is
            NOT simply "the whole gradient scaled by m/(m-1)".
    """
    cfg: types.SimpleNamespace
    generator: torch.nn.Module
    discriminator: torch.nn.Module            # unused, kept for API compat
    discriminator_opt: torch.optim.Optimizer   # unused
    discriminator_scheduler: torch.optim.lr_scheduler._LRScheduler  # unused
    accelerator: accelerate.Accelerator

    def _sigmas(self, x: torch.Tensor, y: torch.Tensor):
        fixed = getattr(self.cfg, "mmd_sigmas", None)
        if fixed:
            return list(fixed)
        scales = getattr(self.cfg, "mmd_sigma_scales", [1.0])
        s0 = _median_sigma_xy(x, y)
        return [s0 * f for f in scales]

    def _mmd2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Explicitly disable autocast for the kernel math, regardless of
        # any enclosing accelerator.autocast() region — .float() alone
        # does not stop autocast-eligible ops (e.g. matmul in _pdist2)
        # from being cast back down to fp16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            y = y.float()
            n, m = x.size(0), y.size(0)

            biased = bool(getattr(self.cfg, "mmd_biased", True))
            if not biased and (n < 2 or m < 2):
                # Unbiased estimator is undefined for n<2 or m<2.
                # Return a graph-connected zero (not a detached constant)
                # so this loss term always has a grad_fn, even in this
                # degenerate edge case.
                return y.sum() * 0.0

            sigmas = self._sigmas(x, y)
            Kxx = _rbf_mix(_pdist2(x, x), sigmas)
            Kyy = _rbf_mix(_pdist2(y, y), sigmas)
            Kxy = _rbf_mix(_pdist2(x, y), sigmas)

            if biased:
                return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

            off_xx = Kxx.sum() - Kxx.diagonal().sum()
            off_yy = Kyy.sum() - Kyy.diagonal().sum()
            return off_xx / (n * (n - 1)) + off_yy / (m * (m - 1)) - 2.0 * Kxy.mean()

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        """Returns (r1_penalty, disc_loss, gen_loss, disc_acc_real,
        disc_acc_fake, gen_acc) — matches VanillaGAN.step(). Only
        gen_loss is real; the rest are dummies.

        real_data is detached: the original GAN's generator loss (for
        vanilla/least_squares, the default) never backprops through
        real_data either — only through fake_data. Without detaching
        here, latent_gan's real_data=reps[sup_emb] would gain a gradient
        path it never had under GAN, changing more than just the loss
        formula."""
        self.generator.train()
        zero = torch.tensor(0.0, device=real_data.device)
        gen_loss = self._mmd2(real_data.detach(), fake_data)
        return zero, zero, gen_loss, 0.0, 0.0, 0.0


@dataclasses.dataclass
class CauchySchwarzGAN:
    """
    Distribution-alignment loss via Cauchy-Schwarz divergence between
    kernel Gram matrices (diagonal included -> simple/stable variant;
    D_CS(X,X) == 0 exactly under this definition).

    cfg (optional):
      - csd_sigma_scales: list[float] = [1.0]
            multipliers on the per-step (pooled) median-heuristic bandwidth.
            NOTE: with more than one scale, kernels are mixed BEFORE the
            log is taken. Averaging per-bandwidth CSDs instead defines a
            different (also valid, but different) objective — here we use
            the CSD induced by the mixture kernel.
      - csd_sigmas: list[float]  (overrides median heuristic, like MMDGAN)
    """
    cfg: types.SimpleNamespace
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    discriminator_opt: torch.optim.Optimizer
    discriminator_scheduler: torch.optim.lr_scheduler._LRScheduler
    accelerator: accelerate.Accelerator

    def _sigmas(self, x: torch.Tensor, y: torch.Tensor):
        fixed = getattr(self.cfg, "csd_sigmas", None)
        if fixed:
            return list(fixed)
        scales = getattr(self.cfg, "csd_sigma_scales", [1.0])
        s0 = _median_sigma_xy(x, y)
        return [s0 * f for f in scales]

    def _cs_divergence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            y = y.float()

            sigmas = self._sigmas(x, y)
            Kxx = _rbf_mix(_pdist2(x, x), sigmas)
            Kyy = _rbf_mix(_pdist2(y, y), sigmas)
            Kxy = _rbf_mix(_pdist2(x, y), sigmas)

            num = Kxy.sum().clamp_min(1e-12)
            den = (Kxx.sum().clamp_min(1e-12) * Kyy.sum().clamp_min(1e-12)).sqrt()
            # Cauchy-Schwarz guarantees ratio<=1 in exact arithmetic; clamp
            # the upper end too so fp round-off can't produce a bogus
            # small negative loss (-log of something a hair above 1).
            ratio = (num / den).clamp(min=1e-12, max=1.0)
            return -torch.log(ratio)

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        self.generator.train()
        zero = torch.tensor(0.0, device=real_data.device)
        gen_loss = self._cs_divergence(real_data.detach(), fake_data)
        return zero, zero, gen_loss, 0.0, 0.0, 0.0