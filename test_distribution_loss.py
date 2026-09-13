"""
utils/gan.py 에 MMDGAN/CauchySchwarzGAN(v2)을 붙여넣은 직후, 학습을 돌리기 전에
먼저 실행하세요:

    python test_distribution_losses.py

GPU 없이 CPU에서도 돌아갑니다. fp16 autocast 스트레스 테스트는 CUDA가 있을 때만 돕니다.
전부 통과해야 Phase 1(N=10,000 sanity run -> 대역폭 스윕)로 넘어가세요.

v3 변경점:
  - test 1의 부등호 버그 수정: `mmd_same_biased > -1e-4` (뭐든 통과하는 조건이었음)
    -> `abs(mmd_same_biased) < 1e-4`
  - NaN 체크(`x == x`, inf를 못 거름) -> `math.isfinite()`로 교체
  - 그래디언트가 finite한 것뿐 아니라 실제로 0이 아닌지(norm>0)도 확인
  - 고정 sigma / 다중 대역폭 분기 테스트 추가 (test 10)
  - test 8 메시지 문구 수정 ("밖에서" -> "안에서")
"""
import math
import sys
import types

import torch

from utils.gan import MMDGAN, CauchySchwarzGAN


def make_gan(cls, **cfg_kwargs):
    cfg = types.SimpleNamespace(**cfg_kwargs)
    return cls(
        cfg=cfg,
        generator=torch.nn.Identity(),
        discriminator=None,
        discriminator_opt=None,
        discriminator_scheduler=None,
        accelerator=None,
    )


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    torch.manual_seed(0)
    ok = True

    mmd_gan = make_gan(MMDGAN)                       # 기본값: mmd_biased=True
    mmd_gan_unbiased = make_gan(MMDGAN, mmd_biased=False)
    csd_gan = make_gan(CauchySchwarzGAN)

    # ---- test 1: X와 완전히 같은 텐서 ----
    # biased: 항상 >=0, 0에 아주 가까워야 함(단측 부등호 버그 수정 -> abs 사용).
    # unbiased: 작은 음수도 정상(추정량 특성).
    X = torch.randn(200, 16)
    mmd_same_biased = mmd_gan._mmd2(X, X).item()
    mmd_same_unbiased = mmd_gan_unbiased._mmd2(X, X).item()
    csd_same = csd_gan._cs_divergence(X, X).item()
    ok &= check("MMD(X,X) biased ≈ 0", abs(mmd_same_biased) < 1e-4, f"value={mmd_same_biased:.6f}")
    ok &= check("MMD(X,X) unbiased 작은 값 (|.| < 0.05, 음수 가능)", abs(mmd_same_unbiased) < 0.05,
                f"value={mmd_same_unbiased:.6f}")
    ok &= check("CSD(X,X) 정확히 0", abs(csd_same) < 1e-4, f"value={csd_same:.6f}")

    # ---- test 2: 같은 분포에서 독립적으로 뽑은 두 표본 ----
    Y = torch.randn(200, 16)
    mmd_same_dist = mmd_gan._mmd2(X, Y).item()
    csd_same_dist = csd_gan._cs_divergence(X, Y).item()
    ok &= check("MMD(같은 분포, 다른 표본) 작음", abs(mmd_same_dist) < 0.05, f"value={mmd_same_dist:.6f}")
    ok &= check("CSD(같은 분포, 다른 표본) 작음", abs(csd_same_dist) < 0.1, f"value={csd_same_dist:.6f}")

    # ---- test 3: 평균이 크게 벌어진 분포 -> 뚜렷하게 커져야 함 (분별력 확인) ----
    Z = torch.randn(200, 16) + 5.0
    mmd_shifted = mmd_gan._mmd2(X, Z).item()
    csd_shifted = csd_gan._cs_divergence(X, Z).item()
    ok &= check("MMD(분포 벌어짐) >> MMD(같은 분포)", mmd_shifted > 10 * abs(mmd_same_dist) + 1e-3,
                f"shifted={mmd_shifted:.4f} vs same={mmd_same_dist:.6f}")
    ok &= check("CSD(분포 벌어짐) >> CSD(같은 분포)", csd_shifted > 10 * csd_same_dist,
                f"shifted={csd_shifted:.4f} vs same={csd_same_dist:.6f}")

    # ---- test 4: .step() 반환 형태가 VanillaGAN과 동일한지 (6-tuple) ----
    r1, dl, gl, dar, daf, ga = mmd_gan.step(X, Y)
    ok &= check(".step() 6-tuple 반환 (MMD)", isinstance(gl, torch.Tensor) and torch.is_tensor(r1))
    r1, dl, gl, dar, daf, ga = csd_gan.step(X, Y)
    ok &= check(".step() 6-tuple 반환 (CSD)", isinstance(gl, torch.Tensor) and torch.is_tensor(r1))

    # ---- test 5: raw 함수(._mmd2/._cs_divergence) 그래디언트가 finite'하고 0이 아닌지 ----
    Xg = torch.randn(64, 32, requires_grad=True)
    Yg = torch.randn(64, 32, requires_grad=True)
    mmd_gan._mmd2(Xg, Yg).backward()
    mmd_grad_finite = (
        Xg.grad is not None and Yg.grad is not None
        and torch.isfinite(Xg.grad).all().item() and torch.isfinite(Yg.grad).all().item()
    )
    mmd_grad_nonzero = Xg.grad.norm().item() > 0 and Yg.grad.norm().item() > 0
    ok &= check("MMD backward(): gradient finite (raw)", mmd_grad_finite)
    ok &= check("MMD backward(): gradient non-zero (raw)", mmd_grad_nonzero,
                f"x_norm={Xg.grad.norm().item():.6e}, y_norm={Yg.grad.norm().item():.6e}")

    Xg2 = torch.randn(64, 32, requires_grad=True)
    Yg2 = torch.randn(64, 32, requires_grad=True)
    csd_gan._cs_divergence(Xg2, Yg2).backward()
    csd_grad_finite = (
        Xg2.grad is not None and Yg2.grad is not None
        and torch.isfinite(Xg2.grad).all().item() and torch.isfinite(Yg2.grad).all().item()
    )
    csd_grad_nonzero = Xg2.grad.norm().item() > 0 and Yg2.grad.norm().item() > 0
    ok &= check("CSD backward(): gradient finite (raw)", csd_grad_finite)
    ok &= check("CSD backward(): gradient non-zero (raw)", csd_grad_nonzero,
                f"x_norm={Xg2.grad.norm().item():.6e}, y_norm={Yg2.grad.norm().item():.6e}")

    # ---- test 6: .step()은 real_data로는 그래디언트를 안 흘려야 함 (detach 확인) ----
    # latent_gan.step(real_data=reps[sup], fake_data=reps[unsup]) 처럼 둘 다
    # translator 출력인 상황을 흉내냄. real_data.grad는 None, fake_data.grad는
    # finite하고 0이 아니어야 함 (원본 GAN의 _step_generator와 동일한 비대칭 구조).
    real = torch.randn(64, 32, requires_grad=True)
    fake = torch.randn(64, 32, requires_grad=True)
    _, _, gl, _, _, _ = mmd_gan.step(real, fake)
    gl.backward()
    ok &= check("MMD .step(): real_data.grad는 None (detach 확인)", real.grad is None)
    fake_ok = (
        fake.grad is not None and torch.isfinite(fake.grad).all().item() and fake.grad.norm().item() > 0
    )
    ok &= check("MMD .step(): fake_data gradient finite & non-zero", fake_ok,
                f"grad_norm={fake.grad.norm().item():.6e}" if fake.grad is not None else "grad is None")

    real2 = torch.randn(64, 32, requires_grad=True)
    fake2 = torch.randn(64, 32, requires_grad=True)
    _, _, gl2, _, _, _ = csd_gan.step(real2, fake2)
    gl2.backward()
    ok &= check("CSD .step(): real_data.grad는 None (detach 확인)", real2.grad is None)
    fake2_ok = (
        fake2.grad is not None and torch.isfinite(fake2.grad).all().item() and fake2.grad.norm().item() > 0
    )
    ok &= check("CSD .step(): fake_data gradient finite & non-zero", fake2_ok,
                f"grad_norm={fake2.grad.norm().item():.6e}" if fake2.grad is not None else "grad is None")

    # ---- test 7: bio 임베딩 스케일(‖x‖≈21, d=640, 정규화 안 됨)에서 finite한지 ----
    Xb = torch.randn(256, 640) * 21.0 / (640 ** 0.5)
    Yb = torch.randn(256, 640) * 19.7 / (640 ** 0.5)
    mmd_bio = mmd_gan._mmd2(Xb, Yb).item()
    csd_bio = csd_gan._cs_divergence(Xb, Yb).item()
    ok &= check("MMD: bio 스케일에서 finite", math.isfinite(mmd_bio), f"value={mmd_bio:.6f}")
    ok &= check("CSD: bio 스케일에서 finite", math.isfinite(csd_bio), f"value={csd_bio:.6f}")

    # ---- test 8: fp16 autocast 안에서도 finite + 값이 fp32 참조값과 가까운지 (CUDA만) ----
    if torch.cuda.is_available():
        Xc, Yc = Xb.cuda(), Yb.cuda()
        mmd_ref = mmd_gan._mmd2(Xc, Yc).item()   # 내부에서 이미 autocast(enabled=False)로 fp32 강제
        csd_ref = csd_gan._cs_divergence(Xc, Yc).item()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mmd_fp16 = mmd_gan._mmd2(Xc, Yc).item()
            csd_fp16 = csd_gan._cs_divergence(Xc, Yc).item()
        ok &= check("MMD: outer fp16 autocast 안에서도 finite", math.isfinite(mmd_fp16), f"value={mmd_fp16:.6f}")
        ok &= check("CSD: outer fp16 autocast 안에서도 finite", math.isfinite(csd_fp16), f"value={csd_fp16:.6f}")
        ok &= check("MMD: autocast enabled=False가 실제로 적용됨 (fp16/fp32 결과 거의 동일)",
                     abs(mmd_fp16 - mmd_ref) < 1e-3, f"fp16={mmd_fp16:.6f} vs fp32={mmd_ref:.6f}")
        ok &= check("CSD: autocast enabled=False가 실제로 적용됨 (fp16/fp32 결과 거의 동일)",
                     abs(csd_fp16 - csd_ref) < 1e-3, f"fp16={csd_fp16:.6f} vs fp32={csd_ref:.6f}")
    else:
        print("[SKIP] fp16 autocast 테스트 (CUDA 없음 — A40/A6000 서버에서 다시 실행하세요)")

    # ---- test 9: 고정 sigma / 다중 대역폭 분기 (Phase 1 스윕에서 실제로 쓸 경로) ----
    mmd_fixed = make_gan(MMDGAN, mmd_sigmas=[1.0, 2.0])
    csd_multi = make_gan(CauchySchwarzGAN, csd_sigma_scales=[0.5, 1.0, 2.0])
    v1 = mmd_fixed._mmd2(X, Y)
    v2 = csd_multi._cs_divergence(X, Y)
    ok &= check("MMD 고정-sigma 분기 finite", math.isfinite(v1.item()), f"value={v1.item():.6f}")
    ok &= check("CSD 다중-대역폭 분기 finite", math.isfinite(v2.item()), f"value={v2.item():.6f}")

    print()
    if ok:
        print("=== 전부 통과. Phase 1(N=10,000 sanity run -> 대역폭 스윕)로 넘어가도 됩니다. ===")
        sys.exit(0)
    else:
        print("=== 일부 실패. gan.py 붙여넣기를 다시 확인하세요. ===")
        sys.exit(1)


if __name__ == "__main__":
    main()