from pathlib import Path
import copy
import toml

BASE = Path("configs/unsupervised_mmd_gte_gtr_10000.toml")

if not BASE.exists():
    raise FileNotFoundError(BASE)

base = toml.load(BASE)

methods = {
    "gan": "least_squares",
    "mmd": "mmd",
    "csd": "csd",
}

sizes = {
    "10k": 10_000,
    "25k": 25_000,
    "50k": 50_000,
}

for method, gan_style in methods.items():
    for label, n in sizes.items():
        cfg = copy.deepcopy(base)

        # Direction: GTE -> GTR
        cfg["general"]["unsup_emb"] = "gte"
        cfg["general"]["sup_emb"] = "gtr"

        # Alignment method
        cfg["discriminator"]["gan_style"] = gan_style

        # Controlled Table 7 training conditions
        cfg["train"]["sup_points"] = 1_000_000
        cfg["train"]["unsup_points"] = n
        cfg["train"]["epochs"] = 500
        cfg["train"]["min_epochs"] = 80
        cfg["train"]["patience"] = 30
        cfg["train"]["min_delta"] = 0.0

        # Table 7 uses 8,192 held-out NQ records, and the paper's rank
        # metric is over the FULL 8,192-record pool (rank range 1..8192,
        # random ~4096) -- not a smaller pool repeated and averaged.
        # val_bs stays small (safe encoder/translator forward batch);
        # top_k_batches * val_bs must equal top_k_size so eval_loop_'s
        # pooling (see utils/eval_utils.py, patch_eval_pooling.py)
        # assembles the full 8192-example pool before scoring it once.
        cfg["eval"]["val_size"] = 8192
        cfg["eval"]["val_bs"] = 1024
        cfg["eval"]["top_k_batches"] = 8       # 8 x 1024 = 8192
        cfg["eval"]["top_k_size"] = 8192        # matches paper: rank 1..8192

        # Remove method-specific kernel settings first
        cfg["gan"].pop("mmd_sigma_scales", None)
        cfg["gan"].pop("csd_sigma_scales", None)
        cfg["gan"].pop("mmd_sigmas", None)
        cfg["gan"].pop("csd_sigmas", None)

        if method == "mmd":
            cfg["gan"]["mmd_sigma_scales"] = [1.0]
        elif method == "csd":
            cfg["gan"]["csd_sigma_scales"] = [1.0]

        cfg["logging"]["wandb_name"] = (
            f"table7_gte_gtr_{label}_{method}_s5_e500"
        )
        cfg["logging"]["save_dir"] = (
            f"checkpoints/table7_{method}/"
            f"gte_gtr_{n}_s5_e500/"
        )

        out = Path(f"configs/table7_{method}_gte_gtr_{label}.toml")
        with out.open("w") as f:
            toml.dump(cfg, f)

        print(f"created: {out}")
