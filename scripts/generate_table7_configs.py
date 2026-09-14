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

        # Table 7 uses 8,192 held-out NQ records
        cfg["eval"]["val_size"] = 8192

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
