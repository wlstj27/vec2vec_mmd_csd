from pathlib import Path

path = Path("train.py")
text = path.read_text()

old = """        elif hasattr(cfg, 'unsup_points'):
            unsupset = dset.select(range(min(cfg.unsup_points, len(dset))))
            supset = dset.select(range(min(cfg.unsup_points, len(dset)), len(dset) - len(unsupset)))
"""

new = """        elif hasattr(cfg, 'unsup_points'):
            if hasattr(cfg, 'sup_points'):
                sup_points = int(cfg.sup_points)
                unsup_points = int(cfg.unsup_points)

                if sup_points <= 0:
                    raise ValueError(f"sup_points must be > 0, got {sup_points}")
                if unsup_points <= 0:
                    raise ValueError(f"unsup_points must be > 0, got {unsup_points}")
                if sup_points + unsup_points > len(dset):
                    raise ValueError(
                        f"Requested sup_points + unsup_points = "
                        f"{sup_points + unsup_points:,}, "
                        f"but training dataset has only {len(dset):,} examples"
                    )

                # Table 7 low-data split:
                # keep the supervised/GTR pool fixed across every N,
                # then vary only the unsupervised/GTE pool.
                supset = dset.select(range(0, sup_points))
                unsupset = dset.select(
                    range(sup_points, sup_points + unsup_points)
                )

                print(
                    f"Using fixed/disjoint training pools: "
                    f"sup({cfg.sup_emb})={len(supset):,}, "
                    f"unsup({cfg.unsup_emb})={len(unsupset):,}"
                )
            else:
                # Preserve the original behavior for legacy configs
                # that specify unsup_points but not sup_points.
                unsupset = dset.select(range(min(cfg.unsup_points, len(dset))))
                supset = dset.select(
                    range(
                        min(cfg.unsup_points, len(dset)),
                        len(dset) - len(unsupset)
                    )
                )
"""

if new in text:
    print("Table 7 sup_points patch is already applied.")
elif old in text:
    path.write_text(text.replace(old, new))
    print("Table 7 sup_points patch applied successfully.")
else:
    raise RuntimeError(
        "Expected unsup_points block was not found. "
        "train.py was NOT modified."
    )
