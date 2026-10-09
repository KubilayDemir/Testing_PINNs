"""Read earlier run/reference schemas without rewriting original files."""

LEGACY_NAMES = {
    "x_h": "x_zeta",
    "h_ref": "zeta_ref",
    "h_pred": "zeta_pred",
    "h_error": "zeta_error",
    "ic_h_pred": "ic_zeta_pred",
    "ic_h_target": "ic_zeta_target",
    "mse_ref_h": "mse_ref_zeta",
    "residual_u": "momentum_residual",
    "residual_h": "continuity_residual",
    "loss_pde_u": "loss_momentum",
    "loss_pde_h": "loss_continuity",
    "loss_ic_h": "loss_ic_zeta",
    "loss_bc_lower": "loss_bc_left",
    "loss_bc_upper": "loss_bc_right",
    "u_bc_lower": "u_bc_left",
    "u_bc_upper": "u_bc_right",
}


def canonical_dataset(dataset):
    return dataset.rename(
        {
            old: new
            for old, new in LEGACY_NAMES.items()
            if old in dataset and new not in dataset
        }
    )
