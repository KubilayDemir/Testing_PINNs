from pathlib import Path
import pytest
import torch

torch.set_num_threads(1)

from swe_pinn.config import load_config
from swe_pinn.synthetic_test_data import make_test_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup_run(tmp_path):
    reference = make_test_data(tmp_path / "reference.nc")
    config = load_config(
        ROOT
        / "software_tests/settings/shallow_water_quick_software_check.yaml"
    )
    config["reference"] = str(reference)
    config["output_root"] = str(tmp_path / "training_results")
    config["output"]["plots"] = False
    return config, reference, tmp_path
