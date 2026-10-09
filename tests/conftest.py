from pathlib import Path
import pytest
import torch

torch.set_num_threads(1)

from swe_pinn.config import load_config
from swe_pinn.fixtures import make_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup_run(tmp_path):
    reference = make_fixture(tmp_path / "reference.nc")
    config = load_config(ROOT / "tests/settings/swe_smoke.yaml")
    config["reference"] = str(reference)
    config["output_root"] = str(tmp_path / "runs")
    config["output"]["plots"] = False
    return config, reference, tmp_path
