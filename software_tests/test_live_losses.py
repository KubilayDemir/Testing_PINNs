"""Live monitoring tolerates interrupted writes and separates SWE time segments."""

import json

from swe_pinn.live_losses import loss_records, loss_payload


def test_partial_reports_and_nonfinite_losses(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"epoch":0,"loss_pde":1.0}\ninvalid\n'
                    '{"epoch":1,"loss_pde":NaN,"loss_ic":0.5}\n'
                    '{"epoch":2,"loss_pde":0.1}', encoding="utf-8")
    assert loss_records(path) == [
        {"epoch": 0, "loss_pde": 1.0}, {"epoch": 1, "loss_ic": 0.5}]
    with path.open("a") as stream:
        stream.write("\n")
    assert loss_records(path)[-1] == {"epoch": 2, "loss_pde": 0.1}


def test_sampling_preserves_full_epoch_range(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps({"epoch": i, "loss_pde": 1 / (i + 1)}) + "\n"
                            for i in range(100)), encoding="utf-8")
    rows = loss_records(path, max_points=5)
    assert len(rows) == 5
    assert rows[0]["epoch"] == 0 and rows[-1]["epoch"] == 99
    assert rows[2]["epoch"] in (49, 50)


def test_segment_histories_and_waiting_for_first_report(tmp_path):
    for label in ("000", "001"):
        (tmp_path / "segments" / label).mkdir(parents=True)
    (tmp_path / "segments/000/history.jsonl").write_text(
        '{"epoch":3,"loss_total":0.2}\n', encoding="utf-8")
    payload = loss_payload(tmp_path)
    assert [series["label"] for series in payload["series"]] == ["000", "001"]
    assert payload["series"][0]["records"][0]["epoch"] == 3
    assert payload["series"][1]["records"] == []
