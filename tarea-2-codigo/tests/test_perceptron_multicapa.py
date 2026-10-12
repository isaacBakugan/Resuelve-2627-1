import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perceptron_multicapa_1 as perceptron


ASSETS = Path(__file__).resolve().parents[1] / "assets"


@pytest.mark.parametrize(
    ("dataset_name", "losses", "expected_axes"),
    [
        ("2-d_2-class_test.csv", None, 5),
        ("3-d_4-class_test.csv", None, 5),
        ("2-d_2-class_test.csv", [1.0, 0.5], 4),
        ("3-d_4-class_test.csv", [1.0, 0.5], 4),
    ],
)
def test_charts_have_separate_panels_without_layout_warnings(
    dataset_name, losses, expected_axes, monkeypatch
):
    inputs, labels, classes = perceptron.read_dataset(ASSETS / dataset_name)
    captured = {}

    def inspect_figure():
        figure = plt.gcf()
        figure.canvas.draw()
        captured["axes"] = len(figure.axes)
        captured["panels"] = [
            figure.axes[index].get_position().bounds for index in (0, 2, 3)
        ]

    monkeypatch.setattr(perceptron.plt, "show", inspect_figure)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        perceptron.show_results(inputs, labels, labels.copy(), classes, losses)

    assert captured["axes"] == expected_axes
    panel_positions = captured["panels"]
    assert all(position[2] > 0.15 for position in panel_positions)
    assert panel_positions[0][0] < panel_positions[1][0] < panel_positions[2][0]
    assert not caught
    assert plt.get_fignums() == []


def test_graph_failure_is_reported_and_figure_is_closed(monkeypatch, capsys):
    inputs, labels, classes = perceptron.read_dataset(ASSETS / "2-d_2-class_test.csv")

    def fail_drawing(*args, **kwargs):
        raise RuntimeError("simulated graphics failure")

    monkeypatch.setattr(perceptron, "_draw_results", fail_drawing)
    perceptron.display_results(inputs, labels, labels.copy(), classes)

    assert "No se pudieron mostrar los graficos" in capsys.readouterr().out
    assert plt.get_fignums() == []


def test_dataset_rejects_nan_values(tmp_path):
    dataset = tmp_path / "nonfinite.csv"
    dataset.write_text("x1,x2,y\nNaN,0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="NaN o infinitos"):
        perceptron.read_dataset(dataset)
