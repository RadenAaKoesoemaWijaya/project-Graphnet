import numpy as np
import pandas as pd
import pytest

from ui.utils import (
    create_histogram_chart,
    create_probability_distribution,
    sample_dataframe_for_visualization,
    sample_values_for_visualization,
)


def test_sample_dataframe_for_visualization_is_bounded_and_deterministic():
    frame = pd.DataFrame({"value": range(100)})

    first = sample_dataframe_for_visualization(frame, max_rows=10)
    second = sample_dataframe_for_visualization(frame, max_rows=10)

    assert len(first) == 10
    pd.testing.assert_frame_equal(first, second)
    assert len(frame) == 100


def test_sample_values_for_visualization_handles_empty_and_large_arrays():
    assert sample_values_for_visualization([]).size == 0

    values = np.arange(100)
    sampled = sample_values_for_visualization(values, max_values=12)

    assert sampled.size == 12
    assert set(sampled).issubset(set(values))


def test_sample_helpers_reject_invalid_limits():
    with pytest.raises(ValueError):
        sample_dataframe_for_visualization(pd.DataFrame({"value": [1]}), max_rows=0)


def test_chart_helpers_guard_empty_inputs_and_build_valid_charts():
    assert create_histogram_chart(pd.DataFrame(), "value") is None
    assert create_probability_distribution([]) is None
    assert create_histogram_chart(pd.DataFrame({"value": [1, 2, 3]}), "value") is not None
    assert create_probability_distribution(np.array([0.1, 0.5, 0.9])) is not None