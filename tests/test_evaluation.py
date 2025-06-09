from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import pytest

from src.backtest.evaluation import evaluate_portfolio


def synthetic_returns_series():
    return pd.Series(
        [0.01, -0.02, 0.03, 0.02],
        index=pd.period_range("2020-01", periods=4, freq="M"),
        name="returns",
    )


def test_evaluate_portfolio_series():
    returns = synthetic_returns_series()
    metrics = evaluate_portfolio(returns)

    # Verify keys
    expected_keys = {
        "cumulative_returns",
        "drawdowns",
        "mean_return",
        "std_dev",
        "sharpe_ratio",
    }
    assert set(metrics.keys()) == expected_keys

    cumulative_expected = pd.Series(
        [1.01, 0.9898, 1.019494, 1.039884], index=returns.index
    )
    drawdown_expected = pd.Series([0.0, -0.02, 0.0, 0.0], index=returns.index)

    pd.testing.assert_series_equal(
        metrics["cumulative_returns"].round(6),
        cumulative_expected,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        metrics["drawdowns"].round(6),
        drawdown_expected,
        check_names=False,
    )
    assert metrics["mean_return"] == pytest.approx(0.01)
    assert metrics["std_dev"] == pytest.approx(0.02160247)
    assert metrics["sharpe_ratio"] == pytest.approx(1.60356745)


def test_evaluate_portfolio_dataframe():
    returns = synthetic_returns_series()
    df = returns.reset_index()
    df.columns = ["month", "portfolio_return"]
    df["month"] = df["month"].astype(str)

    metrics = evaluate_portfolio(df)

    cumulative_expected = pd.Series(
        [1.01, 0.9898, 1.019494, 1.039884], index=pd.to_datetime(df["month"])
    )

    pd.testing.assert_series_equal(
        metrics["cumulative_returns"].round(6),
        cumulative_expected,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        metrics["drawdowns"].round(6),
        pd.Series([0.0, -0.02, 0.0, 0.0], index=pd.to_datetime(df["month"])),
        check_names=False,
    )
    assert metrics["mean_return"] == pytest.approx(0.01)
    assert metrics["std_dev"] == pytest.approx(0.02160247)
    assert metrics["sharpe_ratio"] == pytest.approx(1.60356745)


