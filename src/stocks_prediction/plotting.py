"""Plot generation for per-stock and cross-stock PnL results."""
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PnlSeries = Union[Sequence[float], np.ndarray]


def plot_graphs(
    target_stock: str,
    pnl_code: PnlSeries,
    pnl_test_no_code: PnlSeries,
    percent: bool = False,
    save_path: Optional[str] = None,
    dates: Optional[pd.Index] = None,
) -> None:
    """Plot the return-difference box plot and the accumulated PnL curve for one stock.

    Args:
        target_stock: Stock name used in titles and file names.
        pnl_code: Accumulated strategy PnL.
        pnl_test_no_code: Accumulated benchmark (buy & hold) PnL.
        percent: If True, use percentage labels and file suffixes.
        save_path: Directory where the PNG files are saved. If None, nothing is saved.
        dates: Dates for the x-axis of the PnL curve. If None, the sample index is used.
    """
    
    # Convert the sequences to NumPy arrays for element-wise operations
    pnl_code_arr = np.asarray(pnl_code, dtype=float)
    pnl_no_code_arr = np.asarray(pnl_test_no_code, dtype=float)

    x_values = dates if dates is not None else np.arange(len(pnl_code_arr))

    # Difference between daily results
    pnl_diff = pnl_code_arr - pnl_no_code_arr

    if percent:
        # Compound daily percentage returns
        pnl_code_arr = (1 + pnl_code_arr).cumprod() - 1
        pnl_no_code_arr = (1 + pnl_no_code_arr).cumprod() - 1

        # Convert to percentage points for plotting
        pnl_code_arr *= 100
        pnl_no_code_arr *= 100
        pnl_diff *= 100
    else:
        # Accumulate daily monetary PnL
        pnl_code_arr = np.cumsum(pnl_code_arr)
        pnl_no_code_arr = np.cumsum(pnl_no_code_arr)
    # 1. Box plot of the difference between strategy and benchmark
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        pnl_diff,
        orientation="vertical",
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", color="green"),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", color="red", alpha=0.5),
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1, label="Tie (0)")
    if percent:
        plt.title(f"Distribution of Daily Return Difference - {target_stock}")
        plt.ylabel("Daily Return Difference (percentage points)")
    else:
        plt.title(f"Distribution of Daily PnL Difference - {target_stock}")
        plt.ylabel("Daily PnL Difference ($)")
    plt.xticks([1], ["Performance Difference"])
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    if save_path is not None:
        if percent:
            plt.savefig(f"{save_path}/{target_stock}_returnDiff_percentage.png")
        else:
            plt.savefig(f"{save_path}/{target_stock}_returnDiff.png")
    plt.close()

    # 2. Accumulated PnL curve: strategy vs benchmark
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, pnl_code_arr, label="Accumulated PnL - code", color="blue", linewidth=2)
    plt.plot(x_values, pnl_no_code_arr, label="Accumulated PnL - market", color="red", linewidth=2)
    plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="Break-even (0)")

    # 3. Aesthetics
    plt.title(f"Accumulated PnL Curve - ({target_stock})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Net PnL (%)" if percent else "Net PnL ($)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    if save_path is not None:
        if percent:
            plt.savefig(f"{save_path}/{target_stock}_accumulatedPnL_percentage.png")
        else:
            plt.savefig(f"{save_path}/{target_stock}_accumulatedPnL.png")

    # 4. Display the plot
    plt.close()


def plot_graphs_diff(
    df_pnl_code: pd.DataFrame,
    df_pnl_no_code: pd.DataFrame,
    percent: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Box plot of the strategy-minus-benchmark PnL difference, one box per stock.

    Args:
        df_pnl_code: Strategy PnL table (one column per stock).
        df_pnl_no_code: Benchmark PnL table (one column per stock).
        percent: If True, scale values to percentages and use percentage labels.
        save_path: Directory where the PNG files are saved. If None, nothing is saved.
    """
    if percent:
        # Scale the whole DataFrame; iterating over it would yield column names only
        df_pnl_code = df_pnl_code * 100
        df_pnl_no_code = df_pnl_no_code * 100

    # Keep the benchmark columns in the same order as the strategy columns
    df_pnl_no_code = df_pnl_no_code[df_pnl_code.columns]

    # Difference: strategy minus benchmark (one column per stock)
    df_diff = df_pnl_code - df_pnl_no_code
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df_diff.columns)), 6))
    ax.boxplot(
        [df_diff[column].dropna().values for column in df_diff.columns],
        tick_labels=df_diff.columns,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", color="green"),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", color="red", alpha=0.5),
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Tie (0)")
    if percent:
        ax.set_title("Distribution of Daily Return Difference per Stock")
        ax.set_ylabel("Daily Return Difference (percentage points)")
    else:
        ax.set_title("Distribution of Daily PnL Difference per Stock")
        ax.set_ylabel("Daily PnL Difference ($)")
    ax.set_xlabel("Stock")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        if percent:
            plt.savefig(f"{save_path}/ALL_STOCKS_returnDiff_percentage.png")
        else:
            plt.savefig(f"{save_path}/ALL_STOCKS_returnDiff.png")
    plt.close()


def plot_graphs_profit(
    pnl_code: pd.DataFrame,
    pnl_test_no_code: pd.DataFrame,
    percent: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of the final accumulated PnL per stock, strategy vs benchmark.

    Args:
        pnl_code: Strategy PnL table (one column per stock).
        pnl_test_no_code: Benchmark PnL table (one column per stock).
        percent: If True, scale values to percentages and use percentage labels.
        save_path: Directory where the PNG file is saved. If None, nothing is saved.
    """
    if percent:
        pnl_code = pnl_code * 100
        pnl_test_no_code = pnl_test_no_code * 100

    # Final value of each column (last non-null row)
    final_code = pnl_code.ffill().iloc[-1]
    final_market = pnl_test_no_code.ffill().iloc[-1]
    stocks = pnl_code.columns
    x_positions = np.arange(len(stocks))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(stocks)), 6))
    ax.bar(x_positions - bar_width / 2, final_code.values, bar_width,
           label="With code", color="blue")
    ax.bar(x_positions + bar_width / 2, final_market.values, bar_width,
           label="Market (no code)", color="red")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stocks, rotation=45, ha="right")
    if percent:
        ax.set_title("Final Accumulated Percentage PnL per Stock: with code vs market")
        ax.set_ylabel("Final Accumulated Percentage PnL (%)")
    else:
        ax.set_title("Final Accumulated PnL per Stock: with code vs market")
        ax.set_ylabel("Final Accumulated PnL ($)")
    ax.set_xlabel("Stock")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        if percent:
            plt.savefig(f"{save_path}/ALL_STOCKS_PnL_percentage.png")
        else:
            plt.savefig(f"{save_path}/ALL_STOCKS_PnL.png")
    plt.close()


def plot_profit_boxplots(
    df_pnl_code: pd.DataFrame,
    df_pnl_no_code: pd.DataFrame,
    percent: bool = False,
    save_path: str | None = None,
) -> None:
    """Box plot of the PnL distribution for each stock, strategy vs benchmark side by side.
    Args:
        df_pnl_code: Strategy PnL table (one column per stock), indexed by date.
        df_pnl_no_code: Benchmark PnL table (one column per stock), indexed by date.
        percent: If True, scale values to percentages and use percentage labels.
        save_path: Directory where the PNG file is saved. If None, nothing is saved.
    """
    if percent:
        df_pnl_code = df_pnl_code * 100
        df_pnl_no_code = df_pnl_no_code * 100
    stocks = list(df_pnl_code.columns)
    df_pnl_no_code = df_pnl_no_code[stocks]
    # Each stock gets two adjacent boxes: strategy (left) and benchmark (right)
    pos_code = np.arange(len(stocks)) * 3.0
    pos_no_code = pos_code + 1.0
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(stocks)), 6))
    bp_code = ax.boxplot(
        [df_pnl_code[s].dropna().values for s in stocks],
        positions=pos_code,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", color="blue", alpha=0.4),
    )
    bp_no_code = ax.boxplot(
        [df_pnl_no_code[s].dropna().values for s in stocks],
        positions=pos_no_code,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", color="darkred"),
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", color="darkred", alpha=0.4),
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(pos_code + 0.5)
    ax.set_xticklabels(stocks, rotation=45, ha="right")
    ax.set_xlabel("Stock")
    if percent:
        ax.set_title("Distribution of Daily Returns per Stock")
        ax.set_ylabel("Daily Return (%)")
    else:
        ax.set_title("Distribution of Daily PnL per Stock")
        ax.set_ylabel("Daily PnL ($)")
    ax.legend(
        [bp_code["boxes"][0], bp_no_code["boxes"][0]],
        ["With code", "Market (without code)"],
        loc="best",
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    if save_path is not None:
        suffix = "percentage" if percent else "absolute"
        plt.savefig(f"{save_path}/ALL_STOCKS_profit_boxplot_{suffix}.png")
    plt.close()
