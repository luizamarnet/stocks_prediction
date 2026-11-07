# Stock Market Prediction and Backtesting

A machine learning project exploring whether historical stock market data can be used to identify profitable trading opportunities.

The project combines **Principal Component Analysis (PCA)**, **Multi-Layer Perceptrons (MLPs)**, technical indicators, and a weekly walk-forward re-training and evaluation framework. It investigates the impact of return regression, classification-based trading signals, and periodic model retraining on portfolio performance.

The ultimate goal is not simply to predict stock prices, but to evaluate whether machine learning predictions can be translated into a trading strategy that outperforms a buy-and-hold benchmark after transaction costs.

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Dimensionality Reduction](#3-dimensionality-reduction)
  - [4. Predictive Models](#4-predictive-models)
  - [5. Walk-Forward Evaluation](#5-walk-forward-evaluation)
  - [6. Trading Strategy and Backtesting](#6-trading-strategy-and-backtesting)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
  - [1. Impact of the Regressor](#1-impact-of-the-regressor)
  - [2. Impact of Weekly Model Retraining](#2-impact-of-weekly-model-retraining)
  - [3. Overall Performance Comparison](#3-overall-performance-comparison)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## Overview

The project implements a machine learning-based trading strategy for multiple stocks.

For each stock, historical market information is transformed into a collection of technical and statistical features. PCA is applied to the returns of the stock universe to capture common market movements, and the resulting components are incorporated into the predictive models.

Two complementary prediction tasks are investigated:

1. **Classification:** Estimate the probability that a stock will generate a positive trading signal.
2. **Regression:** Estimate the expected next-day return.

The classification output determines whether a trading position should be opened. When regression is enabled, the predicted return is additionally used to assess whether the expected trade is profitable.

The strategy is evaluated using historical out-of-sample data and compared against a buy-and-hold benchmark.

### Main components

- Multi-stock historical price data.
- Technical indicator and lagged-return feature engineering.
- PCA-based dimensionality reduction.
- MLP classification and regression models.
- Chronological training and validation.
- Weekly walk-forward evaluation.
- Transaction-cost-aware backtesting.
- Per-stock and aggregate performance visualizations.

## Methodology

### 1. Data Preparation

The project uses historical stock market data organized into three chronological datasets:

- Training data
- Validation data
- Test data

The pipeline loads daily OHLCV information:

- Open
- High
- Low
- Close
- Volume

The models use opening prices as their primary prediction reference, while additional OHLCV information is available for feature engineering.

All datasets are indexed by trading date, with individual stocks represented by separate columns.

### 2. Feature Engineering

For each target stock, the feature engineering pipeline combines information from its own historical performance with market-wide information.

The main feature groups include:

| Feature group | Description |
|---|---|
| Historical returns | Lagged daily returns |
| Moving averages | Relative distance to SMA20, SMA50 and SMA200 |
| Bollinger Bands | Relative distance to the upper and lower bands |
| MACD | Lagged MACD histogram |
| Volatility | Rolling standard deviation of returns |
| RSI | Relative Strength Index |
| PCA components | Current and lagged principal component scores |
| OHLCV | Features derived from open, high, low, close and volume |

The features are designed to capture different aspects of market behavior, including momentum, trend, volatility and market-wide movements.

Lagged features allow the models to incorporate historical information instead of relying exclusively on the latest observation.

### 3. Dimensionality Reduction

PCA is applied to the returns of the stock universe to identify common market movements.

For each walk-forward training window:

1. Historical returns are calculated.
2. PCA is fitted using the training observations only.
3. The selected principal components are extracted.
4. The fitted PCA transformation is applied to the evaluation observations.

The resulting components are incorporated into each stock's feature matrix.

Importantly, PCA is refitted using only historical training data, avoiding the use of future observations when estimating the dimensionality reduction transformation.

### 4. Predictive Models

The project uses Multi-Layer Perceptrons implemented with scikit-learn.

Architecture:

- Input: engineered features.
- Hidden layers: 64, 32 and 8 neurons.
- Output: Predicted percentage change between the current trading day's opening price and the next trading day's opening price.
    * label_type = "percentage": The target is the percentage change in the opening price, before transaction fees.
    * label_type = "profit": The target is the expected percentage change in the invested capital after discounting trading fees. The calculation accounts for the amount invested (user-defined budget), so transaction costs are considered relative to the actual investment rather than just the stock price.

#### Classification model

The classifier estimates the probability of a positive trading signal.

Architecture:

- Input: engineered features.
- Hidden layers: 64 and 32 neurons.
- Output: Probability of a positive percentage change in the invested capital on the next trading day.
    * label_type = "percentage": The positive class indicates an increase in the stock's closing price, before transaction fees.
    * label_type = "profit": The positive class indicates a positive percentage change in the invested capital after discounting trading fees, accounting for the user-defined investment budget.

Observe that for both label_type = "percentage" and label_type = "profit", the backtesting consider the trading fees, the only change is in the target values used for training the models

#### Regression model

The regressor estimates the expected next-day return.

Architecture:

- Input: engineered features.
- Hidden layers: 64, 32 and 8 neurons.
- Output: predicted return.

The regression pipeline includes separate feature and target standardization. Predictions are transformed back to their original scale before being used by the trading strategy.

The classifier and the regressor use:

- StandardScaler for feature normalization.
- Stochastic Gradient Descent (SGD).
- Validation datasets.
- Early stopping based on validation loss.
- An adjustable learning rate.

The last 10% of the training observations are used for validation, preserving chronological ordering.


#### Combining classification and regression

When both models are enabled, the trading decision combines their predictions:

1. The classifier estimates the probability of a positive signal.
2. The regressor estimates the expected return.
3. The predicted return is translated into an estimated monetary profit.
4. Expected transaction costs are considered.
5. A position is opened only when the classification signal and expected profitability conditions are satisfied.

This allows the regression model to act as an additional filter rather than relying exclusively on the classification probability.

* When label_type = "profit", the regressor is deactivated.

#### Classification without regression

The classifier can also be trained independently.

In this configuration, the classification target is based on whether the next-day price increases or not.

When label_type = "profit", the target considers if or not the next day generates a profit after the deducting trading fees, and taking into consideration the budget and a minimum profit chosen by the user. 

The classifier therefore learns to identify potentially profitable opportunities directly, without requiring a separate return estimate.

### 5. Walk-Forward Evaluation

If 'retrain_walk_forward'=True, the model is retrained for having an up to date model to use before evaluating each week in the testing data.

In other words, the model is evaluated on successive weeks, with training restricted to information available before each evaluation period.

For every test week:

1. Set 'retrain_walk_forward'=True
2. Select the historical training window.


The training window is defined by a fixed number of historical trading days.

The evaluation is repeated sequentially across all test weeks and individual stocks.


### 6. Trading Strategy and Backtesting

The predictions are converted into simulated trading decisions.

The strategy uses the following entry conditions:

- Classification probability above 0.50.
- When regression is enabled, estimated net profit must be positive after transaction costs.

Positions are closed when the indication to buy is negative or at the end of the evaluation period.

The backtesting implementation accounts for:

- Initial investment budget.
- Number of shares purchased.
- Fixed transaction fees.
- Proportional transaction commissions.
- Net realized profit and loss.


The strategy is compared against a buy-and-hold benchmark, which purchases the stock at the beginning of the evaluation period and holds it until the end.

Both strategies use the same initial capital and transaction-cost assumptions.

Performance is recorded in absolute monetary terms and as a fraction of the initial investment.

---

## Experimental Setup

The experiments investigate two main questions:

1. Does incorporating a regression model improve the trading strategy?
2. Does retraining the models every week improve out-of-sample performance?

The experiments are organized around the following configurations:

| Experiment | Regressor | Weekly retraining |
|---|---|---|
| A | Enabled | Enabled |
| B | Disabled | Enabled |
| C | Enabled | Disabled |
| D | Disabled | Disabled |

This allows the impact of each modeling decision to be investigated independently.

For a fair comparison, the experiments should use the same stocks, historical periods, initial capital and transaction-cost assumptions.

### Current default parameters

The main entry point currently uses the following configuration:

| Parameter | Value |
|---|---:|
| Initial budget per stock | 15,000 |
| Fixed buy/sell fee | 2 |
| Proportional commission | 0.08 bps |
| Number of PCA components | 2 |
| Number of past days | 5 |
| Profit threshold | 5 |
| Classifier probability threshold | 0.50 |
| Classifier hidden layers | (64, 32) |
| Regressor hidden layers | (64, 32, 8) |
| Training window | Training dataset length |
| Retraining frequency | Weekly |

The `label_type` parameter controls the target construction:

- `percentage`: regression target is the next-day percentage return; classification target indicates a positive next-day return.
- `profit`: classification target indicates whether the next-day net profit reaches the configured monetary threshold. The regressor is not trained in this mode.

---

## Results

*Results will be added after completing the experimental runs.*

The following section provide placeholders for the plots and tables generated by the experiments.

### 1. Overall Performance Comparison

The following table consolidates the four experimental configurations.

This experiment uses:
 * The Copenhagen market
 * It includes 70 stocks
 * The total invested money was number_of_stocks * budget
 * At this stage, the backtest assumes that the next prediction can be evaluated at market open and executed immediately at the same opening price.


| Experiment | regressor_flag | retrain_walk_forward | Budget per Stock ($) | Minimum Profit ($) | label_type | Total net PnL with Strategy (%) | Total net PnL with Buy and Hold (%) | Diff (PnL with strategy - PnL without strategy) (%)| Total net PnL with Strategy without NEWCAP.CO (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | True | True | 15000 | Not considered | percentage | 1079.46 | 36.80 |1042.66 |135.58 |
| B | False | True | TBD | TBD | TBD | TBD | TBD |TBD |TBD |
| C | True | False | TBD | TBD | TBD | TBD | TBD |TBD |TBD |
| D | False | False | TBD | TBD | TBD | TBD | TBD |TBD |TBD |

Given that the NEWCAP.CO presented a exceptionaly good result, that does not reflect the results of majority of other stocks, we also present the results without this stock.

* __Total net PnL with Strategy__ and __Total net PnL with Buy and Hold__ reflects the final total PnL after the closing of the last investment day for all stocks combined.

#### Statistics

|        | Experiment A Strategy (%) | No Strategy (%) | Experiment A Strategy without NewCap.CO (%) |
|:-------|-------:|-------:|-------:|
| Mean   | 1079.46 | 137.54 | 36.80 |
| Median | 40.33 | 038.54 | 24.38 |
| Min    | -48.31 | -48.31 | -86.78 |
| Max    | 66071.73 | 1638.11 | 313.61 |

Also, for 61.43% of the stocks, our strategy from experiment A performed better than buy-and-hold.

#### Plots

**Experiment A**

In the graphs bellow, blue represents the results obtained by our strategy on experiment one and red represents the results obteained by buying and holding.

Stocks where the model-driven strategy performed exceptionally well.

![good results](results\plots\AGAT.CO_accumulatedPnL_percentage.png)


Stocks where the buy-and-hold strategy outperformed the model-driven strategy.

![bad results](results\plots\ALSYDB.CO_accumulatedPnL_percentage.png)

**Discussion**

[Summarize the main findings, identifying which configuration performs best and whether the results are robust across individual stocks.]

---

## Project Structure

```text
stocks_prediction/
│
├── data/
│   └── processed_data/
│       ├── train_close.csv
│       ├── validation_close.csv
│       ├── test_close.csv
│       ├── train_open.csv
│       ├── validation_open.csv
│       ├── test_open.csv
│       └── ... (high, low, close, volume)
│
├── src/
│   └── stocks_prediction/
│       ├── backtest.py
│       ├── data_loader.py
│       ├── features.py
│       ├── indicators.py
│       ├── models.py
│       ├── pca_utils.py
│       ├── plotting.py
│       ├── returns.py
│       └── walk_forward.py
│
├── results/
│   └── plots/
│
├── main.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

### Main modules

| Module | Responsibility |
|---|---|
| `main.py` | Main execution pipeline and experiment configuration |
| `data_loader.py` | Loading and combining historical datasets |
| `features.py` | Feature engineering and target construction |
| `indicators.py` | Technical indicator calculations |
| `returns.py` | Return calculations, including transaction costs |
| `pca_utils.py` | PCA fitting and transformation |
| `models.py` | MLP classifier and regressor training |
| `walk_forward.py` | Weekly walk-forward training and prediction |
| `backtest.py` | Trading simulation and buy-and-hold benchmark |
| `plotting.py` | Per-stock and aggregate performance visualizations |

---

## Installation

The project uses Python 3.11 or later and [uv](https://docs.astral.sh/uv/) for dependency management.

Clone the repository:

```bash
git clone https://github.com/luizamarnet/stocks_prediction.git
cd stocks_prediction
```

Install the dependencies:

```bash
uv sync
```

This creates the project environment and installs the dependencies defined in `pyproject.toml`.

## Running the Project

Before executing the pipeline, ensure that the processed datasets are available under:

```text
data/processed_data/
```

The required CSV files should contain:

- A `Date` column used as the index.
- One column per stock.
- Consistent stock symbols across the OHLCV datasets.

Run the complete evaluation pipeline:

```bash
uv run python main.py
```

The pipeline will:

1. Load the historical datasets.
2. Generate weekly test periods.
3. Train and evaluate models for each stock.
4. Simulate the trading strategy.
5. Compare the strategy against buy-and-hold.
6. Save the cumulative PnL results to CSV files.
7. Generate individual and aggregate performance plots.

### Generated outputs

The main execution generates:

```text
resultados_all_stocks_pnl_code.csv
resultados_all_stocks_pnl_no_code.csv
resultados_all_stocks_pnl_code_percent.csv
resultados_all_stocks_pnl_no_code_percent.csv
```

These files contain the cumulative PnL results across the evaluated stocks.

The corresponding plots are saved under:

```text
results/plots/
```

---

## Limitations and Future Improvements

This project is an experimental environment for investigating machine learning-based trading strategies. Several aspects could be extended:

- **Model comparison:** Evaluate additional algorithms, including Random Forests, Gradient Boosting and other neural network architectures.
- **Feature selection:** Investigate the contribution of individual technical indicators and PCA components, and investigate new features to use.
- **Hyperparameter optimization:** Explore different architectures, learning rates and training window sizes.
- **Risk management:** Investigate alternative stop-loss, position-sizing and portfolio allocation mechanisms.
- **Robustness:** Evaluate performance across different market periods and different markets.
- **Computational efficiency:** Compare the cost of weekly retraining with fixed-model inference.

## Disclaimer

This project is intended for research and educational purposes only. Historical backtesting results do not guarantee future performance. The simulated strategy does not account for every real-world trading constraint, including market impact, liquidity limitations and execution slippage.

Nothing in this repository constitutes financial or investment advice.