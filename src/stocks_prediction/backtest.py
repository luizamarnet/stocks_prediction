"""Trade simulation for the model-driven strategy and the buy & hold benchmark."""
import numpy as np
import pandas as pd

PROBABILITY_THRESHOLD = 0.50    # minimum classifier probability to consider a buy



def _calculate_trade_pnl(
    n_stocks: float,
    entry_price: float,
    entry_fee: float,
    exit_price: float,
    buy_sell_fee: float,
    commission_bps: float,
) -> float:
    """Compute the net PnL of a closed position.

    Args:
        n_stocks: Number of shares held.
        entry_price: Price paid per share.
        entry_fee: Total fee paid when opening the position.
        exit_price: Price received per share.
        buy_sell_fee: Fixed fee charged on the sell.
        commission_bps: Proportional commission in basis points.

    Returns:
        Net profit or loss after fees.
    """
    exit_fee = buy_sell_fee + (commission_bps / 10000.0) * (n_stocks * exit_price)
    return n_stocks * (exit_price - entry_price) - entry_fee - exit_fee


def backtest(
    predictions: pd.DataFrame,
    stock_name: str,
    budget: float,
    buy_sell_fee: float,
    commission_bps: float,
    label_type: str = "percentage",
    regressor_flag: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Simulate the strategy and the buy & hold benchmark over the prediction period.

    Entry rule:
        Classifier probability above the threshold.
        When a percentage regressor is enabled, predicted net profit must also be positive.

    Exit rule:
        Classifier signal turns off, predicted net profit becomes non-positive
        (when the regressor is enabled), or the final prediction day is reached.

    Args:
        predictions: Predictions indexed by date with columns
            ``classe_predict_proba``, ``retorno_previsto`` and ``price_test``.
        stock_name: Name of the stock (informational).
        budget: Initial capital.
        buy_sell_fee: Fixed fee per buy and sell.
        commission_bps: Proportional commission in basis points.

    Returns:
        Tuple containing:
            - Daily PnL given by the code strategy.
            - Daily percentage return given by the code strategy as decimal.
            - Daily buy-and-hold PnL.
            - Daily buy-and-hold percentage return as decimal.
    """

    if predictions.empty:
        raise ValueError("predictions cannot be empty")

    if label_type not in ("profit", "percentage"):
        raise ValueError(
            f"Invalid label_type: {label_type!r}. "
            "Please, label_type must be either 'profit' or 'percentage'."
        )

    predictions = predictions.sort_index()
    assert predictions.index.is_monotonic_increasing, (
        "predictions must be sorted chronologically"
    )

    commission_rate = commission_bps / 10_000.0


    pnl_total = 0.0
    open_order = False
    pnl_test: list[float] = []
    new_budget = budget
    pnl_test_no_code: list[float] = []
    n_buyings = 0
    n_days_holding = 0
    benchmark_buying_i = -1
    benchmark_open = False

    for i in range(len(predictions)):
        results = predictions.iloc[i]

        prob_preds = results["classe_predict_proba"]
        test_preds = int(prob_preds > PROBABILITY_THRESHOLD)

        if (label_type == "percentage") and (regressor_flag):
            pred_percent = results["retorno_previsto"]
        price_test = results["price_test"]

        # Candidate entry: size the position with the current budget and estimate its profit
        entry_price_aux = float(price_test)

        if (not benchmark_open) and (i < len(predictions) - 1):
            benchmark_n_stocks = int((budget - buy_sell_fee) // (entry_price_aux * (1 + commission_rate))  )
            if benchmark_n_stocks>0:
                entry_price_0 = entry_price_aux
                entry_fee_0 = buy_sell_fee + commission_rate * (benchmark_n_stocks * entry_price_0)
                assert budget >= ((benchmark_n_stocks*entry_price_0)+entry_fee_0)
                benchmark_open = True
                benchmark_buying_i = i


        if not open_order:
            n_stocks_aux =   int((new_budget - buy_sell_fee) // (entry_price_aux * (1 + commission_rate))  )
            if n_stocks_aux > 0:
                entry_fee_aux = buy_sell_fee + commission_rate * (n_stocks_aux * entry_price_aux)
                assert new_budget >= ((n_stocks_aux*entry_price_aux)+entry_fee_aux)
            else:
                entry_fee_aux = 0.0

        if (label_type == "percentage") and (regressor_flag):
            pred_exit_price = (pred_percent + 1) * entry_price_aux 
            if not open_order:
                if n_stocks_aux>0:
                    pred_exit_fee = buy_sell_fee + commission_rate * (n_stocks_aux * pred_exit_price)
                    pred_profit = n_stocks_aux * (pred_exit_price - entry_price_aux) - entry_fee_aux - pred_exit_fee
                    predicted_profit_is_positive = pred_profit > 0 
                else:
                    predicted_profit_is_positive = False
            else:
                pred_exit_fee = buy_sell_fee + commission_rate * (n_stocks_i * pred_exit_price)
                pred_profit = n_stocks_i * (pred_exit_price - entry_price_i) - entry_fee_i - pred_exit_fee
                predicted_profit_is_positive = pred_profit > 0 
                   
        else:
            predicted_profit_is_positive = True


        if open_order:

            if (test_preds == 0) or (not predicted_profit_is_positive) or (i == len(predictions) - 1):
                # Signal is off (or last day): close the open position
                exit_price_i = float(price_test)
                pnl_trade = _calculate_trade_pnl(
                    n_stocks_i, entry_price_i, entry_fee_i,
                    exit_price_i, buy_sell_fee, commission_bps,
                )
                pnl_total += float(pnl_trade)
                new_budget += pnl_trade
                n_days_holding += i - buying_i
                open_order = False

        else:

            if (n_stocks_aux > 0) and (test_preds == 1) and (i < (len(predictions) - 1)) and (predicted_profit_is_positive):
            
                # Open a new position
                entry_price_i = entry_price_aux
                n_stocks_i = n_stocks_aux
                entry_fee_i = entry_fee_aux

                open_order = True
                buying_i = i
                n_buyings += 1
            

                

        # Buy & hold benchmark: buy on the first affordable day and hold
        if i == 0:
            pnl_test_no_code.append(0.0)
            pnl_test.append(0.0)
        else:
            if benchmark_open:
                if i == benchmark_buying_i:
                    pnl_test_no_code.append(0.0) 
                else:
                    exit_price_i = float(price_test)
                    pnl_test_no_code_i = _calculate_trade_pnl(
                        benchmark_n_stocks, entry_price_0, entry_fee_0,
                        exit_price_i, buy_sell_fee, commission_bps,
                    )
                    pnl_test_no_code.append(pnl_test_no_code_i)
                       
            else:
                pnl_test_no_code.append(0.0)

    
            current_pnl = pnl_total
            if open_order and (i != buying_i):
                current_pnl += _calculate_trade_pnl(
                    n_stocks_i,
                    entry_price_i,
                    entry_fee_i,
                    float(price_test),
                    buy_sell_fee,
                    commission_bps,
                )
            pnl_test.append(current_pnl)


    # Cumulative PnL
    pnl_test = pd.Series(
        pnl_test,
        index=predictions.index,
        name=stock_name,
    )
    # Daily PnL
    daily_pnl_strategy = pnl_test.diff().fillna(pnl_test.iloc[0])
    capital_strategy = budget +  np.array(pnl_test)
    daily_return_strategy_percent = pd.Series(capital_strategy, index=predictions.index, name=stock_name).pct_change().fillna(0)

     


    capital_buy_hold = budget +  np.array(pnl_test_no_code)

    daily_return_buy_hold_percent = pd.Series(capital_buy_hold, index=predictions.index, name=stock_name).pct_change().fillna(0)

    daily_pnl_buy_hold = pd.Series(pnl_test_no_code, index=predictions.index, name=stock_name).diff().fillna(pnl_test_no_code[0])

    #print(f"final budget is equal to: {new_budget}")
    #print(f"Net pnl test: {pnl_total:.2f}")
    
    # Benchmark summary over the full period
    last_price = float(predictions.iloc[-1]["price_test"])
    if benchmark_open:
        no_code_profit = _calculate_trade_pnl(
            benchmark_n_stocks, entry_price_0, entry_fee_0,
            last_price, buy_sell_fee, commission_bps,
        )
    else:
        no_code_profit = 0
    #print(f"no code profit: {no_code_profit}")
    net_percent = pnl_total / budget
    print(f"Net pnl test percent: {net_percent:.2%}")
    print(f"no code percent profit: {no_code_profit / budget:0.2%}")
    print(f"trading days: {n_days_holding}")
    print(f"number of buying actions {n_buyings}")
    
    assert np.isclose(no_code_profit, pnl_test_no_code[-1])
    assert np.isclose(pnl_total, pnl_test.iloc[-1])

    return daily_pnl_strategy, daily_return_strategy_percent, daily_pnl_buy_hold, daily_return_buy_hold_percent
