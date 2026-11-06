"""Trade simulation for the model-driven strategy and the buy & hold benchmark."""
import numpy as np
import pandas as pd

PROBABILITY_THRESHOLD = 0.50    # minimum classifier probability to consider a buy
TRAILING_STOP_RATIO = 0.95      # exit when price falls below 95% of the running maximum


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
    profit: float = 30.00,
    label_type: str = "percentage",

) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    """Simulate the strategy and the buy & hold benchmark over the prediction period.

    Entry rule: classifier probability above the threshold AND predicted profit > 0.
    Exit rule: trailing stop (price below 95% of the running maximum), or
    the signal turning off, or the last day of the period.

    Args:
        predictions: Predictions indexed by date with columns
            ``classe_predict_proba``, ``retorno_previsto`` and ``price_test``.
        stock_name: Name of the stock (informational).
        budget: Initial capital.
        buy_sell_fee: Fixed fee per buy and sell.
        commission_bps: Proportional commission in basis points.

    Returns:
        Tuple of (cumulative strategy PnL, strategy PnL as a fraction of budget,
        cumulative benchmark PnL, benchmark PnL as a fraction of budget).
    """
    pnl_total = 0.0
    open_order = False
    pnl_test: list[float] = []
    new_budget = budget
    n_buyings = 0
    n_days_holding = 0
    entry_price_0 = float(predictions.iloc[0]["price_test"])
    n_stocks_0 = budget // entry_price_0
    entry_fee_0 = buy_sell_fee + (commission_bps / 10000.0) * (n_stocks_0 * entry_price_0)
    pnl_test_no_code: list[float] = []

    for i in range(len(predictions)):
        results = predictions.iloc[i]

        prob_preds = results["classe_predict_proba"]
        test_preds = (prob_preds > PROBABILITY_THRESHOLD).astype(int)

        if (label_type == "percentage"):
            pred_percent = results["retorno_previsto"]
        price_test = results["price_test"]

        # Candidate entry: size the position with the current budget and estimate its profit
        entry_price_aux = float(price_test)
        n_stocks_aux = new_budget // entry_price_aux
        entry_fee_aux = buy_sell_fee + (commission_bps / 10000.0) * (n_stocks_aux * entry_price_aux)
        if (label_type == "percentage"):
            pred_exit_price = pred_percent * entry_price_aux + entry_price_aux
            pred_exit_fee = buy_sell_fee + (commission_bps / 10000.0) * (n_stocks_aux * pred_exit_price)
            pred_profit = n_stocks_aux * (pred_exit_price - entry_price_aux) - entry_fee_aux - pred_exit_fee
            pred_profit_signal = pred_profit > 0        
        else:
            pred_profit_signal = True


        if (test_preds == 1) and (i < (len(predictions) - 1)) and (pred_profit_signal):
            if not open_order:
                # Open a new position
                entry_price_i = entry_price_aux
                n_stocks_i = n_stocks_aux
                entry_fee_i = entry_fee_aux

                open_order = True
                buying_day = predictions.index[i]
                max_price = entry_price_i
                pnl_test.append(0.0)
                n_buyings += 1
            else:
                '''# Position is open: check the trailing stop and update the running maximum
                if float(price_test) < max_price * TRAILING_STOP_RATIO:
                    exit_price_i = float(price_test)
                    pnl_trade = _calculate_trade_pnl(
                        n_stocks_i, entry_price_i, entry_fee_i,
                        exit_price_i, buy_sell_fee, commission_bps,
                    )
                    pnl_total += float(pnl_trade)
                    pnl_test.append(pnl_trade)
                    new_budget += pnl_trade
                    n_days_holding += (predictions.index[i] - buying_day).days
                    open_order = False
                elif float(price_test) > max_price:
                    max_price = float(price_test)
                    pnl_test.append(0.0)
                else:
                    pnl_test.append(0.0)'''
                pnl_test.append(0.0)
        else:
            if open_order:
                # Signal is off (or last day): close the open position
                exit_price_i = float(price_test)
                pnl_trade = _calculate_trade_pnl(
                    n_stocks_i, entry_price_i, entry_fee_i,
                    exit_price_i, buy_sell_fee, commission_bps,
                )
                pnl_total += float(pnl_trade)
                pnl_test.append(pnl_trade)
                new_budget += pnl_trade
                n_days_holding += (predictions.index[i] - buying_day).days
                open_order = False
            else:
                pnl_test.append(0.0)

        # Buy & hold benchmark: buy on the first day and hold
        if i == 0:
            pnl_test_no_code.append(0.0)
        else:
            exit_price_i = float(price_test)
            pnl_test_no_code_i = _calculate_trade_pnl(
                n_stocks_0, entry_price_0, entry_fee_0,
                exit_price_i, buy_sell_fee, commission_bps,
            )
            pnl_test_no_code.append(pnl_test_no_code_i)

    pnl_test = np.cumsum(pnl_test)
    pnl_test_no_code_percent = [pnl / budget for pnl in pnl_test_no_code]
    pnl_test_percent = [pnl / budget for pnl in pnl_test]

    print(f"final budget is equal to: {new_budget}")
    print(f"Net pnl test: {pnl_total:.2f}")
    net_percent = pnl_total / budget
    print(f"Net pnl test percent: {net_percent:.2f}")
    print(f"trading days: {n_days_holding}")
    print(f"number of buying actions {n_buyings}")

    # Benchmark summary over the full period
    n_stocks_i = budget // entry_price_0
    last_price = float(predictions.iloc[-1]["price_test"])
    no_code_profit = _calculate_trade_pnl(
        n_stocks_i, entry_price_0,
        buy_sell_fee + (commission_bps / 10000.0) * (n_stocks_i * entry_price_0),
        last_price, buy_sell_fee, commission_bps,
    )
    print(f"no code profit: {no_code_profit}")
    print(f"no code percent profit: {no_code_profit / budget}")

    return pnl_test, pnl_test_percent, pnl_test_no_code, pnl_test_no_code_percent
