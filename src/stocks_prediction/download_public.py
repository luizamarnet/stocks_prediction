import pandas as pd
import yfinance as yf
from time import sleep
import requests
from io import StringIO
from typing import List, Optional
import logging
import argparse
from datetime import date, datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default tickers for US market
US_DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "JPM", "XOM", "JNJ", "PG",
    "HD", "UNH", "V", "MA", "COST"
]

# Market-specific configurations
MARKET_CONFIGS = {
    "US": {
        "ticker_suffix": "",
        "default_tickers": {
            "value": US_DEFAULT_TICKERS,
            "source": "defined_by_developer",
        },
        "url": {
            "value": "https://stockanalysis.com/list/nasdaq-stocks/",
            "source": "StockAnalysis",
        },
    },
    "COPENHAGEN": {
        "ticker_suffix": ".CO",
        "source": "stockanalysis",
        "url": "https://stockanalysis.com/list/copenhagen-stock-exchange/"
    }
}


def get_market_tickers(market: str = "US") -> List[str]:
    """
    Get ticker list for a specific market.
    
    Args:
        market: Market identifier (e.g., "US", "COPENHAGEN")
    
    Returns:
        List of ticker symbols
    """
    market = market.upper()
    if market not in MARKET_CONFIGS:
        raise ValueError(f"Unsupported market: {market}. Available: {list(MARKET_CONFIGS.keys())}")
    
    config = MARKET_CONFIGS[market]
    
    if market == "US":
        return _scrape_tickers(config["url"]["value"])
    elif market == "COPENHAGEN":
        return _scrape_tickers(config["url"])
    else:
        raise ValueError(f"Market '{market}' is listed but was not implemented yet.")


def _scrape_tickers(url: str) -> List[str]:
    """
    Scrape ticker symbols from stockanalysis.com for market exchange.
    
    Args:
        url: URL to scrape
    
    Returns:
        List of ticker symbols
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            logger.warning("No ticker tables found in response")
            return []
        
        df = tables[0]
        
        # Find ticker column
        ticker_col_candidates = [
            c for c in df.columns
            if any(keyword in c.lower() for keyword in ["symbol", "ticker", "code"])
        ]
        
        if not ticker_col_candidates:
            logger.warning("No ticker column found in table")
            return []
        
        ticker_col = ticker_col_candidates[0]
        
        # Extract and clean tickers
        tickers_raw = df[ticker_col].dropna().unique().tolist()
        tickers = [
            t.strip() for t in tickers_raw 
            if isinstance(t, str) and t.strip()
        ]
        
        logger.info(f"Found {len(tickers)} tickers from chosen market exchange")
        return tickers
        
    except Exception as e:
        logger.error(f"Failed to scrape market tickers: {e}")
        return []


def format_ticker(ticker: str, market: str) -> str:
    """
    Format ticker symbol according to market conventions.
    
    Args:
        ticker: Raw ticker symbol
        market: Market identifier
    
    Returns:
        Formatted ticker symbol
    """
    market = market.upper()
    t = ticker.strip()
    
    if market == "COPENHAGEN":
        # Remove existing suffixes
        for suffix in [".CO", ".XCSE", ".CPH"]:
            if t.upper().endswith(suffix):
                t = t[:-len(suffix)]
        # Normalize and add Copenhagen suffix
        t = t.replace(" ", "-").replace(".", "-")
        return f"{t.upper()}.CO"
    else:
        return t.upper()


def download_data(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    market: str = "US",
    batch_size: int = 40,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> pd.DataFrame:
    """
    Download historical price data for given tickers.
    
    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), defaults to today
        market: Market identifier
        batch_size: Number of tickers per batch
        max_retries: Maximum retry attempts per batch
        retry_delay: Delay between retries in seconds
    
    Returns:
        DataFrame with multi-index columns (ticker, OHLCV fields)
    """
    # Format tickers according to market
    formatted_tickers = list(dict.fromkeys(
        format_ticker(t, market) for t in tickers
    ))
    
    all_data = []
    
    for i in range(0, len(formatted_tickers), batch_size):
        batch = formatted_tickers[i:i + batch_size]
        logger.info(f"Downloading batch {i//batch_size + 1}/{(len(formatted_tickers) + batch_size - 1)//batch_size}: {len(batch)} tickers")

        download_kwargs = {
            "tickers": batch,
            "auto_adjust": True,
            "progress": False,
            "group_by": "ticker",
            "interval": "1d",
            "threads": True,
        }

        if start is None and end is None:
            download_kwargs["period"] = "max"

        elif start is not None:
            download_kwargs["start"] = start
            download_kwargs["end"] = end or date.today().strftime("%Y-%m-%d")

        else:
            download_kwargs["period"] = "max"
        
        for attempt in range(max_retries):
            try:
                data = yf.download(**download_kwargs)
                
                if data.empty:
                    logger.warning(f"Empty data for batch {batch}")
                    break
                
                all_data.append(data)
                break
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for batch {batch}: {e}")
                if attempt < max_retries - 1:
                    sleep(retry_delay)
                else:
                    logger.error(f"Failed to download batch {batch} after {max_retries} attempts")

        if start is None and end is not None and not all_data.empty:
            all_data = all_data.loc[all_data.index < end]
    
    if not all_data:
        raise RuntimeError("No data was downloaded successfully")
    
    # Combine all batches
    combined_data = pd.concat(all_data, axis=1)
    
    # Remove any all-NaN columns (delisted tickers)
    combined_data = combined_data.dropna(axis=1, how='all')

    logger.info(f"Successfully downloaded data for {len(combined_data.columns.get_level_values(0).unique())} tickers")
    return combined_data


def download_market_data(
    #market: str = "US",
    start: str = "2020-01-01",
    end: Optional[str] = None,
    batch_size: int = 40,
    save_downloaded_data: bool = False
) -> pd.DataFrame:
    """
    Main function to download market data for a specified market.
    
    Args:
        market: Market identifier ("US", "COPENHAGEN")
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        batch_size: Number of tickers per download batch
        save_downloaded_data: Whether to save the downloaded data to CSV.
    
    Returns:
        DataFrame with price data
    """
    market: str = "Copenhagen"
    # Get tickers for the market
    tickers = get_market_tickers(market)
    logger.info(f"Downloading data for {len(tickers)} tickers from {market} market")
    
    # Download price data
    data = download_data(
        tickers=tickers,
        start=start,
        end=end,
        market=market,
        batch_size=batch_size
    )

    if isinstance(data.columns, pd.MultiIndex):
        data = (
            data.stack(level=0, future_stack=True)
            .rename_axis(index=["Date", "ticker"])
            .reset_index()
        )

    # Save to CSV if save_downloaded_data is True
    if save_downloaded_data:
        # Generate timestamp formatted as YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Dynamically generate output path with timestamp
        output_path = f"{market.lower()}_market_data_{timestamp}"
        data.to_csv(f"data/raw_data/{output_path}.csv", index=True)
        logger.info(f"Saved data ({len(data)} rows) to {output_path}.csv")
        
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical market data.",
                                     formatter_class=lambda prog: argparse.HelpFormatter(
                                        prog,
                                        max_help_position=35,
                                        width=100
                                    ))
    '''parser.add_argument(
        "--market", 
        type=str, 
        default="US", 
        help="Market to download (e.g., copenhagen, US, OMXC20)"
    )'''
    parser.add_argument("--start_date", type=str)#, default="2020-01-01")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the downloaded data (Default = False)"
    )

    args = parser.parse_args()

    # Normalize market name to uppercase
    market = "Copenhagen" #args.market.upper()

    data = download_market_data(
     #   market=market,
        start=args.start_date,
        save_downloaded_data=args.save,
    )
    print(f"Downloaded data shape: {data.shape}")
