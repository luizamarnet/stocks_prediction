
"""Prepare market data before running the main application."""

import argparse
import logging
from pathlib import Path

from src.stocks_prediction.download_public import download_market_data
from src.stocks_prediction.data_preprocessing import preprocess_and_split


ROOT_DIR = Path(__file__).parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw_data"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"

PRICE_MOMENTS = ["Open", "Close", "High", "Low", "Volume"]

DEFAULT_TRAIN_RATIO = 0.8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_data(
    start_date: str | None = None,
    end_date: str | None = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    skip_download: bool = False,
) -> None:
    """Download and preprocess market data."""

    market: str = "Copenhagen"
        
    market = market.upper()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = RAW_DATA_DIR / f"{market.lower()}_market_data.csv"

    # 1. Download market data
    if skip_download:
        if not raw_file.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {raw_file}"
            )

        logger.info("Using existing raw market data: %s", raw_file)

    else:
        logger.info("Downloading %s market data...", market)

        raw_data = download_market_data(
           # market=market,
            start=start_date,
            end=end_date,
            save_downloaded_data=False,
        )

        if raw_data.empty:
            raise RuntimeError("Downloaded market data is empty.")

        raw_data.to_csv(raw_file, index=False)

        logger.info(
            "Downloaded %d rows and saved to %s",
            len(raw_data),
            raw_file,
        )

    # 2. Preprocess and split the data
    for price_moment in PRICE_MOMENTS:

        logger.info("Preprocessing %s prices...", price_moment)

        train_data, test_data, _ = preprocess_and_split(
            file_path=str(raw_file),
            price_moment=price_moment,
            train_ratio=train_ratio,
            validation_ratio=None,
            min_observations=100,
            max_missing_pct=0.05,
            save_data=True,
            output_dir=str(PROCESSED_DATA_DIR),
        )

        if train_data.empty or test_data.empty:
            raise RuntimeError(
                f"Preprocessing produced an empty dataset for {price_moment}."
            )

        logger.info(
            "%s: train=%s, test=%s",
            price_moment,
            train_data.shape,
            test_data.shape,
        )

        logger.info(
            "Training period: %s to %s",
            train_data.index.min().date(),
            train_data.index.max().date(),
        )

        logger.info(
            "Testing period: %s to %s",
            test_data.index.min().date(),
            test_data.index.max().date(),
        )

    logger.info("Data setup completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download and prepare stock market data."
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Historical data start date (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Historical data end date (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Proportion of data allocated to training.",
    )

    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Reuse existing raw data instead of downloading again.",
    )

    args = parser.parse_args()

    setup_data(
        start_date=args.start_date,
        end_date=args.end_date,
        train_ratio=args.train_ratio,
        skip_download=args.skip_download,
    )
