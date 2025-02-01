
import ssl
import datetime
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from pathlib import Path
from typing import Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import logging
from logging.handlers import RotatingFileHandler
import subprocess

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings('ignore')


TRAINING_DATA_DIR = Path("/Users/rohabkashif/psx/PSX_Market_Summary_Playwright")
ACTUAL_DATA_DIR = Path("/Users/rohabkashif/psx/PSX_Market_Summary_Playwright_present")

TRAIN_START_DATE = datetime.date(2024, 6, 1)
TRAIN_END_DATE = datetime.date(2024, 12, 31)

PREDICTION_START_DATE = datetime.date(2025, 1, 1)
PREDICTION_END_DATE = datetime.date(2025, 1, 8)

EVALUATION_START_DATE = PREDICTION_START_DATE
EVALUATION_END_DATE = datetime.date(2025, 1, 8)

DOWNLOAD_DIR = Path("/Users/rohabkashif/psx/PSX_Market_Summary_Playwright")

log_file = 'psx_scraper_playwright.log'
handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
logging.basicConfig(
    handlers=[handler],
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)


def create_directory(path: Path):
    """Creates a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)

def extract_z_file(file_path: Path):
    """
    Extracts a .Z file using the 'uncompress' utility.
    """
    try:
        subprocess.run(['uncompress', str(file_path)], check=True)
        logging.info(f"Successfully extracted {file_path.name}")
    except FileNotFoundError:
        logging.error(
            "The 'uncompress' utility is not found. Please install it using Homebrew: 'brew install gzip'"
        )
        print("The 'uncompress' utility is not found. Please install it using Homebrew: 'brew install gzip'")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting {file_path.name}: {e}")
        print(f"Error extracting {file_path.name}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during extraction: {e}")
        print(f"Unexpected error during extraction: {e}")

def load_data_from_directory(data_dir: Path, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Loads and concatenates all pipe-delimited files from a directory within a date range.
    Assumes all files are pipe-delimited (`|`) text files named as 'YYYY-MM-DD' with optional '.csv' extension.

    Returns:
        A pandas DataFrame containing all concatenated data.
    """
    all_data = pd.DataFrame()

    expected_num_columns = 13

    column_names = [
        'Date', 'Symbol', 'Attribute1', 'Company_Name',
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Additional1', 'Additional2', 'Additional3', 'Additional4'
    ]

    pattern = re.compile(r'^\d{4}-\d{2}-\d{2}(\.csv)?$')

    files = [file for file in data_dir.iterdir() if file.is_file() and pattern.match(file.stem)]

    if not files:
        print("No data files found matching the pattern 'YYYY-MM-DD' or 'YYYY-MM-DD.csv'.")
        logging.warning("No data files found matching the pattern 'YYYY-MM-DD' or 'YYYY-MM-DD.csv'.")

    for file in files:
        try:
            print(f"Processing file: {file}")
            logging.info(f"Processing file: {file}")

            df = pd.read_csv(
                file, 
                sep='|',
                header=None,
                names=column_names,
                engine='python',
                on_bad_lines='skip'
            )

            if df.shape[1] != expected_num_columns:
                logging.error(f"File {file} has {df.shape[1]} columns, expected {expected_num_columns}. Skipping.")
                print(f"File {file} has {df.shape[1]} columns, expected {expected_num_columns}. Skipping.")
                continue

            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d%b%Y').dt.date
            except ValueError:
                try:
                    df['Date'] = pd.to_datetime(df['Date']).dt.date
                except Exception as e:
                    logging.error(f"Date parsing failed for {file}: {e}")
                    print(f"Date parsing failed for {file}: {e}")
                    continue

            df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            print(f"Loaded {len(df_filtered)} records from {file}")
            logging.info(f"Loaded {len(df_filtered)} records from {file}")

            all_data = pd.concat([all_data, df_filtered], ignore_index=True)
        except pd.errors.ParserError as e:
            logging.error(f"Parser error in file {file}: {e}")
            print(f"Parser error in file {file}: {e}")
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            print(f"Error loading {file}: {e}")

    print(f"Total records loaded: {len(all_data)}")
    logging.info(f"Total records loaded: {len(all_data)}")
    return all_data

def moving_average(close_prices: pd.Series, window: int) -> pd.Series:
    """
    Compute the moving average over a specified window.
    """
    return close_prices.rolling(window=window).mean()

def compute_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a given series of close prices.
    """
    delta = close_prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_features(all_data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Prepares features and target for the model.

    Returns:
        all_data (pd.DataFrame): DataFrame with computed features and target.
        unique_symbols (list): List of unique symbols used for training.
    """
    print("Initial data shape:", all_data.shape)
    logging.info(f"Initial data shape: {all_data.shape}")

    all_data['Date'] = pd.to_datetime(all_data['Date'])

    all_data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    print("After dropping rows with missing Open, High, Low, Close, Volume:", all_data.shape)
    logging.info(f"After dropping rows with missing Open, High, Low, Close, Volume: {all_data.shape}")

    all_data = all_data.drop_duplicates(subset=['Symbol', 'Date'])
    print("After removing duplicates:", all_data.shape)
    logging.info(f"After removing duplicates: {all_data.shape}")

    all_data.set_index('Date', inplace=True)
    all_data.sort_index(inplace=True)
    print("After setting Date as index and sorting:", all_data.shape)
    logging.info(f"After setting Date as index and sorting: {all_data.shape}")

    min_data_points = 50  
    symbol_counts = all_data.groupby('Symbol').size()
    symbols_to_keep = symbol_counts[symbol_counts >= min_data_points].index
    all_data = all_data[all_data.index.notnull()]
    all_data = all_data[all_data['Symbol'].isin(symbols_to_keep)]
    print(f"Symbols with at least {min_data_points} data points: {len(symbols_to_keep)}")
    logging.info(f"Symbols with at least {min_data_points} data points: {len(symbols_to_keep)}")

    unique_symbols = symbols_to_keep.tolist()

    all_data['MA_10'] = all_data.groupby('Symbol')['Close'].transform(lambda x: moving_average(x, 10))
    all_data['MA_20'] = all_data.groupby('Symbol')['Close'].transform(lambda x: moving_average(x, 20))
    all_data['RSI'] = all_data.groupby('Symbol')['Close'].transform(lambda x: compute_rsi(x, window=10))

    print("After computing MA_10, MA_20, RSI:", all_data.shape)
    logging.info(f"After computing MA_10, MA_20, RSI: {all_data.shape}")

    all_data.fillna(method='ffill', inplace=True)
    all_data.fillna(method='bfill', inplace=True)

    print("After filling missing values:", all_data.shape)
    logging.info(f"After filling missing values: {all_data.shape}")

    all_data['Target'] = all_data.groupby('Symbol')['Close'].shift(-1)

    all_data.dropna(subset=['Target'], inplace=True)
    print("After dropping rows with NaN Target:", all_data.shape)
    logging.info(f"After dropping rows with NaN Target: {all_data.shape}")

    all_data.reset_index(inplace=True)

    print("Final prepared data shape:", all_data.shape)
    logging.info(f"Final prepared data shape: {all_data.shape}")

    print(f"Date range in training data: {all_data['Date'].min()} to {all_data['Date'].max()}")
    logging.info(f"Date range in training data: {all_data['Date'].min()} to {all_data['Date'].max()}")

    record_counts = all_data['Symbol'].value_counts()
    print("\nRecord counts per symbol (Top 10):")
    print(record_counts.head(10))
    logging.info("Record counts per symbol:")
    logging.info(f"\n{record_counts.head(10)}")

    return all_data, unique_symbols

def get_features_and_target(prepared_data: pd.DataFrame, unique_symbols: list) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Defines features and target from the prepared data.
    Ensures one-hot encoded symbols match the training symbols.

    Returns:
        X (pd.DataFrame): Feature set
        y (pd.Series): Target variable
        feature_columns (list): List of feature column names
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_20', 'RSI']
    target = 'Target'

    missing_features = [f for f in features if f not in prepared_data.columns]
    if missing_features:
        raise KeyError(f"Missing features in prepared data: {missing_features}")

    X = prepared_data[features].copy()
    y = prepared_data[target].copy()

    symbols_encoded = pd.get_dummies(prepared_data['Symbol'], prefix='Sym')

    expected_symbol_columns = [f'Sym_{sym}' for sym in unique_symbols]
    for sym_col in expected_symbol_columns:
        if sym_col not in symbols_encoded.columns:
            symbols_encoded[sym_col] = 0

    symbols_encoded = symbols_encoded[expected_symbol_columns]

    X = pd.concat([X, symbols_encoded], axis=1)

    X.fillna(method='ffill', inplace=True)
    X.fillna(method='bfill', inplace=True)

    feature_columns = X.columns.tolist()

    print("Features and target prepared.")
    logging.info("Features and target prepared.")

    return X, y, feature_columns

def inspect_symbol_data(historical_data: pd.DataFrame, symbol: str):
    """
    Prints the last few records of a specified symbol to verify data integrity.
    """
    symbol_data = historical_data[historical_data['Symbol'] == symbol].copy()
    if symbol_data.empty:
        print(f"No data found for symbol: {symbol}")
        logging.warning(f"No data found for symbol: {symbol}")
        return
    print(f"\nInspecting data for symbol: {symbol}")
    print(symbol_data.tail(10))
    logging.info(f"Inspecting data for symbol: {symbol}")
    logging.info(f"\n{symbol_data.tail(10)}")


def main():
    print("Loading training data...")
    logging.info("Loading training data...")
    training_data = load_data_from_directory(TRAINING_DATA_DIR, TRAIN_START_DATE, TRAIN_END_DATE)
    if training_data.empty:
        print("No training data loaded. Exiting...")
        logging.error("No training data loaded. Exiting...")
        return
    print(f"Training data loaded: {len(training_data)} records.\n")
    logging.info(f"Training data loaded: {len(training_data)} records.")

    print("Extracting unique symbols from training data and preparing features...")
    prepared_data, unique_symbols = prepare_features(training_data)
    print(f"Number of unique symbols used for training: {len(unique_symbols)}")
    logging.info(f"Number of unique symbols used for training: {len(unique_symbols)}")

    print("Defining features and target...")
    X, y, feature_columns = get_features_and_target(prepared_data, unique_symbols)
    if X.empty or y.empty:
        print("Features or target are empty. Exiting...")
        logging.error("Features or target are empty. Exiting...")
        return
    print(f"Features and target prepared: {X.shape}, {y.shape}\n")
    logging.info(f"Features and target prepared: {X.shape}, {y.shape}")

    feature_columns_file = "feature_columns.csv"
    pd.Series(feature_columns).to_csv(feature_columns_file, index=False, header=False)
    logging.info(f"Feature columns saved to {feature_columns_file}")

    print("Training the model...")
    logging.info("Training the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("Model training completed.\n")
    logging.info("Model training completed.")

    print("Preparing prediction data...")
    logging.info("Preparing prediction data...")
    historical_data = training_data.copy()
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data.set_index('Date', inplace=True)
    historical_data.sort_index(inplace=True)

    print(f"Latest date in historical data: {historical_data.index.max()}")
    logging.info(f"Latest date in historical data: {historical_data.index.max()}")

    sample_symbols = unique_symbols[:5]
    for sym in sample_symbols:
        inspect_symbol_data(historical_data, sym)

    prediction_dates = pd.date_range(start=PREDICTION_START_DATE, end=PREDICTION_END_DATE, freq='B')
    predictions = []

    if not Path(feature_columns_file).is_file():
        logging.error(f"Feature columns file '{feature_columns_file}' not found. Exiting.")
        print(f"Feature columns file '{feature_columns_file}' not found. Exiting...")
        return
    feature_columns_list = pd.read_csv(feature_columns_file, header=None).iloc[:, 0].tolist()

    if '0' in feature_columns_list:
        logging.error("Feature columns list contains an unexpected feature '0'. Exiting.")
        print("Feature columns list contains an unexpected feature '0'. Please check 'feature_columns.csv'. Exiting...")
        return

    for date in prediction_dates:
        date_str = date.date()
        logging.info(f"Processing prediction for date: {date_str}")
        print(f"\nProcessing predictions for date: {date_str}")
        symbols_predicted = 0
        symbols_skipped = 0

        test_symbols = unique_symbols 
        for symbol in test_symbols:
            symbol_data = historical_data[historical_data['Symbol'] == symbol].copy()
            if symbol_data.empty:
                symbols_skipped += 1
                print(f"Symbol: {symbol} - No historical data. Skipping.")
                continue
            recent_data = symbol_data.tail(50)
            if len(recent_data) < 50:
                symbols_skipped += 1
                print(f"Symbol: {symbol} - Insufficient data ({len(recent_data)} records). Skipping.")
                continue

            open_price = symbol_data.iloc[-1]['Open']
            high_price = symbol_data.iloc[-1]['High']
            low_price = symbol_data.iloc[-1]['Low']
            close_price = symbol_data.iloc[-1]['Close']
            volume = symbol_data.iloc[-1]['Volume']
            ma_10 = moving_average(symbol_data['Close'], 10).iloc[-1]
            ma_20 = moving_average(symbol_data['Close'], 20).iloc[-1]
            rsi = compute_rsi(symbol_data['Close'], window=10).iloc[-1]

            if pd.isna(ma_10) or pd.isna(ma_20) or pd.isna(rsi):
                symbols_skipped += 1
                print(f"Symbol: {symbol} - Computed features have NaN. Skipping.")
                continue

            print(f"Symbol: {symbol}, Date: {date_str}, MA_10: {ma_10}, MA_20: {ma_20}, RSI: {rsi}")

            feature_vector = {
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'MA_10': ma_10,
                'MA_20': ma_20,
                'RSI': rsi
            }

            symbol_encoded = {f'Sym_{sym}': 0 for sym in unique_symbols}
            symbol_encoded[f'Sym_{symbol}'] = 1

            feature_vector.update(symbol_encoded)
            feature_df = pd.DataFrame([feature_vector])

            for col in feature_columns_list:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[feature_columns_list]

            if feature_df.isnull().any().any():
                symbols_skipped += 1
                print(f"Symbol: {symbol} - Features contain NaN. Skipping.")
                continue

            try:
                predicted_close = model.predict(feature_df)[0]
                print(f"Symbol: {symbol} - Predicted Close: {predicted_close}")
            except Exception as e:
                symbols_skipped += 1
                print(f"Symbol: {symbol} - Prediction failed: {e}")
                continue

            predictions.append({
                'Date': date_str,
                'Symbol': symbol,
                'Predicted_Close': predicted_close
            })
            symbols_predicted += 1

            new_record = {
                'Date': pd.to_datetime(date_str),
                'Symbol': symbol,
                'Open': close_price,
                'High': close_price,
                'Low': close_price,
                'Close': predicted_close,
                'Volume': volume,
                'Attribute1': symbol_data.iloc[-1]['Attribute1'],
                'Company_Name': symbol_data.iloc[-1]['Company_Name'],
                'Additional1': symbol_data.iloc[-1]['Additional1'],
                'Additional2': symbol_data.iloc[-1]['Additional2'],
                'Additional3': symbol_data.iloc[-1]['Additional3'],
                'Additional4': symbol_data.iloc[-1]['Additional4']
            }
            new_row = pd.Series(new_record)
            historical_data = pd.concat([historical_data, pd.DataFrame([new_row]).set_index('Date')])

        print(f"Date: {date_str} - Symbols Processed: {symbols_predicted}, Skipped: {symbols_skipped}")

    prediction_df = pd.DataFrame(predictions)
    print(f"Predictions made for {len(prediction_df)} records.\n")
    logging.info(f"Predictions made for {len(prediction_df)} records.")

    if prediction_df.empty:
        print("No predictions were made. Exiting...")
        logging.error("No predictions were made. Exiting...")
        return

    print("Loading actual data for evaluation...")
    logging.info("Loading actual data for evaluation...")
    actual_data = load_data_from_directory(ACTUAL_DATA_DIR, EVALUATION_START_DATE, EVALUATION_END_DATE)
    if actual_data.empty:
        print("No actual data loaded for evaluation. Exiting...")
        logging.error("No actual data loaded for evaluation. Exiting...")
        return
    print(f"Actual data loaded: {len(actual_data)} records.\n")
    logging.info(f"Actual data loaded: {len(actual_data)} records.")

    print("Preparing actual data for comparison...")
    logging.info("Preparing actual data for comparison...")
    actual_data['Date'] = pd.to_datetime(actual_data['Date']).dt.date
    actual_data = actual_data[['Date', 'Symbol', 'Close']].copy()
    actual_data.rename(columns={'Close': 'Actual_Close'}, inplace=True)

    print("Merging predictions with actual data...")
    logging.info("Merging predictions with actual data...")
    comparison_df = pd.merge(prediction_df, actual_data, on=['Date', 'Symbol'], how='inner')

    if comparison_df.empty:
        print("No overlapping data between predictions and actual data. Exiting...")
        logging.error("No overlapping data between predictions and actual data. Exiting...")
        return

    print("Evaluating predictions...")
    logging.info("Evaluating predictions...")
    mse = mean_squared_error(comparison_df['Actual_Close'], comparison_df['Predicted_Close'])
    print(f"Mean Squared Error on Evaluation Set: {mse:.2f}")
    logging.info(f"Mean Squared Error on Evaluation Set: {mse:.2f}")

    sectors = {
        'OIL & GAS EXPLORATION COMPANIES': ['OGDC', 'PPL', 'POL', 'MARI'],
        'COMMERCIAL BANKS': ['BOP', 'NBP', 'MEBL', 'BAFL', 'HBL', 'UBL', 'MCB', 'BAHL'],
        'CEMENT': ['FCCL', 'MLCF', 'DGKC', 'LUCK', 'CHCC'],
        'FERTILISER': ['EFERT', 'FFC', 'ENGRO'],
        'POWER GENERATION & DISTRIBUTION': ['HUBC'],
        'FOOD & PERSONAL CARE PRODUCTS': ['UNITY'],
        'OIL & GAS MARKETING COMPANIES': ['HASCOL', 'SNGP', 'PSO'],
        'CABLE & ELECTRICAL GOODS': ['PAEL'],
        'TECHNOLOGY & COMMUNICATION': ['TRG'],
        'ENGINEERING': ['ISL'],
        'PHARMACEUTICALS': ['SEARL'],
        'TEXTILE COMPOSITE': ['NML']
    }

    print("Plotting Actual vs Predicted Stock Prices by sector...")
    logging.info("Plotting Actual vs Predicted Stock Prices by sector...")

    for sector_name, sector_symbols in sectors.items():
        sector_data = comparison_df[comparison_df['Symbol'].isin(sector_symbols)]
        if sector_data.empty:
            logging.info(f"No data to plot for sector: {sector_name}")
            continue

        plt.figure(figsize=(12, 6))
        for symbol in sector_symbols:
            symbol_df = sector_data[sector_data['Symbol'] == symbol].copy()
            if symbol_df.empty:
                continue
            symbol_df.sort_values(by='Date', inplace=True)
            plt.plot(symbol_df['Date'], symbol_df['Actual_Close'], label=f"{symbol} Actual", marker='o')
            plt.plot(symbol_df['Date'], symbol_df['Predicted_Close'], label=f"{symbol} Predicted", marker='x')

        plt.title(f"{sector_name} - Actual vs Predicted Close Prices")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    comparison_df.to_csv(DOWNLOAD_DIR / "predictions_vs_actual.csv", index=False)
    print("Comparison results saved to 'predictions_vs_actual.csv'.")
    logging.info("Comparison results saved to 'predictions_vs_actual.csv'.")

if __name__ == '__main__':
    main()
