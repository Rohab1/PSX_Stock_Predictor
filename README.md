
# PSX Stock Predictor and Investment Advisor

## Overview

The **PSX Stock Predictor and Investment Advisor** project is a comprehensive tool designed to:
- Scrape market summary data from the Pakistan Stock Exchange (PSX) using Playwright.
- Process and clean the data, compute technical indicators (e.g., moving averages, RSI).
- Train a Random Forest regression model to predict future stock prices.
- Evaluate model performance using metrics such as Mean Squared Error (MSE).
- Provide investment recommendations by calculating historical volatility and a risk-adjusted return metric to identify stocks with the highest predicted returns while maintaining relatively low risk.
- Visualize actual versus predicted stock prices across various sectors.

> **Disclaimer:** This project is for personal use only and is not intended for distribution or commercial purposes. All code and data are provided "as-is" for educational and personal experimentation. No copyright claims are asserted on this work.

## Features

- **Data Scraping:**  
  Uses Playwright (async) to navigate to the PSX downloads page, set the date dynamically, and download compressed market summary data.

- **Data Extraction:**  
  Downloads and extracts `.Z` compressed files using the `uncompress` utility.

- **Data Preparation:**  
  Processes raw data, computes technical indicators (10-day MA, 20-day MA, RSI), and prepares features and targets for machine learning.

- **Machine Learning:**  
  Trains a RandomForestRegressor model to predict future stock closing prices based on historical data.

- **Investment Recommendations:**  
  Calculates historical volatility and computes a risk-adjusted return (similar to the Sharpe Ratio) to recommend the top high-return, low-risk stocks.

- **Evaluation and Visualization:**  
  Evaluates predictions using Mean Squared Error (MSE) and plots actual versus predicted stock prices by sector.

- **Logging:**  
  Logs key events and errors to help with troubleshooting and auditing.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- [Playwright](https://playwright.dev/python/docs/intro)  
  *(After installing Playwright, run `playwright install` to download necessary browser binaries.)*

### Required Python Libraries

Install the necessary libraries using pip. It is recommended to create a virtual environment first.

1. Create and activate a virtual environment:


   python -m venv env
   # On macOS/Linux:
   source env/bin/activate
   # On Windows:
   env\Scripts\activate

2. Install required packages:


   pip install asyncio playwright requests tqdm matplotlib numpy pandas scikit-learn


### Virtual Environment

Ensure your virtual environment folder (typically named `env/`) is added to your `.gitignore` so that large dependency files are not committed to your repository.

## Usage

### Data Scraping

The scraper script (e.g., `your_scraper_script.py`) uses Playwright to download PSX market summary data for a specified date range.  
To run the scraper:


python your_scraper_script.py


Adjust the `START_DATE` and `END_DATE` variables in the script as needed.

### Investment Advisor

The main script, `InvestmentAdvisor.py`, processes the scraped data, trains a RandomForestRegressor model, and provides investment recommendations.

To run the Investment Advisor:


python InvestmentAdvisor.py


You will be prompted to enter:
- **Training Start Date** (e.g., `2024-06-01`)
- **Training End Date** (e.g., `2024-12-31`)
- **Prediction End Date** (e.g., `2025-01-08`)

The script will then:
- Train the model on the historical data.
- Generate predictions for the specified prediction period.
- Evaluate predictions using Mean Squared Error (MSE).
- Calculate volatility and compute a risk-adjusted return metric.
- Recommend the top high-return, low-risk stocks (saved as `best_high_return_low_risk_stocks.csv`).
- Plot sector-wise Actual vs. Predicted stock prices.
- Save merged predictions and actual data to `predictions_vs_actual.csv`.

## Project Structure


PSX_Stock_Predictor/
├── env/                        # Virtual environment (should be excluded from version control)
├── PSX_Market_Summary_Playwright/         # Folder for scraped data files
├── PSX_Market_Summary_Playwright_present/   # Folder for actual market summary data
├── InvestmentAdvisor.py        # Main script for predictions and investment recommendations
├── your_scraper_script.py      # Scraper script using Playwright
├── psx_scraper_playwright.log  # Log file
├── feature_columns.csv         # Generated file containing feature columns
└── README.md                   # This file


> **Note:** Ensure that your virtual environment (`env/`) is added to `.gitignore` so that large dependency files are not pushed to GitHub.

## Contributing

Contributions are welcome! Since this project is for personal use only, please note that it is not intended for distribution or commercial purposes. Feel free to fork the repository and create pull requests with your proposed changes. Ensure your changes follow the project's coding style and include appropriate tests.

## License

This project is for personal use only and is not intended for distribution. No commercial license is granted, and it is provided "as-is" for educational and personal experimentation purposes.

## Contact

For questions or suggestions, please contact Rohab Kashif at [Rohabkashif123@gmail.com](mailto:Rohabkashif123@gmail.com).


---

Feel free to modify any sections or text as needed. Save this as `README.md` in your project root and commit it to your repository.
