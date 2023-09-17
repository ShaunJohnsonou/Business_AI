import yfinance as yf
import datetime
def download_Business_info_csv(symbol):
    # Define the start and end date for the historical data
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5*365)  # Assuming 1 year = 365 days

    # Fetch historical data for Microsoft
    data = yf.download(symbol, start=start_date, end=end_date)

    # Save the data to a CSV file
    data.to_csv(f'Business_AI/test_data/{symbol}_historical_data.csv')

    print(f"Data extraction complete. The historical stock data for {symbol} has been saved to '{symbol}_historical_data.csv'.")
download_Business_info_csv(symbol='SOL.JO')