import pandas as pd
import matplotlib.pyplot  as plt
import os

current_dir = os.getcwd()
# Load the historical stock data
df = pd.read_csv(f'{current_dir}\\test_data\\moving_average_test_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define the short-term and long-term moving average windows
short_window = 50  # Adjust this as needed
long_window = 200  # Adjust this as needed

# Calculate the short-term and long-term moving averages
df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()

# Generate buy/sell signals based on the moving average crossover
df['Signal'] = 0  # Initialize the signal column with zeros
df.loc[df['Short_MA'] > df['Long_MA'], 'Signal'] = 1  # Buy signal
df.loc[df['Short_MA'] < df['Long_MA'], 'Signal'] = -1  # Sell signal

# Calculate daily returns
df['Daily_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']

# Plot the moving averages and buy/sell signals
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='MSFT Close Price', alpha=0.5)
plt.plot(df.index, df['Short_MA'], label=f'{short_window}-Day MA', linestyle='--')
plt.plot(df.index, df['Long_MA'], label=f'{long_window}-Day MA', linestyle='--')
plt.plot(df.index, df['Signal'], marker='o', markersize=5, label='Signal', linestyle='', color='g')
plt.title('Moving Average Crossover Strategy for MSFT')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print strategy statistics
total_return = (df['Strategy_Return'].dropna() + 1).cumprod()[-1]
print(f'Total Strategy Return: {total_return:.2f}')
print(f'Number of Trades: {df["Signal"].sum()}')



print('---------------------------------------------')
print('The accuracy of the method will now me calculated')
correct = 0
nr_of_predictions = 0
previous_close = None
buffer = 200
for index, row in df.iterrows():
    #First runs through the buffer that is equal to the long window margin because otherwise the testing might not be accurative because the long term margin has not yet had its full influence.
    buffer -= 1
    if buffer <= 1:
        if previous_close == None:
            previous_close = row['Close']
        else:
            if row['Close'] > previous_close:#If the current close is higher that the previous days close, then the ground truth signal should have been 1 yesterday
                ground_truth = True
            else:
                ground_truth = False
            prediction = (False if row['Signal'] == -1 else True)
            if ground_truth == prediction:
                correct += 1
            nr_of_predictions += 1
print(f"This method has a accuracy of {correct/nr_of_predictions}, The number of predictions made were {nr_of_predictions}")


