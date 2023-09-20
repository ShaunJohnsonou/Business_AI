import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential#To initializing the neural network
from keras.layers import LSTM#LSTM to add the LSTM layer
from keras.layers import Dropout#Dropout for preventing overfitting with dropout layers
from keras.layers import Dense#Dense to add a densely connected neural network layer
import datetime
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

class LSTM_class():
    def build_and_save_model(epochs, symbol):
        #Load the data into data into a pandas dataframe and 
        # url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
        # data = pd.read_csv(url)
        current_wd = os.getcwd()
        df = pd.read_csv(f'{current_wd}\\test_data\\{symbol}_historical_data.csv')
        df = df[['Date', 'Close']]
        df['Date'] = df['Date'].apply(LSTM_class.str_to_datetime)
        df.index = df.pop('Date')

        n=3
        windowed_df = LSTM_class.df_to_windowed_df(df, df.index[n], df.index[-1])
        dates, X, y = LSTM_class.windowed_df_to_date_X_y(windowed_df)    

        q_80 = int(len(dates) * .8)
        q_90 = int(len(dates) * .9)

        dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

        dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
        dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

        # plt.plot(dates_train, y_train)
        # plt.plot(dates_val, y_val)
        # plt.plot(dates_test, y_test)

        # plt.legend(['Train', 'Validation', 'Test'])  
        model = Sequential([layers.Input((3, 1)),
                            layers.LSTM(64),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(1)])

        model.compile(loss='mse',
                    optimizer=Adam(learning_rate=0.001),
                    metrics=['mean_absolute_error'])

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1)
        train_predictions = model.predict(X_train).flatten()
        print(X_train)

        # plt.plot(dates_train, train_predictions)
        # plt.plot(dates_train, y_train)
        # plt.legend(['Training Predictions', 'Training Observations'])

        val_predictions = model.predict(X_val).flatten()

        # plt.plot(dates_val, val_predictions)
        # plt.plot(dates_val, y_val)
        # plt.legend(['Validation Predictions', 'Validation Observations'])

        test_predictions = model.predict(X_test).flatten()

        model.save(f'{current_wd}/models/lstm_models/{symbol}_lstm_model')

    def update_model():
        pass

    def predict(model_path):
        model = load_model(model_path)
        model.predict


    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    @classmethod
    def df_to_windowed_df(cls, dataframe, first_date_str, last_date_str, n=3):
        # first_date = LSTM_class.str_to_datetime(first_date_str)
        # last_date  = LSTM_class.str_to_datetime(last_date_str)

        target_date = first_date_str
        
        dates = []
        X, Y = [], []

        last_time = False
        while True:
            df_subset = dataframe.loc[:target_date].tail(n+1)
            
            if len(df_subset) != n+1:
                print(f'Error: Window of size {n} is too large for date {target_date}')
                return

            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

            next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
            
            if last_time:
                break
            
            target_date = next_date

            if target_date == last_date_str:
                last_time = True
            
        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates
        
        X = np.array(X)
        for i in range(0, n):
            X[:, i]
            ret_df[f'Target-{n-i}'] = X[:, i]
        
        ret_df['Target'] = Y

        return ret_df
    
    def windowed_df_to_date_X_y(windowed_dataframe: pd.DataFrame):
        df_as_np = windowed_dataframe.to_numpy()
        dates = df_as_np[:, 0]
        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
        Y = df_as_np[:, -1]
        return dates, X.astype(np.float32), Y.astype(np.float32)



current_wd = os.getcwd()
# path_name = f"{current_wd}/LLM/models/MSFT_lstm_model"
# LSTM_class.predict(path_name)
LSTM_class.build_and_save_model(epochs = 1, symbol='MSFT')
# current_wd = os.getcwd()
# LSTM_class.predict(f"{current_wd}//models//lstm_models//lstm_100")