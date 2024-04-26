import os
import glob
from datetime import datetime
import dask
import dask.dataframe as dd
import gzip
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gzip
import re
import numpy as np
from keras.models import Sequential
import os
import glob
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, Dense
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# # Specify the path to your Parquet file
# parquet_file_path = 

# # Read the Parquet file using Dask
# df = dd.read_parquet(parquet_file_path)

# def get_sentences(df):
#     # df['daily_return'] = df.groupby('Symbol')['Trade Price']
#     df = df.groupby('Symbol')['average_volume']
#     return df

def create_sequences(data, window_size=5):
    sequences = []
    for i in range(data.shape[0] - window_size):
        sequences.append(data.iloc[i:(i + window_size), :].values)
    return np.array(sequences)

def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(50)(inputs)
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(50, return_sequences=True)(decoded)
    decoded = Dense(input_shape[1], activation='linear')(decoded)
    
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

#Get all the files in final_project folder
def get_files(directory, pattern):
    return glob.glob(os.path.join(directory, pattern))

# def process_data(files):
#     data_frames = []
#     for file in files:
#         df = dd.read_parquet(file).compute()  # Ensure this fits in memory or handle chunkwise
#         data_frames.append(df)
#     full_df = pd.concat(data_frames, ignore_index=True)
#     return full_df

def preprocess_features(df):
    # Define which columns are categorical and which are numeric
    categorical_cols = ['Exchange', 'Symbol', 'Time', 'Participant Timestamp']  # assuming these are your categorical columns
    numeric_cols = [col for col in df.columns if col not in categorical_cols]  # all other columns are treated as numeric



    # for col in numeric_cols:
    #     if df[col].dtype not in ['float64', 'int64']:
    #         print(f"Warning: Column {col} has non-numeric type {df[col].dtype}")

    # for col in categorical_cols:
    #     if df[col].isnull().any():
    #         print(f"Warning: Column {col} contains NaN values")

    # Create a transformer pipeline
    transformer = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

    # Fit and transform the data
    pipeline = Pipeline(steps=[('preprocessor', transformer)])
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    print(df.head(10))
    df = pd.DataFrame(df.compute())
    try:
        processed_data = pipeline.fit_transform(df)
    except Exception as e:
        print("Error during transformation:", e)
        raise

    # processed_data = pipeline.fit_transform(df)

    return processed_data



def main():
    directory = os.getcwd()
    print("Directory: ", directory)
    directory += "/parquet_output"
    pattern = 'EQY_US_ALL_TRADE_*.parquet'
    files = get_files(directory, pattern)
    for file in files:
        df = dd.read_parquet(file)
    print(f"Files matched: {files}")  #Debugging output
    output_dir = 'embeddings_output'
    os.makedirs(output_dir, exist_ok=True)
    # modified_df = create_sequences(df)

    # Compute the Dask DataFrame to retrieve the actual values
    # computed_df = modified_df.compute()

    # Now you can view the 'features' column
    # print(modified_df.head())

    # Preprocess features
    # processed_data = preprocess_features(df)

    # df.fillna(method='ffill', inplace=True)  # Handle NaNs
    print("Preprocessing features...")
    processed_data = pd.DataFrame(preprocess_features(df).toarray())
    print(processed_data.head(10))
    
    print("Features processed. Proceeding to create sequences...")

    # Create sequences
    sequences = create_sequences(processed_data, 30)
    volprice = pd.DataFrame(df[["Trade Volume", "Trade Price"]].compute())
    rolling_sum = volprice.rolling(window=30).sum().iloc[30:].sum(axis=1)
    # Split data for training and testing
    print(rolling_sum.head(10))
    X_train, X_test, Y_train, Y_test = train_test_split(sequences, rolling_sum.values.reshape(969, 1, -1), test_size=0.2, random_state=42)
   

    # Build and train autoencoder
    autoencoder, encoder = build_autoencoder(X_train[0].shape)
    autoencoder.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test))

    # Optionally, extract embeddings
    embeddings = encoder.predict(X_train)

    print(embeddings)
    # Save or process embeddings further
    print("Embeddings have been generated and can be used for clustering or other tasks.")


    # # Print the modified DataFrame
    # print(modified_df.head(10))

    # # Or you can access specific columns
    # print(modified_df['features'].head(10))


    # total_files_processed = 0

    # if files:
    #     volume_mean, volume_std, price_offset = compute_global_normalization_params(files, *define_dtypes_and_columns())

    #     for file in files:
    #         preprocess_and_convert_to_parquet(file, output_dir, volume_mean, volume_std, price_offset)
    #         total_files_processed += 1
    #     print(f"All files have been processed and converted to Parquet. Total files processed: {total_files_processed}")
    # else:
    #     print("No files processed or files were empty.")

if __name__ == "__main__":
    main()