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
from scipy.sparse import vstack
import tensorflow as tf

MIN_ROWS = 100  #minimum number of rows needed for processing

def create_sequences(data, window_size=5):
    sequences = []
    print(f"Data shape before sequence creation: {data.shape}")
    if data.shape[0] > window_size:
        for i in range(data.shape[0] - window_size + 1):
            sequence = vstack([data[j] for j in range(i, i + window_size)])  #Needed for sparse matrix 
            sequences.append(sequence)
        print(f"Number of sequences created: {len(sequences)}")
    else:
        print("Not enough data to create sequences")
    return sequences

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

def preprocess_features(df):
    # Define which columns are categorical and which are numeric
    categorical_cols = ['Exchange', 'Symbol', 'Time', 'Participant Timestamp']  # assuming these are your categorical columns
    numeric_cols = [col for col in df.columns if col not in categorical_cols]  # all other columns are treated as numeric

    # Create a transformer pipeline
    transformer = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

    # Fit and transform the data
    pipeline = Pipeline(steps=[('preprocessor', transformer)])
    """ print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    print(df.head(10)) """
    # df = pd.DataFrame(df.compute())
    try:
        processed_data = pipeline.fit_transform(df)
        return processed_data
    except Exception as e:
        print("Error during transformation:", e)
        raise

def process_symbol_group(group, autoencoder, encoder, window_size):
    processed_data = preprocess_features(group)
    sequences = create_sequences(processed_data, window_size)  #Using dynamic window size
    sequences = [seq.toarray() for seq in sequences]  #Convert sequences to dense format after change create sequence function
    if len(sequences) > 0:
        sequences = np.array(sequences)
        autoencoder.fit(sequences, sequences, epochs=20)  #Training the autoencoder
        symbol_embeddings = encoder.predict(sequences)
        return symbol_embeddings.mean(axis=0)  #Return mean embedding for the symbol
    return None

#Create a dynamic window size because 30 for sequence is not working
"""debug output indicates that while the script is processing rows for each symbol, it only uses the first 30 rows for the sequence generation, which isn't enough to create any sequences with your current setup. This is because the window size (30) and the number of rows you're using (30) match exactly, which means no sliding window of data can be created—it would only form one sequence exactly matching the input if the loop condition allowed for zero iterations, which it doesn't as currently set up."""
def choose_window_size(group, min_size=30, max_size=100):
    proposed_size = int(len(group) * 0.1)  # 10% of the group's length
    return max(min_size, min(proposed_size, max_size))  # Ensures the size is within specified bounds

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) #Need to check if enough GPU after script got killed :( 
    directory = os.getcwd() + "/parquet_output"
    pattern = 'EQY_US_ALL_TRADE_*.parquet'
    files = get_files(directory, pattern)
    embeddings = {}
    for file in files:
        ddf = dd.read_parquet(file)
        df = ddf.compute()
        grouped = df.groupby('Symbol')
        for symbol, group in grouped:
            print(f"Processing {symbol} with {group.shape[0]} rows.")
            if group.shape[0] > MIN_ROWS:
                window_size = choose_window_size(group) #Get a dynamic window size instead of 30
                sample_processed_data = preprocess_features(group)
                if sample_processed_data.shape[0] > 0:
                    sample_sequences = create_sequences(sample_processed_data, window_size)
                    if len(sample_sequences) > 0:
                        sequence_shape = (sample_sequences[0].shape[0], sample_sequences[0].shape[1])  #Adjusting shape after changing create sequence function to get sparse matrix
                        autoencoder, encoder = build_autoencoder(sequence_shape)
                        symbol_embedding = process_symbol_group(group, autoencoder, encoder, window_size)
                        if symbol_embedding is not None:
                            embeddings[symbol] = symbol_embedding

    print("Embeddings have been generated and can be used for clustering or other tasks.")
    print(f"Processed {len(embeddings)} embeddings.")

if __name__ == "__main__":
    main()