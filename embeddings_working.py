import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
import logging
from preprocess import get_files
from keras.layers import Concatenate  # Import required for Concatenate

def get_symbol_id_mapping(df):
    symbol_ids = {symbol: idx for idx, symbol in enumerate(df['Symbol'].unique())}
    print("Symbol IDs generated:", len(symbol_ids))  # Debugging output
    return symbol_ids

def replace_symbols_with_ids(df, symbol_ids):
    # Replace symbols with numeric IDs and keep them as integers
    df['Symbol'] = df['Symbol'].apply(lambda x: symbol_ids[x])
    return df

def add_datetime_features(df):
    df['Execution DateTime'] = pd.to_datetime(df['Execution DateTime'])

    # Cyclic transformations for day, month, hour, minute, second
    df['month_sin'] = np.sin(2 * np.pi * df['Execution DateTime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Execution DateTime'].dt.month / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['Execution DateTime'].dt.day / df['Execution DateTime'].dt.days_in_month)
    df['day_cos'] = np.cos(2 * np.pi * df['Execution DateTime'].dt.day / df['Execution DateTime'].dt.days_in_month)
    df['hour_sin'] = np.sin(2 * np.pi * df['Execution DateTime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Execution DateTime'].dt.hour / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['Execution DateTime'].dt.minute / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['Execution DateTime'].dt.minute / 60)
    df['second_sin'] = np.sin(2 * np.pi * df['Execution DateTime'].dt.second / 60)
    df['second_cos'] = np.cos(2 * np.pi * df['Execution DateTime'].dt.second / 60)

    # Year and nanoseconds can be scaled linearly if needed
    df['year'] = df['Execution DateTime'].dt.year
    df['nanoseconds'] = df['Execution DateTime'].dt.nanosecond

    return df

def normalize_and_log_scale(df, window_size=50):
    # Calculate the moving average
    df['Trade Volume MA'] = df['Trade Volume'].rolling(window=window_size, min_periods=1).mean()
    df['Trade Price MA'] = df['Trade Price'].rolling(window=window_size, min_periods=1).mean()

    # Apply logarithmic scaling, adding a small constant to avoid log(0)
    df['Trade Volume Log'] = np.log(df['Trade Volume MA'] + 1)
    df['Trade Price Log'] = np.log(df['Trade Price MA'] + 1)
    return df

def add_volume_price_ratio(df):
    # Avoid division by zero by adding a small constant to the denominator if necessary
    df['Volume_Price_Ratio'] = df['Trade Volume'] / (df['Trade Price'] + 1e-8)
    return df

def build_embedding_model(num_symbols, embedding_dim=20, feature_dim=3):
    input_layer = Input(shape=(1,))
    embedding = Embedding(input_dim=num_symbols + 1, output_dim=embedding_dim)(input_layer)
    flat = Flatten()(embedding)
    
    dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(flat)
    dropout = Dropout(0.5)(dense)
    output_layer = Dense(feature_dim, activation='linear')(dropout)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='mse')
    return model

def train_embedding_model(df, model):
    logging.info("Starting model training.")

    numerical_features = ['Trade Volume Log_mean', 'Trade Price Log_mean', 'Volume_Price_Ratio_mean']
    
    # Confirm columns
    if not all(col in df.columns for col in numerical_features):
        raise ValueError("One or more required columns are missing from the DataFrame.")

    # Split the data
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]
    val_df = df.iloc[split_point:]

    logging.info(f"Training DataFrame shape: {train_df.shape}")
    logging.info(f"Validation DataFrame shape: {val_df.shape}")

    # Prepare inputs for embedding model
    train_inputs = train_df['Symbol'].values.reshape(-1, 1)  # Reshape for embedding input
    val_inputs = val_df['Symbol'].values.reshape(-1, 1)      # Reshape for embedding input

    train_targets = train_df[numerical_features].values
    val_targets = val_df[numerical_features].values

    # Setup callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Fit the model
    history = model.fit(train_inputs, train_targets, epochs=50, batch_size=1000,
                        validation_data=(val_inputs, val_targets), callbacks=[early_stopping, reduce_lr])

    return model, history

def scale_data(df):
    # Features that require scaling
    features_to_scale = [
        'Trade Volume Log_mean', 'Trade Price Log_mean', 'Volume_Price_Ratio_mean'
    ]

    scaler = StandardScaler()

    # Scale these features
    scaled_values = scaler.fit_transform(df[features_to_scale])
    scaled_df = pd.DataFrame(scaled_values, columns=features_to_scale, index=df.index)

    non_scaled_features = ['Symbol']
    
    for feature in non_scaled_features:
        scaled_df[feature] = df[feature]

    return scaled_df

def extract_embeddings(model, num_symbols):
    # Extract embeddings from the model
    return model.layers[1].get_weights()[0][1:num_symbols+1]  # Skip the first row which is for padding index

def save_embeddings_to_parquet(embeddings, symbol_ids, filename):
    """ Save the embeddings with symbol IDs to a Parquet file using PyArrow for append support. """
    # Convert embeddings to DataFrame
    embeddings_df = pd.DataFrame(embeddings, index=pd.Index(symbol_ids.keys(), name='Symbol'))
    embeddings_df.reset_index(inplace=True)
    
    table = pa.Table.from_pandas(embeddings_df, preserve_index=False)
    
    # Check if the file exists, if not, create it, else append to it
    if not os.path.exists(filename):
        # Write a new Parquet file
        pq.write_table(table, filename)
    else:
        # Open the existing Parquet file and append data
        writer = pq.ParquetWriter(filename, table.schema, flavor='spark')
        writer.write_table(table)
        writer.close()

def process_aggregated_data(aggregated_data, embeddings_file):
    symbol_ids = get_symbol_id_mapping(aggregated_data)
    aggregated_data = replace_symbols_with_ids(aggregated_data, symbol_ids)
    logging.info(f"Columns available in DataFrame: {aggregated_data.columns}")
    scaled_data = scale_data(aggregated_data)
    #pca_data, pca_model = apply_pca(scaled_data, n_components=0.95)
    model = build_embedding_model(len(symbol_ids))
    #trained_model, history = train_embedding_model(pca_data, model)
    trained_model, history = train_embedding_model(scaled_data,model)
    embeddings = extract_embeddings(trained_model, len(symbol_ids))
    save_embeddings_to_parquet(embeddings, symbol_ids, embeddings_file)
    
def process_file(file_path, embeddings_file):
    logging.info("Processing file: %s", file_path)
    ddf = dd.read_parquet(file_path, blocksize='500MB')
    
    # Debugging output to confirm columns post-loading
    print("Columns available after loading:", ddf.columns)

    # Apply date-time and other feature transformations
    ddf = ddf.map_partitions(add_datetime_features)

    ddf = ddf.map_partitions(add_volume_price_ratio)
    ddf = ddf.map_partitions(normalize_and_log_scale)

    if ddf.npartitions == 0:
        logging.info("No valid symbols found in file: %s", file_path)
        return

    aggregated_data = aggregate_and_compute(ddf)
    if aggregated_data.empty:
        logging.info("Aggregated data is empty after processing file: %s", file_path)
        return

    print("Columns before processing aggregated data:", aggregated_data.columns)
    process_aggregated_data(aggregated_data, embeddings_file)

    # More debugging outputs to trace the data
    print("Final columns in aggregated data:", aggregated_data.columns)

def aggregate_and_compute(ddf):
    # Define simplified metadata to test the basic functionality
    meta = {
        'Symbol': 'object',
        'Trade Volume Log_mean': 'float64',
        'Trade Price Log_mean': 'float64',
        'Volume_Price_Ratio_mean': 'float64'
    }

    return ddf.map_partitions(aggregate_data, meta=meta).compute()

def aggregate_data(df):
    # Perform a simplified aggregation to debug
    df_agg = df.groupby('Symbol').agg({
        'Trade Volume Log': 'mean',
        'Trade Price Log': 'mean',
        'Volume_Price_Ratio': 'mean',
    }).reset_index()

    df_agg.columns = ['Symbol', 'Trade Volume Log_mean', 'Trade Price Log_mean', 'Volume_Price_Ratio_mean']
    return df_agg

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configuration and paths
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "parquet_output")
    pattern = 'EQY_US_ALL_TRADE_20240102.parquet'
    embeddings_file = os.path.join(output_dir, 'embeddings.parquet')

    files = get_files(output_dir, pattern)

    for file in files:
        process_file(file, embeddings_file)

    # Perform clustering and evaluation if embeddings are available
    if os.path.exists(embeddings_file):
        logging.info("Embeddings file exists.")
    else:
        logging.info("Embeddings file does not exist, skipping clustering and evaluation.")

if __name__ == "__main__":
    main()