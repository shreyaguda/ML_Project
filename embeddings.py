import dask.dataframe as dd
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, LeakyReLU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocess import get_files
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import json

#Set the environment variable for TensorFlow to use asynchronous GPU allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def load_sector_data(filepath):
    #Load sector data
    return pd.read_csv(filepath, header=None, names=['Symbol', 'Sector'], dtype={'Symbol': str})

def add_cyclic_time_features(df):
    # Adding cyclic features for hour
    df['hour_sin'] = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2. * np.pi / 24))
    return df

def get_symbol_id_mapping(df, included_symbols):
    #Map each symbol to a unique ID for use in the model
    df = df[df['Symbol'].isin(included_symbols['Symbol'])]
    symbol_ids = {symbol: idx for idx, symbol in enumerate(df['Symbol'].unique())}
    logging.info(f"Symbol IDs generated: {len(symbol_ids)}")
    return symbol_ids

def replace_symbols_with_ids(df, symbol_ids):
    #Replace symbol strings with their corresponding IDs in the dataframe
    df['Symbol'] = df['Symbol'].apply(lambda x: symbol_ids.get(x, -1))  #Use -1 for missing symbols to debug
    logging.info("After replacing symbols, data sample: {}".format(df.head()))
    return df

def normalize_and_log_scale(df, window_size=50):
    #Normalize trade volume and price using log scaling
    logging.info("Starting normalization and log scaling.")

    #Calculate the moving average
    df['Trade Volume MA'] = df['Trade Volume'].rolling(window=window_size, min_periods=1).mean()

    #Apply logarithmic scaling, adding a small constant to avoid log(0)
    df['Trade Volume Log'] = np.log(df['Trade Volume MA'] + 1)

    return df

def add_volume_price_ratio(df):
    #Calculate the ratio of trade volume to trade price and handle division by zero issues.
    df['Volume_Price_Ratio'] = np.log((df['Trade Volume'] / (df['Trade Price'])) + 1)
    return df

def build_embedding_model(num_symbols, embedding_dim=20, feature_dim=4, dropout_rate=0.4, l2_reg=0.01, learning_rate=0.01, num_dense_layers=32):
    #Build a neural network model with an embedding layer
    input_layer = Input(shape=(1,))
    embedding = Embedding(input_dim=num_symbols + 1, output_dim=embedding_dim)(input_layer)
    flat = Flatten()(embedding)
    
    dropout1 = Dropout(dropout_rate)(flat)
    
    #Adding L2 regularization to this dense layer
    dense1 = Dense(num_dense_layers, kernel_regularizer=l2(l2_reg))(dropout1)
    act1 = LeakyReLU(negative_slope=0.01)(dense1)
    dropout2 = Dropout(dropout_rate)(act1)
    
    #Adding another dense layer with L2 regularization
    dense2 = Dense(num_dense_layers, kernel_regularizer=l2(l2_reg))(dropout2)
    act2 = LeakyReLU(negative_slope=0.01)(dense2)
    dropout3 = Dropout(dropout_rate)(act2)
    
    output_layer = Dense(feature_dim, activation='linear')(dropout3)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model

def apply_pca(scaled_df, n_components=2):
    #Apply PCA to reduce dimensions of the scaled data
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(data=principal_components, columns=scaled_df.columns)
    return pca_df

def scale_data(df):
    #Scale data features, apply PCA, and combine results with non-scaled features
    features_to_scale = ['Volume_Price_Ratio', 'Trade Volume Log']
    scaler = StandardScaler()
    if features_to_scale:
        scaled_values = scaler.fit_transform(df[features_to_scale])
        scaled_df = pd.DataFrame(scaled_values, columns=features_to_scale, index=df.index)
    else:
        raise ValueError("Feature list for scaling is empty. Check feature selection.")
    
    #Non-scaled features, typically categorical or already scaled differently
    non_scaled_features = ['Symbol', 'hour_sin', 'hour_cos']
    
    for feature in non_scaled_features:
        scaled_df[feature] = df[feature]

    #Apply PCA on scaled data
    pca_df = apply_pca(scaled_df[features_to_scale], n_components=2) 
    print("PCA DataFrame shape:", pca_df.shape) 
    
    #Resetting indices to ensure unique index values
    pca_df.reset_index(drop=True, inplace=True)
    non_scaled_data = scaled_df[non_scaled_features].reset_index(drop=True)
    print("Non-scaled DataFrame shape after index reset:", non_scaled_data.shape)

    #Combine PCA features with non-scaled features
    final_df = pd.concat([pca_df, non_scaled_data], axis=1)
    print("Final DataFrame shape after concatenation:", final_df.shape)
    return final_df

def train_embedding_model(df, model, batch_size=512):
    #Train the neural network model on the embedding features
    logging.info("Starting model training.")
    print("Columns being passed: ", df.columns)

    numerical_features = ['Volume_Price_Ratio', 'Trade Volume Log', 'hour_sin', 'hour_cos']

    if not all(col in df.columns for col in numerical_features + ['Symbol']):
        logging.error("One or more required columns are missing from the DataFrame.")
        raise ValueError("One or more required columns are missing from the DataFrame.")

    train_df = df.iloc[:int(len(df) * 0.8)]
    val_df = df.iloc[int(len(df) * 0.8):]
    train_inputs = train_df['Symbol'].values.reshape(-1, 1)
    val_inputs = val_df['Symbol'].values.reshape(-1, 1)
    train_targets = train_df[numerical_features].values
    val_targets = val_df[numerical_features].values

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

    history = model.fit(train_inputs, train_targets, epochs=20, batch_size=batch_size,
                        validation_data=(val_inputs, val_targets), callbacks=[early_stopping, reduce_lr])
    logging.info("Model training completed.")
    return model, history

def extract_embeddings(model, num_symbols):
    #Extract embeddings from the trained model
    return model.layers[1].get_weights()[0][1:num_symbols+1]  #Skip the first row which is for padding index

def save_embeddings_to_parquet(embeddings, symbol_ids, filename):
    #Save the embeddings with symbol IDs to a Parquet file using PyArrow for append support.
    #Convert embeddings to DataFrame
    embeddings_df = pd.DataFrame(embeddings, index=pd.Index(symbol_ids.keys(), name='Symbol'))
    embeddings_df.reset_index(inplace=True)
    
    table = pa.Table.from_pandas(embeddings_df, preserve_index=False)
    
    #Check if the file exists, if not, create it, else append to it
    if not os.path.exists(filename):
        #Write a new Parquet file
        pq.write_table(table, filename)
    else:
        #Open the existing Parquet file and append data
        writer = pq.ParquetWriter(filename, table.schema, flavor='spark')
        writer.write_table(table)
        writer.close()

def process_file(file_path, embeddings_file, sector_file):
    #Process each file, apply data transformations, train the model, and save embeddings
    logging.info("Processing file: %s", file_path)
    ddf = dd.read_parquet(file_path, blocksize='500MB')
    #Load sectors and create a list of included symbols
    included_symbols = load_sector_data(sector_file)
    #Debugging output to confirm columns post-loading
    logging.info("Columns available after loading: %s", ddf.columns)

    logging.info("Applying data transformations") 
    ddf = ddf.map_partitions(add_volume_price_ratio)
    ddf = ddf.map_partitions(normalize_and_log_scale)
    ddf = ddf.map_partitions(add_cyclic_time_features)

    #Compute to convert Dask DataFrame to Pandas DataFrame for operations not supported in Dask
    logging.info("Coverting to pandas dataframe") 
    df = ddf.compute()

    if df.empty:
        logging.info("Data is empty after processing file: %s", file_path)
        return
    
    symbol_ids = get_symbol_id_mapping(df, included_symbols)
    logging.info("Symbol ids: %s", included_symbols[:10])
    df = replace_symbols_with_ids(df, symbol_ids)
    scaled_data = scale_data(df)

    model = build_embedding_model(len(symbol_ids), dropout_rate=0.4, l2_reg=0.01, learning_rate=0.01, num_dense_layers=32)
    trained_model, history = train_embedding_model(scaled_data, model, batch_size=1024)

    embeddings = extract_embeddings(trained_model, len(symbol_ids))
    save_embeddings_to_parquet(embeddings, symbol_ids, embeddings_file)
    return history.history

def aggregate_and_plot_histories(histories, file_name):
    # Find the minimum number of epochs completed in all training runs
    min_epochs = min(len(h['loss']) for h in histories)

    avg_loss = [np.mean([h['loss'][i] for h in histories if i < len(h['loss'])]) for i in range(min_epochs)]
    avg_val_loss = [np.mean([h['val_loss'][i] for h in histories if i < len(h['val_loss'])]) for i in range(min_epochs)]

    plt.figure(figsize=(10, 5))
    plt.plot(avg_loss, label='Average Training Loss')
    plt.plot(avg_val_loss, label='Average Validation Loss')
    plt.title('Average Model Loss Across All Files')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'{file_name}.png')
    plt.show()

def save_histories_to_json(histories, filename):
    #Save histories to json format for later analysis
    json_ready_histories = []
    for history in histories:
        json_ready_history = {}
        for key, values in history.items():
            json_ready_history[key] = [float(value) for value in values]
        json_ready_histories.append(json_ready_history)

    with open(filename, 'w') as f:
        json.dump(json_ready_histories, f, indent=4)

def setup_tensorflow_gpu():
    #Configure TensorFlow to use GPU efficiently
    logging.info("Setting up GPU...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            #Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            #Memory growth must be set before GPUs have been initialized
            print(e)

def main():
    #Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    setup_tensorflow_gpu()

    #Configuration and paths
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "parquet_output")
    pattern = 'EQY_US_ALL_TRADE_2024*.parquet'
    embeddings_file = os.path.join(output_dir, 'embeddings.parquet')
    sector_file = os.path.join(base_dir, 'sectors.csv')

    files = get_files(output_dir, pattern)
    all_histories = []
    
    for file in files:
            history = process_file(file, embeddings_file, sector_file)
            if history:
                all_histories.append(history)  #Get last validation loss

    #After all files are processed, aggregate histories and plot
    if all_histories:
        aggregate_and_plot_histories(all_histories, "Aggregate Training and Validation Metrics")
        #Save histories to JSON for later analysis
        save_histories_to_json(all_histories, "training_histories.json")

    #Perform clustering and evaluation if embeddings are available
    if os.path.exists(embeddings_file):
        logging.info("Embeddings file exists.")
    else:
        logging.info("Embeddings file does not exist, skipping clustering and evaluation.")

if __name__ == "__main__":
    main()