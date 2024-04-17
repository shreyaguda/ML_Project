"""
Created by: Chynna Hernandez
NetId: ch4262
Date Created: April 14, 2024
"""
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os

#The following function takes the year from the filename to determine the number of seconds since start of year
def get_year_start_from_filename(filename):
    year_part = filename.split('_')[-1][:4]
    return datetime.strptime(year_part, '%Y')

#The function takes the data types and columns and returns their types to safely handle alphanumeric columns
def define_dtypes_and_columns():
    dtypes = {
        'Time': str,
        'Symbol': str,
        'Sale Condition': str,
        'Trade Volume': float,
        'Trade Price': float,
        'Trade Stop Stock Indicator': str,
        'Trade Reporting Facility': str,
        'Trade Correction Indicator': str,
        'Sequence Number': str,
        'Trade Id': str,
        'Source of Trade': str,
        'Participant Timestamp': str,
        'Trade Reporting Facility TRF Timestamp': str,
        'Trade Through Exempt Indicator': str
    }
    use_cols = ['Time', 'Symbol', 'Trade Volume', 'Trade Price']
    return dtypes, use_cols

#Function to read chunks from the respective files
def read_data_chunks(filename, chunksize, dtypes):
    return pd.read_csv(filename, delimiter='|', compression='gzip', dtype=dtypes, chunksize=chunksize, on_bad_lines='skip')

#Function that filters out rows that do not have expected number of fields:
#b'Skipping line 110: expected 15 fields, saw 16\n'
#It's designed to handle errors for the last line of files: END|20240102|78739635|||||||||||||
def clean_data(chunk):
    chunk = chunk[chunk.apply(lambda x: len(x) == len(chunk.columns), axis=1)]
    return chunk

#Function to process the timestamps and convert it to number of seconds since start of year
def process_timestamps(chunk, filename, start_of_year):
    chunk['hour_min_sec'] = chunk['Time'].astype(str).str.slice(start=0, stop=6)
    chunk['nanoseconds'] = chunk['nanoseconds'] = chunk['Time'].astype(str).str.slice(start=6).astype(int)
    chunk['datetime'] = pd.to_datetime(chunk['hour_min_sec'], format='%H%M%S')
    chunk['full_datetime'] = chunk['datetime'] + pd.to_timedelta(chunk['nanoseconds'], unit='ns')
    file_date = get_date_from_filename(filename)
    chunk['full_datetime'] = pd.to_datetime(file_date.strftime('%Y-%m-%d')) + (chunk['full_datetime'] - chunk['full_datetime'].dt.normalize())
    chunk['seconds_since_start_of_year'] = (chunk['full_datetime'] - start_of_year).dt.total_seconds()
    return chunk

#Function to normalize the price and the volume
def normalize_data(chunk):
    scale_price = StandardScaler()
    scale_volume = StandardScaler()
    chunk['normalized_price'] = scale_price.fit_transform(chunk[['Trade Price']])
    chunk['normalized_volume'] = scale_volume.fit_transform(chunk[['Trade Volume']])
    return chunk

#Function to select and rename the columns that were preprocessed
def select_and_rename_columns(chunk):
    return chunk[['Time', 'Symbol', 'Trade Volume', 'Trade Price', 'normalized_price', 'normalized_volume', 'seconds_since_start_of_year']]

#Function to get the date from the filename. It's being used to convert the timestamp into number of seconds since the start of the year
def get_date_from_filename(filename):
    date_part = filename.split('_')[-1].split('.')[0]
    return datetime.strptime(date_part, '%Y%m%d')

#Function that preprcesses the data
def preprocess(filename, chunksize=100000):
    start_of_year = get_year_start_from_filename(filename)
    processed_chunks = []
    dtypes, use_cols = define_dtypes_and_columns()
    
    for chunk in read_data_chunks(filename, chunksize, dtypes):
        chunk = clean_data(chunk)
        chunk = process_timestamps(chunk, filename, start_of_year)
        chunk = normalize_data(chunk)
        processed_chunks.append(select_and_rename_columns(chunk))
    
    return pd.concat(processed_chunks)

def main(): 
    pd.set_option('display.max_columns', None)
    print("Current path: ", os.getcwd())
    filename = "/scratch/ch4262/final_project/EQY_US_ALL_TRADE_20240102.gz"
    processed_data = preprocess(filename)
    print(processed_data.head(50))
    #processed_data.to_csv('output.csv', index=False) # To get the data in csv to review. Note that I did this with a small file

if __name__ == "__main__":
    main()
