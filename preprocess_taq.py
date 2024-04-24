import os
import glob
from datetime import datetime
import dask
import gzip
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gzip
import re
import numpy as np

#Get all the files in final_project folder
def get_files(directory, pattern):
    return glob.glob(os.path.join(directory, pattern))

#Get the year from the filename
def get_year_start_from_filename(filename):
    year_part = filename.split('_')[-1][:4]
    return datetime.strptime(year_part, '%Y')

#Get the date from the filename
def get_date_from_filename(filename):
    date_part = filename.split('_')[-1].split('.')[0]
    return datetime.strptime(date_part, '%Y%m%d')

#Count the number of lines in the final, mostly used as debugging tool 
def count_lines_in_gz_file(file_path):
    with gzip.open(file_path, 'rt') as f:
        return sum(1 for _ in f)

#Get the size of the file, similarly used as debugging tool
def get_file_size(file_path):
    return os.path.getsize(file_path)

#Get all the datatypes and columns from file
def define_dtypes_and_columns():
    dtypes = {
        'Time': str,
        'Exchange': str,
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
    use_cols = ['Time', 'Exchange','Symbol', 'Trade Volume', 'Trade Price', 'Participant Timestamp']
    return dtypes, use_cols

#Parse the 'Time' column to combine hours, minutes, seconds, and nanoseconds into a datetime.
def parse_trade_execution_time(chunk):
    datetime_component = pd.to_datetime(chunk['Time'].str.slice(0, 6), format='%H%M%S', errors='coerce')
    nanoseconds = pd.to_timedelta(chunk['Time'].str.slice(6).astype(int), unit='ns')
    return datetime_component + nanoseconds

#Parse the 'Participant Timestamp' in HHMMSS followed by nanoseconds.
def parse_participant_timestamp(chunk):
    participant_time_data = chunk['Participant Timestamp'].str.extract(r'(\d{6})(\d{9})')
    participant_time_data[1] = participant_time_data[1].fillna('0').astype(int)
    datetime_component = pd.to_datetime(participant_time_data[0], format='%H%M%S', errors='coerce')
    nanoseconds = pd.to_timedelta(participant_time_data[1], unit='ns')
    return datetime_component + nanoseconds

#Add the date component from the filename to time data.
def add_date_component(time_data, file_date):
    date_timestamp = pd.to_datetime(file_date.strftime('%Y-%m-%d'))
    return date_timestamp + (time_data - time_data.dt.normalize())

#Compute latency as the difference between execution and participant reporting timestamps.
def compute_latency(participant_datetime, execution_datetime):
    return (execution_datetime - participant_datetime).dt.total_seconds() * 1e9

def process_timestamps(chunk, filename, start_of_year):
    valid_rows = chunk['Time'].str.match(r'^\d{6}\d{9}$')
    filtered_chunk = chunk[valid_rows].copy()

    filtered_chunk['execution_datetime'] = parse_trade_execution_time(filtered_chunk)
    filtered_chunk['participant_datetime'] = parse_participant_timestamp(filtered_chunk)

    file_date = get_date_from_filename(filename)
    filtered_chunk['execution_datetime'] = add_date_component(filtered_chunk['execution_datetime'], file_date)
    filtered_chunk['participant_datetime'] = add_date_component(filtered_chunk['participant_datetime'], file_date)

    filtered_chunk['seconds_since_start_of_year'] = (filtered_chunk['execution_datetime'] - start_of_year).dt.total_seconds()
    filtered_chunk['latency_ns'] = compute_latency(filtered_chunk['participant_datetime'], filtered_chunk['execution_datetime'])

    #Drop intermediate columns if they are not needed in the output
    filtered_chunk.drop(columns=['execution_datetime' , 'participant_datetime'], inplace=True)

    return filtered_chunk

def preprocess_and_convert_to_parquet(filename, output_dir):
    dtypes, use_cols = define_dtypes_and_columns()
    chunksize = 100000  #Adjust based on memory constraints
    parquet_writer = None
    schema = None  #Initialize schema as None

    line_count = count_lines_in_gz_file(filename)  #Count lines
    print(f"Processing {filename}: {line_count} lines found.")

    with gzip.open(filename, 'rt') as f:
        for chunk in pd.read_csv(f, delimiter='|', dtype=dtypes, usecols=use_cols, chunksize=chunksize):
            chunk = process_timestamps(chunk, filename, get_year_start_from_filename(filename))

            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if parquet_writer is None:
                if table.schema is not None:
                    schema = table.schema
                    parquet_file_path = os.path.join(output_dir, os.path.basename(filename).replace('.gz', '.parquet'))
                    parquet_writer = pq.ParquetWriter(
                        parquet_file_path,
                        schema,
                        compression='snappy',
                        version='2.6',
                        use_deprecated_int96_timestamps=False,
                        coerce_timestamps=None,  #Ensure nanosecond precision is maintained
                        allow_truncated_timestamps=False
                    )
            if parquet_writer is not None:
                parquet_writer.write_table(table)

    if parquet_writer:
        parquet_writer.close()
        parquet_file_size = get_file_size(parquet_file_path)
        print(f"Parquet file written: {parquet_file_path} (Size: {parquet_file_size} bytes)")

def main():
    directory = os.getcwd()
    print("Directory: ", directory)
    pattern = 'EQY_US_ALL_TRADE_*.gz'
    files = get_files(directory, pattern)
    print(f"Files matched: {files}")  #Debugging output
    output_dir = 'parquet_output'
    os.makedirs(output_dir, exist_ok=True)
    total_files_processed = 0

    if files:
        for file in files:
            preprocess_and_convert_to_parquet(file, output_dir)
            total_files_processed += 1
        print(f"All files have been processed and converted to Parquet. Total files processed: {total_files_processed}")
    else:
        print("No files processed or files were empty.")

if __name__ == "__main__":
    main()