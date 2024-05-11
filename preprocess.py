import os
import glob
from datetime import datetime
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

def clean_data(chunk):
    # Assuming all rows should have exactly the same number of fields as there are columns defined
    expected_num_fields = len(chunk.columns)
    chunk = chunk.dropna(how='any')  # Drop rows with any NaN values which might indicate missing fields
    return chunk[chunk.apply(lambda x: len(x) == expected_num_fields, axis=1)]

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

#Add the date component from the filename to time data.
def add_date_component(time_data, file_date):
    date_timestamp = pd.to_datetime(file_date.strftime('%Y-%m-%d'))
    return date_timestamp + (time_data - time_data.dt.normalize())

def add_full_nanoseconds(chunk):
    # Calculate full nanoseconds since the last full second
    chunk['nanoseconds'] = chunk['Execution DateTime'].dt.microsecond * 1000 + chunk['Execution DateTime'].dt.nanosecond
    return chunk

# Parse the 'Time' and 'Participant Timestamp' columns to extract time components and calculate latency
def parse_trade_execution_time(chunk, file_date):
    base_date = pd.to_datetime(file_date.strftime('%Y-%m-%d'))

    chunk['Execution DateTime'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + chunk['Time'].str.slice(0, 6), format='%Y-%m-%d %H%M%S')
    chunk['Execution DateTime'] += pd.to_timedelta(chunk['Time'].str.slice(6).astype(int), unit='ns')

    participant_base_datetime = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + chunk['Participant Timestamp'].str.slice(0, 6), format='%Y-%m-%d %H%M%S')
    chunk['Participant DateTime'] = participant_base_datetime + pd.to_timedelta(chunk['Participant Timestamp'].str.slice(6).astype(int), unit='ns')

    chunk['latency_ns'] = (chunk['Execution DateTime'] - chunk['Participant DateTime']).dt.total_seconds() * 1e9

    # Additional step to calculate full nanoseconds
    chunk = add_full_nanoseconds(chunk)

    chunk['year'] = chunk['Execution DateTime'].dt.year
    chunk['month'] = chunk['Execution DateTime'].dt.month
    chunk['day'] = chunk['Execution DateTime'].dt.day
    chunk['hour'] = chunk['Execution DateTime'].dt.hour
    chunk['minute'] = chunk['Execution DateTime'].dt.minute
    chunk['second'] = chunk['Execution DateTime'].dt.second

    # Clean up the DataFrame
    chunk.drop(columns=['Participant Timestamp', 'Time', 'Participant DateTime' ], inplace=True)
    return chunk

#THis function retrieves the file and preprocesses it. It then stores the preprocessed data into parquet files 
def preprocess_and_convert_to_parquet(filename, output_dir):
    dtypes, use_cols = define_dtypes_and_columns()
    chunksize = 100000  #Adjust based on memory constraints
    parquet_writer = None
    schema = None  #Initialize schema as None

    line_count = count_lines_in_gz_file(filename)  #Count lines
    print(f"Processing {filename}: {line_count} lines found.")

    file_date = get_date_from_filename(filename)

    with gzip.open(filename, 'rt') as f:
        for chunk in pd.read_csv(f, delimiter='|', dtype=dtypes, usecols=use_cols, chunksize=chunksize): 
            chunk = clean_data(chunk)
            chunk = parse_trade_execution_time(chunk, file_date)

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
    start_time = datetime.now()  #Start timing for the whole main function
    directory = os.getcwd()
    print("Directory: ", directory)
    pattern = 'EQY_US_ALL_TRADE_20240102.gz'
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

    end_time = datetime.now()  # End timing for the whole main function
    print(f"Total time taken for main execution: {end_time - start_time}")  # Print elapsed time for the whole main function

if __name__ == "__main__":
    main()