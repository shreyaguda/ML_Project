import argparse
import dask.dataframe as dd
import pandas as pd

def read_parquet_file(filename):
    # Setting the display option for columns
    pd.set_option('display.max_columns', None)

    # Reading the Parquet file using Dask
    ddf = dd.read_parquet(filename)

    # Printing the first 20 rows
    print(ddf.head(10))

def main():
    # Creating an ArgumentParser object
    parser = argparse.ArgumentParser(description='Read a Parquet file and display the first 20 rows.')

    # Adding an argument for the Parquet file path
    parser.add_argument('filename', type=str, help='The path to the Parquet file to be read.')

    # Parsing the arguments
    args = parser.parse_args()

    # Reading the Parquet file
    read_parquet_file(args.filename)

if __name__ == "__main__":
    main()
