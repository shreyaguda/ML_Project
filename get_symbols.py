import gzip
import os

def get_unique_symbols_from_file(file_path):
    symbols = set()
    with gzip.open(file_path, 'rt') as file:  # Open the file in text mode
        for line in file:
            if not line.startswith("END"):
                parts = line.strip().split('|')
                if len(parts) > 2:
                    symbols.add(parts[2])
    return symbols

def get_all_unique_symbols(directory, output_path):
    all_symbols = set()
    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            file_path = os.path.join(directory, filename)
            file_symbols = get_unique_symbols_from_file(file_path)
            all_symbols.update(file_symbols)
    with open(output_path, 'w') as out_file:
        for symbol in sorted(all_symbols):
            out_file.write(f"{symbol}\n")
    return output_path

def main():
    directory_path = '/scratch/ch4262/final_project/'
    output_file_path = '/scratch/ch4262/final_project/unique_symbols.csv'  # Path where the results will be written
    result_file = get_all_unique_symbols(directory_path, output_file_path)
    print(f"Unique symbols have been written to: {result_file}")

if __name__ == "__main__":
    main()
