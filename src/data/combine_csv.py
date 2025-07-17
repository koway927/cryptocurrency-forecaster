import os
import pandas as pd
import argparse
import shutil
import gzip
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Combining CSV files')
parser.add_argument('--data_directory', type=str, default='../data',
                    help='location of market data')
args = parser.parse_args()

def combine():
    if os.path.isdir(args.data_directory):
        combined_df = pd.DataFrame()
        # Get a list of gz files
        gz_files = [filename for filename in os.listdir(args.data_directory) if filename.endswith('.gz') and 'book_updates' in filename]

        # Iterate over gz files with a progress bar
        for filename in tqdm(gz_files, desc="Unzipping files", unit="file"):
            with gzip.open(os.path.join(args.data_directory, filename), 'rb') as f_in:
                with open(os.path.join(args.data_directory, filename[:-3]), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


        # Get a list of csv files
        csv_files = [filename for filename in os.listdir(args.data_directory) if filename.endswith('.csv') and 'book_updates' in filename]

        # Iterate over csv files with a progress bar
        for filename in tqdm(csv_files, desc="Combining CSV files", unit="file"):
            df = pd.read_csv(os.path.join(args.data_directory, filename))
            combined_df = pd.concat([combined_df, df])

        # Save the combined dataframe to a new csv file
        combined_df.to_csv(os.path.join(args.data_directory, 'combined.csv'), index=False)
        return combined_df
    else:
        print("Data directory does not exist")

if  __name__ == '__main__':
    combine()