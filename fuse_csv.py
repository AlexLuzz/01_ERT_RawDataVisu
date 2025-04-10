import os
import pandas as pd
import glob

def fuse_csv_files(folder_path, output_file):
    # Define the header with all required columns
    header = [
        'No.', 'Time', 'A', 'B', 'M', 'N', 'I(mA)', 'Voltage(V)', 'Res.(ohm)', 'Error(%)', 'T(On)', 'T(0)', 'T(N):01',
        'App.Ch.(ms)', 'Error(ms)', 'T(N):02', 'App.Ch.(ms)', 'Error(ms)', 'T(N):03', 'App.Ch.(ms)', 'Error(ms)',
        'T(N):04', 'App.Ch.(ms)', 'Error(ms)', 'T(N):05', 'App.Ch.(ms)', 'Error(ms)', 'T(N):06', 'App.Ch.(ms)',
        'Error(ms)', 'T(N):07', 'App.Ch.(ms)', 'Error(ms)', 'T(N):08', 'App.Ch.(ms)', 'Error(ms)', 'T(N):09',
        'App.Ch.(ms)', 'Error(ms)', 'T(N):10', 'App.Ch.(ms)', 'Error(ms)', 'Survey', 'SurveyDate', 'MeasDate', 'meas',
        'k', 'meas_ele', 'rhoa'
    ]

    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # List to store DataFrames
    all_dfs = []

    for file in csv_files:
        # Read the CSV file
        df = pd.read_csv(file, sep=';')

        # Ensure all columns are present
        for col in header:
            if col not in df.columns:
                df[col] = pd.NA

        # Reorder columns to match the header
        df = df[header]

        # Append the DataFrame to the list
        all_dfs.append(df)

    # Concatenate all DataFrames
    df_concat = pd.concat(all_dfs, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    df_concat.to_csv(output_file, sep=';', index=False)

# Example usage
if __name__ == '__main__':
    folder_path = 'D:/02_ERT_Data/All_Data_csv/'
    output_file = 'D:/02_ERT_Data/fused_data_04oct24_01mar25.csv'
    fuse_csv_files(folder_path, output_file)