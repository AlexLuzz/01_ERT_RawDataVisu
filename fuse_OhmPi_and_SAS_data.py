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
        df = pd.read_csv(file, sep=',')

        # Append the DataFrame to the list
        all_dfs.append(df)

    # Concatenate all DataFrames
    df_concat = pd.concat(all_dfs, ignore_index=True)
    # Save the concatenated DataFrame to a new CSV file
    df_concat.to_csv(output_file, sep=';', index=False)

# Example usage
if __name__ == '__main__':
    fused_s4k_data = 'C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/fused_AMP_SAS4000.csv'
    ohmpi_data_folder = 'C:/Users/AQ96560/OneDrive - ETS/Géophysique appliquée - GTO365 - 03 - Ohmpi - IV à Laval/'


    fused_ohmpi_data = 'C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/fused_OhmPi.csv'
    fuse_csv_files(ohmpi_data_folder, fused_ohmpi_data)