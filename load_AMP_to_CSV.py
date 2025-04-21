#from tools.tools import load_amp_files
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

def add_meas_k(df, n_elec_bh=8):
    """
    Add a 'meas' column to the DataFrame based on measurement logic.

    :param df: Input DataFrame containing columns A, B, and M.
    :param n_elec_bh: Number of electrodes for borehole measurements (default is 8).
    :return: The DataFrame with the added 'meas' column.
    """
    df['meas'] = 'Unknown'  # Default value
    #df['weight'] = 1.0  # Default value

    # Calculate the 'meas' based on the provided logic
    for index, row in df.iterrows():
        # Convert 'A', 'B', and 'M' to integers
        A = int(row['A'])
        B = int(row['B'])
        M = int(row['M'])
        B_A = B - A
        M_A = M - A
        if B_A == n_elec_bh:
            if M_A == 1:
                df.at[index, 'meas'] = 'KR0'
                df.at[index, 'k'] = 0.815
            elif M_A == 2:
                df.at[index, 'meas'] = 'KR1'
                df.at[index, 'k'] = 2.189
            elif M_A == 3:
                df.at[index, 'meas'] = 'KR2' 
                df.at[index, 'k'] = 4.453
            elif M_A == 4:
                df.at[index, 'meas'] = 'KR3' 
                df.at[index, 'k'] = 7.988
        elif B_A == 2 * n_elec_bh:
            if M_A == 1:
                df.at[index, 'meas'] = 'KJ0' 
                df.at[index, 'k'] = 0.711
            elif M_A == 2:
                df.at[index, 'meas'] = 'KJ1'
                df.at[index, 'k'] = 1.63
            elif M_A == 3:
                df.at[index, 'meas'] = 'KJ2'
                df.at[index, 'k'] = 2.825
            elif M_A == 4:
                df.at[index, 'meas'] = 'KJ3' 
                df.at[index, 'k'] = 4.377
    return df

def add_meas_ele(df):
    # Check if 'meas_ele' column exists, if not, add it
    if 'meas_ele' not in df.columns:
        df.loc[:, 'meas_ele'] = df.apply(lambda row: f"{row['A']}_{row['B']}", axis=1)
    return df

def load_amp_files(file_paths=None, n_elec_bh=8, delete_columns=None, clear_electrodes=None, detect_by_first_measurement=True):
    """
    Load multiple ABEM Multi-Purpose Format (.AMP) files into a single pandas DataFrame.
    
    :param file_paths: List of paths to .AMP files.
    :param delete_columns: List of columns to delete from the DataFrame.
    :param clear_electrodes: List of electrode values to clear from the DataFrame (defaults to including '32768').
    :return: A concatenated pandas DataFrame containing the data from all files with an additional 'Survey' column.
    """
    try:
        # Default folder path
        default_folder = 'C:/Users/AQ96560/OneDrive - ETS/Géophysique appliquée - GTO365 - Berlier-Bergman Time-Lapse/'

        # If file_paths is None, load all .AMP files from the default folder
        if file_paths is None:
            file_paths = glob.glob(os.path.join(default_folder, '*.AMP'))

        # Set default columns to delete if none provided
        if delete_columns is None:
            delete_columns = ['A(y)', 'A(z)', 'B(y)', 'B(z)', 'M(y)', 'M(z)', 'N(y)', 'N(z)', 'T(On)', 'T(0)', 'Error(ms)',
                              'T(N):01', 'T(N):02', 'T(N):03', 'T(N):04', 'T(N):05', 'T(N):06', 'T(N):07', 'T(N):08', 'T(N):09', 'T(N):10']
        
        # Ensure '32767' is always included in electrodes to clear
        if clear_electrodes is None:
            clear_electrodes = ['32767']
        else:
            # Convert all to string format and ensure '32768' is included
            clear_electrodes = list(set(map(str, clear_electrodes)) | {'32767'})

        all_dfs = []  # List to store DataFrames from each file
        survey_counter = 1  # Initialize the survey counter

        for file_path in file_paths:
            # Read the file and extract the relevant data
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Initialize variables
            start_time = None
            header = None
            data_start_index = None

            # Loop to find the start time and header line index
            for i, line in enumerate(lines):
                if start_time is None and 'Date & Time:' in line:
                    start_time_str = line.split(':', 1)[1].strip()
                    start_time = datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S").replace(second=0, microsecond=0)

                if header is None and 'No.' in line:
                    header = line.strip().split()
                    data_start_index = i + 1  # Set the data start index

            # Ensure both start time and header were found
            if start_time is None:
                raise ValueError(f"Error: Start time not found in the metadata for file {file_path}.")
            if header is None or data_start_index is None:
                raise ValueError(f"Error: Data header not found in the file {file_path}.")

            # Read the data section after the header line
            data = [line.strip().split() for line in lines[data_start_index:] if line.strip()]

            # Create a DataFrame with the header
            df = pd.DataFrame(data, columns=header)

            # Remove specified columns if they exist in the DataFrame
            df.drop(columns=[col for col in delete_columns if col in df.columns], inplace=True)

            # Check if 'App.Ch.(ms)' columns exist and rename them based on their index
            app_ch_indices = [i for i, col in enumerate(df.columns) if col == 'App.Ch.(ms)']
            for i, col_index in enumerate(app_ch_indices, start=1):
                df.columns.values[col_index] = f'App.Ch.{i:02}(ms)'

            # Rename columns to remove '(x)' and keep only the base name
            df.columns = df.columns.str.replace(r'/(x/)', '', regex=True)

            for electrode_col in ['A', 'B', 'M', 'N']:
                if electrode_col in df.columns:
                    # Convert column values to strings for comparison and filter rows
                    df = df[~df[electrode_col].astype(str).isin(clear_electrodes)]

            # Reset the index to make it continuous
            df = df.reset_index(drop=True)

            # Convert 'Time' column to numeric for processing
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

            # Identify survey breaks and label surveys
            df['Survey'] = survey_counter  # Initialize with current survey number

            if detect_by_first_measurement:
                # Use the first measurement to detect new surveys
                first_measurement = df.iloc[0][['A', 'B', 'M', 'N']].astype(str).values

                for i in range(1, len(df)):
                    # Compare the current measurement with the first measurement
                    current_measurement = df.iloc[i][['A', 'B', 'M', 'N']].astype(str).values

                    # Detect new survey when encountering the first measurement again
                    # or when A=1, B=9, M=2, N=10
                    if (
                        np.array_equal(current_measurement, first_measurement) or
                        (df.iloc[i]['A'] == '1' and df.iloc[i]['B'] == '9' and df.iloc[i]['M'] == '2' and df.iloc[i]['N'] == '10')
                    ):
                        survey_counter += 1  # Increment survey number

                    # Correctly assign the survey number to the DataFrame
                    df.at[i, 'Survey'] = survey_counter
                                     
            else:
                # Use time difference to detect new surveys
                for i in range(1, len(df)):
                    if pd.isna(df.at[i, 'Time']) or pd.isna(df.at[i - 1, 'Time']):
                        continue  # Skip if either current or previous 'Time' is NaN
                    if df.at[i, 'Time'] - df.at[i - 1, 'Time'] > 35:
                        survey_counter += 1  # Increment survey number for new survey
                    df.at[i, 'Survey'] = survey_counter
                    
            df['SurveyDate'] = None  # Initialize column with None
            df['MeasDate'] = None  # Initialize column with None

            current_survey_start_time = start_time  # Set initial survey start time

            # Iterate through the DataFrame to assign the survey date
            for i in range(len(df)):
                if i == 0 or df.iloc[i]['Survey'] != df.iloc[i - 1]['Survey']:
                    # Update the current survey start time for new surveys
                    current_survey_start_time = start_time + pd.to_timedelta(df.iloc[i]['Time'], unit='s')

                current_meas_date = start_time + pd.to_timedelta(df.iloc[i]['Time'], unit='s')

                # Assign measurement date to each row
                df.at[i, 'MeasDate'] = current_meas_date.strftime('%Y-%m-%d %H:%M')

                # Assign the survey date to each row
                df.at[i, 'SurveyDate'] = current_survey_start_time.strftime('%Y-%m-%d %H:%M')

            df = add_meas_k(df, n_elec_bh)
            df = add_meas_ele(df)
            df = df.apply(pd.to_numeric, errors='ignore')
            df["rhoa"] = df['k']*df['Res.(ohm)']

            # Append the DataFrame to the list
            all_dfs.append(df)

        # Find the longest header
        longest_header = max(all_dfs, key=lambda x: len(x.columns)).columns

        # Ensure all DataFrames have the same columns
        for df in all_dfs:
            for col in longest_header:
                if col not in df.columns:
                    df[col] = np.nan

        df_concat = pd.concat(all_dfs, ignore_index=False, join='outer')        # Concatenate all DataFrames into one large DataFrame
        return df_concat
    
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None

def files_to_convert(raw_data_folder, csv_data_folder):
    """
    Compare the files in two folders and return the paths of files that aren't in both.

    Parameters:
    - raw_data_folder: Path to folder with .AMP files
    - csv_data_folder: Path to folder with csv files

    Returns:
    - missing files addresses to convert
    """
    raw_data = set(os.path.splitext(file)[0] for file in os.listdir(raw_data_folder) if file.endswith('.AMP'))
    csv_data = set(os.path.splitext(file)[0] for file in os.listdir(csv_data_folder) if file.endswith('.csv'))

    to_convert = raw_data - csv_data

    to_convert_paths = [os.path.join(raw_data_folder, file + '.AMP') for file in to_convert]

    return to_convert_paths

# Add the __name__ guard
if __name__ == '__main__':

    fused_s4k_data = 'C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/fused_AMP_SAS4000.csv'

    # Choose a specific file to load
    #FLO_1= file + '08_BB_1211-1311_2h' + '.AMP'
    #FLO_2= file + '11_BB_1311-FLO_8mn' + '.AMP'
    #FLO_3= file + '12_BB_1311_1811_3h' + '.AMP'


    df = load_amp_files()

    df.to_csv(fused_csv_file, sep=';', index=False)

    """
    data_folder_path = 'D:/02_ERT_Data/All_Data/'
    csv_folder_path = 'D:/02_ERT_Data/All_Data_csv/'

    files = files_to_convert(data_folder_path, csv_folder_path)

    for file in files:
        df = load_amp_files([file], clear_electrodes=[], detect_by_first_measurement=True)
        new_file_path = os.path.join(csv_folder_path, os.path.basename(file).replace('.AMP', '.csv'))
        df.to_csv(new_file_path, sep=';')
    
    fused_data = 'D:/02_ERT_Data/fused_data_04oct24_15mar25.csv'
    fuse_csv_files(csv_folder_path, fused_data)"
    """
    


