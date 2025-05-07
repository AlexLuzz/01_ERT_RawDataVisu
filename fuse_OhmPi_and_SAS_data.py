import os
import pandas as pd
import glob
import numpy as np

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

def fuse_csv_files(folder_path, output_file):
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # List to store DataFrames
    all_dfs = []
    survey_counter = 487  # Initialize survey counter
    for file in csv_files:
        # Read the CSV file
        df = pd.read_csv(file, sep=',')

        # Extract the last 15 characters of the filename and convert to datetime
        survey_date_str = os.path.basename(file)[-19:-4]
        df['SurveyDate'] = pd.to_datetime(survey_date_str, format='%Y%m%dT%H%M%S', errors='coerce')
        df['SurveyDate'] = pd.to_datetime(df['SurveyDate']).dt.strftime('%Y-%m-%d %H:%M')
        # Convert 'time' column to datetime using the specified format and then to numeric (timestamp)
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

        df['Survey'] = survey_counter
        survey_counter += 1

        df = add_meas_k(df, 8)
        df = add_meas_ele(df)
        df["rhoa"] = df['k']*df['R [Ohm]']

        # Append the DataFrame to the list
        all_dfs.append(df)

    # Concatenate all DataFrames
    df_concat = pd.concat(all_dfs, ignore_index=True)
    # Save the concatenated DataFrame to a new CSV file
    df_concat.to_csv(output_file, sep=';', index=False)

def fuse_sas4k_OhmPi_data(fused_sas4k_file, fused_ohmpi_file, output_file, column_mapping):
    """
    Fuse and harmonize SAS4000 and OhmPi data by renaming columns and saving the result to a CSV file.

    Parameters:
    - fused_sas4k_file: Path to the fused SAS4000 CSV file.
    - fused_ohmpi_file: Path to the fused OhmPi CSV file.
    - output_file: Path to save the final fused and harmonized CSV file.
    - column_mapping: Dictionary mapping old column names to new column names.
    """
    # Read the fused SAS4000 and OhmPi CSV files
    sas4k_df = pd.read_csv(fused_sas4k_file, sep=';')
    ohmpi_df = pd.read_csv(fused_ohmpi_file, sep=';')

    # Rename columns in the OhmPi DataFrame using the mapping
    ohmpi_df.rename(columns=column_mapping, inplace=True)

    # Concatenate the two DataFrames
    fused_df = pd.concat([sas4k_df, ohmpi_df], ignore_index=True)

    # Save the fused DataFrame to a new CSV file
    fused_df.to_csv(output_file, sep=';', index=False)

# Example usage
if __name__ == '__main__':

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_home

    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/'
    fused_s4k_data = Onedrive_path + '02 - Alexis Luzy/ERT_Data/fused_AMP_SAS4000.csv'
    ohmpi_data_folder = Onedrive_path + 'Géophysique appliquée - GTO365 - 03 - Ohmpi - IV à Laval/'
    fused_ohmpi_data = Onedrive_path + '02 - Alexis Luzy/ERT_Data/fused_OhmPi.csv'

    #fused_s4k_data = 'C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/fused_AMP_SAS4000.csv'
    #ohmpi_data_folder = 'C:/Users/AQ96560/OneDrive - ETS/Géophysique appliquée - GTO365 - 03 - Ohmpi - IV à Laval/'
    #fused_ohmpi_data = 'C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/fused_OhmPi.csv'
    
    fuse_csv_files(ohmpi_data_folder, fused_ohmpi_data)

    # Dictionary to map old column names to new column names
    column_mapping = {
        # OhmPi columns > SAS4000 columns
        'time': 'MeasDate',
        'Vmn [mV]': 'Voltage(V)',
        'I [mA]': 'I(mA)',
        'R [Ohm]': 'Res.(ohm)',
        'R_std [%]': 'Error(%)',
    }

    fuse_sas4k_OhmPi_data(fused_s4k_data, fused_ohmpi_data, Onedrive_path + '02 - Alexis Luzy/ERT_Data/fused_SAS4000_OhmPi.csv', column_mapping)