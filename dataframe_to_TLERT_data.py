from datetime import datetime
from tools.tools import load_amp_files, create_crosshole_data_df, save_data
import pandas as pd
from pygimli.physics import ert

file = 'C:/Users/alexi/OneDrive - ETS/00-Maitrise_Recherche/01-Geophysique/00-Berlier-Bergman/06-Mesures_SAS4000/Mesures TL-ERT SAS4000 2024/'
FLO_1= file + '08_BB_1211-1311_2h' + '.AMP'
FLO_2= file + '11_BB_1311-FLO_8mn' + '.AMP'
FLO_3= file + '12_BB_1311_1811_3h' + '.AMP'


#df = load_amp_files([FLO_1,FLO_2,FLO_3], clear_electrodes=[18], detect_by_first_measurement=False)

#df.to_csv('D:/02_ERT_Data/FLOOD_V2.csv', sep=';', index=False)

df = pd.read_csv('D:/02_ERT_Data/FLOOD_V2.csv', sep=';')

#df['Res.(ohm)'] = abs(df['Res.(ohm)'])

# Flood test
#df = df[df['SurveyDate'] > '2024-11-08 00:00:00']
#df = df[df['SurveyDate'] < '2024-11-18 00:00:00']

# Raining Even
#df = df[df['SurveyDate'] > '2024-11-21 00:00:00']
#df = df[df['SurveyDate'] < '2024-11-26 00:00:00']

# Période de redoux
#df = df[df['SurveyDate'] > '2025-02-15 00:00:00']
#df = df[df['meas'] != 'Unknown']

DATA, survey_dates, _  = create_crosshole_data_df(df)

# Convert survey_dates from strings to datetime objects
survey_dates_datatime = [datetime.strptime(date, '%Y-%m-%d %H:%M') for date in survey_dates]

file = 'D:/01-Coding/01_BB_ERT/03_TEST/'
date_1 = survey_dates_datatime[0].strftime('%m-%d_%Hh')
date_2 = survey_dates_datatime[-1].strftime('%m-%d_%Hh')

#filename = f"{file}REDOUX_{date_1}_{date_2}"
filename = f"{file}FLOOD_V2"

inv = ert.TimelapseERT(DATA=DATA, times=survey_dates_datatime)
inv.saveData(filename, masknan=True)

#save_data(DATA, survey_dates_datatime, filename)


