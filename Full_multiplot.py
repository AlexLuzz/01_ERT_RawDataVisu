import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import pygimli as pg
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from pygimli.physics import ert
import matplotlib.gridspec as gridspec
import os
from meteostat import Hourly, Stations
from datetime import datetime

# Set general font size
plt.rcParams.update({'font.size': 13})

def plot_raw_meas(df, filter_meas_ele=None, filter_meas=None, sample_factor=1, ax=None, plot_type='absolute', data_type='rhoa'):
    """
    Plot raw measurements: apparent resistivity, its variation, or IP raw data.

    Parameters:
    - df: DataFrame containing the data.
    - filter_meas_ele: Filter for specific measurement electrodes.
    - filter_meas: Filter for specific measurement types.
    - sample_factor: Sampling factor to reduce the number of points plotted.
    - ax: Matplotlib axis object (optional).
    - plot_type: 'absolute' for absolute values, 'variation' for relative variations.
    - data_type: 'rhoa' for apparent resistivity, 'ip' for IP decay data, 'R_ab' for R_ab [KOhm] data, 'Vmn_std' for Vmn_std [%] data.

    Returns:
    - fig, ax: The figure and axis objects.
    """
    # Apply filters if provided
    filtered_df = df.copy()
    if filter_meas_ele is not None:
        filtered_df = filtered_df[filtered_df['meas_ele'] == filter_meas_ele]

    if filter_meas is not None:
        filtered_df = filtered_df[filtered_df['meas'] == filter_meas]
    
    # Apply sampling if sample_factor > 1
    if sample_factor > 1:
        filtered_df = filtered_df.iloc[::sample_factor]

    # Create the figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.get_figure()

    grouped = filtered_df.groupby('meas')
    original_cmap = cm.Spectral_r
    new_cmap = original_cmap(np.linspace(0, 1, len(grouped)))

    for (i, (meas, group)) in enumerate(grouped):
        t = pd.to_datetime(group['SurveyDate'], format='%Y-%m-%d %H:%M')

        if data_type == 'rhoa':
            if plot_type == 'absolute':
                ax.scatter(t, group['rhoa'], color=new_cmap[i], label=f"{meas}, Channel:{i}, (k={group['k'].iloc[0]:.2f})", alpha=0.8, s=25)
            elif plot_type == 'variation':
                ratio_data = (group['rhoa'] / group['rhoa'].iloc[0] - 1) * 100
                ax.scatter(t, ratio_data, color=new_cmap[i], label=f"{meas}_ratios", alpha=0.6, s=25, marker='^')
        elif data_type == 'ip':
            ip_sum = group[[f'App.Ch.{j:02}(ms)' for j in range(1, 11)]].sum(axis=1)
            if plot_type == 'absolute':
                ax.scatter(t, ip_sum, color=new_cmap[i], label=f"{meas}, IP Sum", alpha=0.8, s=25)
            elif plot_type == 'variation':
                ratio_data = (ip_sum / ip_sum.iloc[0] - 1) * 100
                ax.scatter(t, ratio_data, color=new_cmap[i], label=f"{meas}_IP ratios", alpha=0.6, s=25, marker='^')
        elif data_type == 'R_ab':
            if plot_type == 'absolute':
                ax.scatter(t, group['R_ab [kOhm]'], color=new_cmap[i], label=f"{meas}, IP Sum", alpha=0.8, s=25)
            elif plot_type == 'variation':
                ratio_data = (group['R_ab [kOhm]'] / group['R_ab [kOhm]'].iloc[0] - 1) * 100
                ax.scatter(t, ratio_data, color=new_cmap[i], label=f"{meas}_IP ratios", alpha=0.6, s=25, marker='^')
        elif data_type == 'Vmn_std':
            if plot_type == 'absolute':
                ax.scatter(t, group['Vmn_std [%]'], color=new_cmap[i], label=f"{meas}, IP Sum", alpha=0.8, s=25)
            elif plot_type == 'variation':
                ratio_data = (group['Vmn_std [%]'] / group['Vmn_std [%]'].iloc[0] - 1) * 100
                ax.scatter(t, ratio_data, color=new_cmap[i], label=f"{meas}_IP ratios", alpha=0.6, s=25, marker='^')

    ylabel = {
        ('rhoa', 'absolute'): 'App. Resistivity : ρ app (ohm.m)',
        ('rhoa', 'variation'): 'App. Res. variations : Δρ/ρ (%)',
        ('ip', 'absolute'): 'IP Raw Data Sum',
        ('ip', 'variation'): 'IP Data variations : ΔIP/IP (%)',
        ('R_ab', 'absolute'): 'R_ab [kOhm]',
        ('R_ab', 'variation'): 'R_ab variations : ΔR_ab/R_ab (%)',
        ('Vmn_std', 'absolute'): 'Vmn_std [%]',
        ('Vmn_std', 'variation'): 'Vmn_std variations : ΔVmn_std/Vmn_std (%)'
    }
    ax.set_ylabel(ylabel[(data_type, plot_type)])

    ax.grid(visible=True, linestyle='--', alpha=0.6)
    ax.xaxis.set_visible(False)
    plt.close('all')

    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.6)

    return fig, ax

def plot_weather(start_date, end_date, ax=None, temp_step=2, precip_step=24, date=False, display_ref=False):
    """
    Fetches weather data from the YUL station in Montreal and plots precipitation and temperature.

    Parameters:
    - start_date (str): Starting date for data selection (format: 'YYYY-MM-DD').
    - end_date (str): Ending date for data selection (format: 'YYYY-MM-DD').
    - ax (Axes): Matplotlib axes object (optional).
    - temp_step (int): Step for temperature data aggregation.
    - precip_step (int): Step for precipitation data aggregation.

    Returns:
    - fig (Figure): The figure object containing the plots.
    - ax1 (Axes): The axes object for precipitation.
    - ax2 (Axes): The axes object for temperature (if plotted).
    """
    # Convert start_date and end_date to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

    stations = Stations()
    station = stations.region('CA', 'QC')

    # Fetch daily weather data for the specified date range
    data = Hourly('SOK6B', start_date, end_date)
    data = data.fetch()

    # Extract precipitation and temperature data
    precipitation = data['prcp'].fillna(0)
    temperature = data['temp']

    times = data.index

    # Aggregate data based on the specified steps
    times_aggregated_temp = times[::temp_step]
    times_aggregated_precip = times[::precip_step]
    precipitation_aggregated = [sum(precipitation[i:i + precip_step]) for i in range(0, len(precipitation), precip_step)]
    temperature_aggregated = [temperature.iloc[i] for i in range(0, len(temperature), temp_step)]

    # Create a new figure and axes if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()  # Get the figure from the provided axis

    # Plot precipitation data
    ax.bar(times_aggregated_precip, precipitation_aggregated, width=0.3, alpha=0.8, color='royalblue', label='Precipitation (mm)')
    ax.set_ylabel('Precipitation (mm)', color='royalblue')
    ax.tick_params(axis='y', labelcolor='royalblue')

    # Plot temperature data with color change below 0°C
    ax2 = ax.twinx()
    temp_colors = ['deepskyblue' if temp < 0 else 'orange' for temp in temperature_aggregated]
    ax2.plot(times_aggregated_temp, temperature_aggregated, color='orange', linestyle='-', linewidth=2.5, label='Temperature (°C)')
    for i in range(len(times_aggregated_temp) - 1):
        ax2.plot(
            times_aggregated_temp[i:i + 2],
            temperature_aggregated[i:i + 2],
            color=temp_colors[i],
            linestyle='-',
            linewidth=4,
        )
    ax2.set_ylabel('Temperature (°C)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add a horizontal line at 0°C
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    # Format x-axis for date and time
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right')

    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    # Add legends
    # ax.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    if date:
        date = pd.to_datetime(date)
        ax.axvline(x=date, color='red', linewidth=2, alpha=0.8)

    title = "Weather Data: Precipitation and Temperature"

    if display_ref:
        display_ref = pd.to_datetime(display_ref)
        ax.axvline(x=display_ref, color='grey', linewidth=1.5, alpha=0.7)
        title += f", ref survey:{display_ref}"
    
    # plt.title(title)

    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.6)

    return fig, ax

def getABMN(scheme, measurement_idxs):
    """ Get coordinates of four-point configuration with ids from `measurement_idxs` array in DataContainerERT `scheme`. """
    coords = {}
    for elec in "abmn":
        elec_ids = [int(scheme(elec)[idx]) for idx in measurement_idxs]
        elec_pos = [scheme.sensorPosition(elec_id) for elec_id in elec_ids]
        coords[elec] = [(pos.x(), pos.y()) for pos in elec_pos]
    return coords

def plotABMN(ax, scheme, measurement_idxs):
    """ Visualize four-point configuration for multiple measurements on given axes. """
    coords = getABMN(scheme, measurement_idxs)
    
    # Plot coordinates for 'a', 'b', 'm', 'n'
    for elec in coords:
        for i, (x, y) in enumerate(coords[elec]):
            # Plot the electrode positions
            if elec in "ab":
                ax.plot(x, y, marker="o", color='red', alpha =0.8, ms=8)
            if elec in "mn":
                ax.plot(x, y, marker="o", color='blue', alpha =0.8, ms=8)
            """
            offset_x = 10
            offset_y = 8
            if elec in "a":
                color = "red"
                xytext = (offset_x, offset_y)
            elif elec in "b":
                color = "red"
                xytext = (-offset_x, offset_y)
            elif elec in "m":
                color = "blue"
                xytext = (offset_x, -offset_y)
            elif elec in "n":
                color = "blue"
                xytext = (-offset_x, -offset_y)
            
            # Plot the electrode positions
            ax.plot(x, y, marker=".", color=color, ms=10)

            # Annotate the electrode names
            ax.annotate(elec.upper(), xy=(x, y), ha="center", fontsize=10, bbox=dict(
                boxstyle="round", fc=(0.8, 0.8, 0.8), ec=color), xytext=xytext,
                textcoords='offset points', arrowprops=dict(
                    arrowstyle="wedge, tail_width=.5", fc=color, ec=color,
                    patchA=None, alpha=0.75))
            """
    
    ax.set_title(f"Measurement sensitivity \n (up to 4 Channels)")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    txt_AB = f"A_B: {scheme['a'][measurement_idxs[i]]+1}_{scheme['b'][measurement_idxs[i]]+1} \n "
    txt_MNs = ""
    for i in range(len(measurement_idxs)):
        txt_MNs = txt_MNs + f"M_N_C{i}: {scheme['m'][measurement_idxs[i]]+1}_{scheme['n'][measurement_idxs[i]]+1} \n "
    ax.text(0.5, -1.2, txt_AB + txt_MNs, transform=ax.transAxes, fontsize=15, verticalalignment="top", horizontalalignment="center", bbox=props)
    return ax

def plot_sensitivity(jacobian, mesh, measurement_idx, ax=None, num_levels=6, cMin=3, cMax=4.5):
    jacobian_array = np.array(jacobian[measurement_idx, :].sum(axis=0))
    normsens = pg.utils.logDropTol(jacobian_array / mesh.cellSizes(), 8e-4)

    # Create a custom colormap (e.g., magma_r) and discretize it
    original_cmap = cm.afmhot_r
    new_cmap = original_cmap(np.linspace(0, 1, num_levels))
    # Convert it to a ListedColormap
    discrete_cmap = ListedColormap(new_cmap)

    # If ax is not provided, create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the sensitivity model with the defined colormap
    ax, cb = pg.show(mesh, abs(normsens), nLevs=num_levels, ax=ax, cMap=discrete_cmap, cMax=cMax, logscale=True, contourLines=True)

    plt.close('all')

    # Set title and labels
    ax.set_xlabel("X (m)")
    ax.set_ylim(-1.8,0.2)
    ax.set_ylabel("Z (m)")

    return ax, cb

def saveFiguresToPDF(figures, pdf_filename, figsize=(20, 10), verbose=False):
    """Save a list of figures to a multi-page PDF.

    Parameters:
        figures (list): List of Matplotlib figure objects to be saved.
        pdf_filename (str): The name of the output PDF file.
        figsize (tuple): Size of the figures. Default is (12, 7).
        verbose (bool): If True, prints additional information. Default is False.
        front_page (matplotlib.figure.Figure, optional): A front page figure to be added as the first page.
    """
    with PdfPages(pdf_filename) as pdf:
        for i, fig in enumerate(figures):
            if verbose:
                print(f"Saving figure {i + 1}/{len(figures)} to PDF.")
            fig.set_size_inches(figsize)  # Set the figure size
            pdf.savefig(fig, bbox_inches='tight')  # Save the current figure to the PDF
            plt.close(fig)  # Close the figure to free memory
    if verbose:
        print(f"All figures saved to {pdf_filename}.")

def draw_shapes(ax):
    """
    Draws lines on the given axis.
    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the lines on.
    lines (list of list of tuples): A list where each element is a list of points (tuples) defining a line.
    """
    lines = [[(-3.1, 0), (-0.2, 0), (-0.15, -0.2), (-3.1,-0.2)],  # Road
        [(-0.2, 0.1), (-0.05, 0.1), (-0, -0.4), (-0.25,-0.4), (-0.2, 0.1)],  # curb1
        [(-0.05, 0), (0.8, -0.05), (1.65, 0), (1.65, -1), (-0.05, -1), (-0.05, 0)],   # IV
        [(1.65, 0.1), (1.8, 0.1), (1.85, -0.4), (1.6,-0.4), (1.65, 0.1)],  # curb2
        [(1.85, 0), (3.1, 0), (3.1, -0.15), (1.8, -0.2)],  # Sidewalk
    ]
    for line in lines:
        x, y = zip(*line)  # Unzip the list of points into x and y coordinates
        ax.plot(x, y, linewidth = 2, color='grey', alpha=0.7)  # Plot the line with points

def plot_histogram(df, column, bins=20, ax=None):
    """
    Plot a histogram of the voltage values from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The name of the column containing the values.
    - bins (int): The number of bins for the histogram.
    - ax (matplotlib.axes.Axes): The axis to plot the histogram on. If None, a new figure and axis are created.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    - ax (matplotlib.axes.Axes): The axis object containing the plot.
    """
    # Create the figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    df[column] = pd.to_numeric(df[column], errors='coerce')

    # Plot the histogram
    ax.hist(df[column], bins=bins, color='blue', alpha=0.7, edgecolor='black')

    # Set labels and title
    ax.set_xlabel(column)
    ax.set_title(f"{column} Histogram")

    plt.close('all')

    return ax

if __name__ == "__main__":

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_home

    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/02 - Alexis Luzy/'

    stuff_path = Onedrive_path + '00_Github_Code/01_ERT_RawDataVisu/'

    df = pd.read_csv(Onedrive_path + 'ERT_Data/fused_SAS4000_OhmPi.csv', sep=';')
    
    # Where to save the multiplot figures
    save_path = os.path.join(Onedrive_path, 'ERT_Raw_Data_Visu/1erAvril_au_7mai_abs_VmnStd_Rab.pdf')

    # Filter data
    df = df[df['meas'] != 'Unknown']

    # For HRE : between 2024-11-18 and 2024-11-30
    # For OhmPi : after 2025-04-01
    df = df[df['SurveyDate'] > '2025-04-01 12:00:00']
    #df = df[df['SurveyDate'] < '2024-11-30 12:00:00']

    # Load data and mesh files
    data_file = os.path.join(stuff_path, 'stuff/data_20_11_2024_20_20.ohm')
    mesh_file = os.path.join(stuff_path, 'stuff/mesh_high_densite.bms')
    one_df_file = os.path.join(stuff_path, 'stuff/06_BB_1211_FAST-2STK_KR.csv')
    one_df = pd.read_csv(one_df_file, sep=';')

    # Load Jacobian matrix and calculate normalized sensitivity
    jacobian = np.load(os.path.join(stuff_path, "stuff/jacobian_matrix.npy"))

    data = pg.load(data_file, verbose=True)
    mesh = pg.load(mesh_file)

    # Setup ERT modeling
    fop = ert.ERTModelling()
    fop.setData(data)
    fop.setMesh(mesh)

    # To plot sensitivity of one measurement
    meas_ele = df['meas_ele'].unique()
    figs = []
    for meas in meas_ele:
        
        filtered_df = df.copy()
        filtered_df = filtered_df[filtered_df['meas_ele'] == meas]

        # Create figure and GridSpec layout
        fig = plt.figure(figsize=(20, 7))  # Adjust figure size if needed
        gs = gridspec.GridSpec(5, 5)  # 5 rows, 5 columns
        fig.subplots_adjust(wspace=1.5)

        # Create subplots in specified positions
        ax1 = fig.add_subplot(gs[0:2, 0:4])  # First figure: 2 rows, 4 columns
        ax2 = fig.add_subplot(gs[2:4, 0:4], sharex=ax1)  # Second figure: 2 rows, 4 columns
        ax3 = fig.add_subplot(gs[4, 0:4], sharex=ax1)    # Third figure: 1 row, 4 columns
        ax4 = fig.add_subplot(gs[2:4, 4])      # Fourth figure: 2 row, 1 column
        ax5 = fig.add_subplot(gs[0, 4])      # Fifth figure: 1 row, 1 column
        ax6 = fig.add_subplot(gs[1, 4])      # Sixth figure: 1 row, 1 column

        # To plot sensitivity of one measurement
        index_jaco = one_df.loc[one_df['meas_ele'] == meas].index

        # Plotting rhoa or IP data, absolute or variation
        plot_raw_meas(filtered_df, ax=ax1, plot_type='absolute', data_type='Vmn_std')
        plot_raw_meas(filtered_df, ax=ax2, plot_type='absolute', data_type='R_ab') 

        plot_sensitivity(jacobian, fop.paraDomain, index_jaco, ax=ax4)
        plotABMN(ax4, data, index_jaco)
        draw_shapes(ax4)
        plot_histogram(filtered_df, column='I(mA)', ax=ax5, bins=5)  # Voltage(V)
        plot_weather(filtered_df['SurveyDate'].iloc[0], filtered_df['SurveyDate'].iloc[-1], ax=ax3)

        plt.tight_layout()  # Adjust spacing
        plt.close('all')

        figs.append(fig)

    saveFiguresToPDF(figs, save_path)
