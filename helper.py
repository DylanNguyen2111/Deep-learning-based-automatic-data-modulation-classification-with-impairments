import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def preprocess_data(csv_file):
    data_dict = {}
    df = pd.read_csv(csv_file)
    try:
        header = df.columns[0]
        match = re.search(r'(\d+)dB', header.split('_')[1])
    except Exception as e:
        print(f"Error processing header '{header}': {header}. Skipping this column.")
    # Check if there is a match for the numeric part in the header
    if match:
        if header.split('_')[1].startswith('N') or header.split('_')[1].startswith('n'):
            SNR_level = -int(match.group(1))
        else:
            # If numeric part found, extract the SNR level
            SNR_level = int(match.group(1))
    else:
        # If no numeric part and no 'N', set SNR level to 0
        SNR_level = 0
    modulation_type = header.split('_')[0]
    fading = int(re.search(r'(\d+)Hz', header.split('_')[2]).group(1))
    key = tuple((modulation_type, SNR_level, fading))
    # Extract the column data from the DataFrame
    complex_column = df[header]
    try:
        complex_column = complex_column.apply(lambda x: re.sub(r'i.*$', 'i', x))
        complex_column = complex_column.apply(lambda x: x.replace('i', 'j') if ('i' in x or 'j' in x) and '.' in x else 0)
        complex_column = complex_column.apply(lambda x: complex(x))
    except ValueError as e:
        print(f"Error processing header '{header}': {e}.")
    real_part = complex_column.apply(lambda x: x.real).astype(np.float32)
    imaginary_part = complex_column.apply(lambda x: x.imag).astype(np.float32)
    
    data_dict[key] = np.array([real_part, imaginary_part])

    for key, value in data_dict.items():
            
        # Transpose signal data to shape (*, 128, 2)
        data_dict[key] = np.reshape(value.T, (1, 128, 2))

    return data_dict



def plot_signal_plotly(signal_data, modulation_type):
    """
    Plot the real and imaginary parts of a signal data array using Plotly.

    Parameters:
    signal_data (ndarray): Input signal data array with shape (1, N, 2),
                           where N is the number of data points.
    modulation_type (str): Modulation type , SNR level and Fading level.

    Returns:
    None (displays the interactive plot)
    """
    # Extract real and imaginary parts from the signal data
    real_part = signal_data[0, :, 0]
    imaginary_part = signal_data[0, :, 1]

    # Create a Plotly figure for an interactive line plot
    fig = go.Figure()

    # Add a line trace for the real part
    fig.add_trace(go.Scatter(x=np.arange(len(real_part)), y=real_part,
                             mode='lines', name='Real Part', line=dict(color='blue')))

    # Add a line trace for the imaginary part
    fig.add_trace(go.Scatter(x=np.arange(len(imaginary_part)), y=imaginary_part,
                             mode='lines', name='Imaginary Part', line=dict(color='orange', dash='dash')))

    # Update layout properties
    fig.update_layout(title={
        'text': modulation_type,
        'x': 0.4,  # Center align the title horizontally
        'y': 0.95,  # Position the title at the top (adjust value as needed)
        'xanchor': 'center',  # Anchor point for horizontal alignment
        'yanchor': 'top',  # Anchor point for vertical alignment
        'font': {
            'size': 30,  # Set the font size of the title
            'color': 'white'  # Set the font color of the title
        }
    },
                      xaxis_title='Time',
                      yaxis_title='Amplitude',
                      hovermode='closest',
                      width = 1600,
                      height = 700)

    # Show the interactive plot
    st.plotly_chart(fig)

modulation_output_map = {0: 'BPSK',
                         1: 'FSK2',
                         2: 'FSK4',
                         3: 'FSK8',
                         4: 'LFM100K',
                         5: 'LFM10M',
                         6: 'LFM1M',
                         7: 'PSK8',
                         8: 'QAM16',
                         9: 'QAM256',
                         10: 'QAM8',
                         11: 'QPSK'}