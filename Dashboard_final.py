import pandas as pd
import streamlit as st
import folium
from datetime import datetime
from streamlit_folium import folium_static 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import calmap

st.set_page_config(
    page_title="AQI Dash Board",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

df_gangapur = pd.read_csv('Gangapur_AQI_data.csv')
df_gangapur['Timestamp'] = pd.to_datetime(df_gangapur['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

df_csmt = pd.read_csv('CSMT_AQI_data.csv')
df_csmt['Timestamp'] = pd.to_datetime(df_csmt['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

df_gangapur_forecasted = pd.read_csv('Gangapur_Forecasted_AQI.csv')
df_gangapur_forecasted['Timestamp'] = pd.to_datetime(df_gangapur_forecasted['Timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')

df_csmt_forecasted = pd.read_csv('CSMT_Forecasted_AQI.csv')
df_csmt_forecasted['Timestamp'] = pd.to_datetime(df_csmt_forecasted['Timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')

# df = df_gangapur
# df_forecasted = df_gangapur_forecasted

def GIS_integration(df1, df2, df1_f, df2_f, timestamp_method):
    selected_time_df1 = None  # Initialize selected_time variable
    selected_data_df1 = None  # Initialize selected_data variable
    selected_time_df2 = None  # Initialize selected_time variable
    selected_data_df2 = None  # Initialize selected_data variable

    current_time = round_to_quarter_hour(datetime.now())
    st.markdown(f"""<h4 style='font-size: 17px'>Current Datetime for System Operations: {current_time}</h4>""", unsafe_allow_html=True)

    st.title("Air Quality Index (AQI) Map")

    if timestamp_method == "Last Available":
        selected_time_df1 = df1["Timestamp"].max()
        selected_data_df1 = df1[df1['Timestamp'] == selected_time_df1]

        # Find data related to the current time from df2
        selected_time_df2 = df2["Timestamp"].max()
        selected_data_df2 = df2[df2['Timestamp'] == selected_time_df1]
        
        # Concatenate the two DataFrames
        combined_data = pd.concat([selected_data_df1, selected_data_df2])
        selected_time = min(selected_time_df1, selected_time_df2)
        selected_map = create_map(selected_time, combined_data)
        if selected_map:
            st.write("Map for Last Data Available")
            folium_static(selected_map,width = 800)
        else:
            st.warning("No valid data points for creating the map for selected time.")

    else:  # Current Time    
        current_time = round_to_quarter_hour(datetime.now())

        # Find data related to the current time
        nearest_time_df1 = min(df1_f['Timestamp'], key=lambda x: abs(x - current_time))
        current_data_df1 = df1_f[df1_f['Timestamp'] == nearest_time_df1]

        nearest_time_df2 = min(df2_f['Timestamp'], key=lambda x: abs(x - current_time))
        current_data_df2 = df2_f[df2_f['Timestamp'] == nearest_time_df2]

        combined_data = pd.concat([current_data_df1, current_data_df2])
        selected_time = min(nearest_time_df1, nearest_time_df2)

        # Create Folium map for current time
        current_map = create_map(selected_time, combined_data)

        if current_map:
            st.write("Map for System Datetime")
            folium_static(current_map, width=800)
        else:
            st.warning("No valid data points for creating the map for current time.")

def create_map(selected_time, data):
    # Filter data based on selected time
    filtered_data = data[data['Timestamp'] == selected_time]

    # Drop rows with missing or invalid latitude and longitude values
    filtered_data = filtered_data.dropna(subset=['latitude', 'longitude'])

    if filtered_data.empty:
        return None  # Return None if there are no valid data points

    # Create Folium map
    mymap = folium.Map(location=[filtered_data['latitude'].mean(), filtered_data['longitude'].mean()], zoom_start=9)

    # Find the column with the maximum value (excluding certain columns)
    if 'AQI' in filtered_data.columns:
        aqi_column = 'AQI'
    elif 'Predicted_AQI' in filtered_data.columns:
        aqi_column = 'Predicted_AQI'
    else:
        return None  # Return None if AQI column not found

    max_column = filtered_data.drop(columns=['Timestamp', 'latitude', 'longitude', aqi_column]).idxmax(axis=1)

    for index, row in filtered_data.iterrows():
        max_column_name = max_column[index]
        max_column_value = row[max_column_name]
        
        # Customize the HTML content for the popup
        popup_content = f"<div style='font-size: 14px; width: 100px;'>" \
                        f"<p>{max_column_name}: {max_column_value}</p>" \
                        f"<p>{aqi_column}: {row[aqi_column]}</p>" \
                        f"</div>"
        
        # Create a popup with customized HTML content
        popup = folium.Popup(popup_content, max_width=250)
        
        # Add a marker to the map with the customized popup
        folium.Marker(location=[row['latitude'], row['longitude']], popup=popup).add_to(mymap)

    return mymap

# Function to round to the nearest quarter-hour
def round_to_quarter_hour(dt):
    dt = pd.to_datetime(dt)  # Convert numpy.datetime64 to datetime.datetime
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)

timestamp_method = st.sidebar.radio("Choose Timestamp Method", ["Last Available", "Current Time (Forecasted AQI)"])
GIS_integration(df_gangapur, df_csmt, df_gangapur_forecasted, df_csmt_forecasted,timestamp_method)

st.markdown(f"""<h2>Choose an Air Quality Monitoring Station:</h2>""", unsafe_allow_html=True)
selected_station = st.selectbox(
    "",
    ("Gangapur Road, Nashik", "Chhatrapati Shivaji International Airport (T2), Mumbai"),
    index=0
)
if(selected_station == "Gangapur Road, Nashik"):
    df = df_gangapur
    df_forecasted = df_gangapur_forecasted
elif(selected_station == "Chhatrapati Shivaji International Airport (T2), Mumbai"):
    df = df_csmt
    df_forecasted = df_csmt_forecasted

def gaugeChart():
    st.header("Concentration of pollutants")

    color_mapping = {
        'Green': '#00FF00',
        'LightGreen': '#90EE90',
        'Yellow': '#FFFF00',
        'LightRed': '#FF6347',
        'Red': '#FF0000',
        'DarkRed': '#8B0000'
    }

    # Define ranges and colors for each pollutant
    pollutants = {
        'PM10': {'range': [0, 50, 100, 250, 350, 430, 500], 'colors': ['Green', 'LightGreen', 'Yellow', 'LightRed', 'Red', 'DarkRed']},
        'PM2.5': {'range': [0, 30, 60, 90, 120, 250, 400], 'colors': ['Green', 'LightGreen', 'Yellow', 'LightRed', 'Red', 'DarkRed']},
        'NO2': {'range': [0, 40, 80, 180, 280, 400, 800], 'colors': ['Green', 'LightGreen', 'Yellow', 'LightRed', 'Red', 'DarkRed']},
        'CO': {'range': [0, 1, 2, 10, 17, 34, 50], 'colors': ['Green', 'LightGreen', 'Yellow', 'LightRed', 'Red', 'DarkRed']},
        'Ozone': {'range': [0, 50, 100, 168, 208, 748, 1250], 'colors': ['Green', 'LightGreen', 'Yellow', 'LightRed', 'Red', 'DarkRed']},
        'SO2': {'range': [0, 40, 80, 380, 800, 1600, 2000], 'colors': ['Green', 'LightGreen', 'Yellow', 'LightRed', 'Red', 'DarkRed']}
    }

    # Create subplots with three columns
    num_plots = len(pollutants)
    num_rows = (num_plots + 2) // 3  # Round up to ensure there are enough rows

    fig = make_subplots(rows=num_rows, cols=3, subplot_titles=list(pollutants.keys()), specs=[[{'type': 'indicator'}]*3]*num_rows)
    # fig.update_layout(height=num_rows * 300, showlegend=False)

    # Generate gauge plots for each pollutant
    for i, (pollutant, config) in enumerate(pollutants.items(), start=1):
        # Get the last value of the pollutant
        current_time = round_to_quarter_hour(datetime.now())

        nearest_time = min(df_forecasted['Timestamp'], key=lambda x: abs(x - current_time))

        last_line = df_forecasted[df_forecasted['Timestamp'] == nearest_time]

        last_value = last_line[pollutant].values[0]

        # Create a gauge plot using Plotly
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=last_value,
            # title={'text': f"{pollutant.upper()}"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, max(config['range'])]},
                   'bar': {'color': '#1e90ff'},
                   'steps': [{'range': [config['range'][i], config['range'][i + 1]], 'color': color_mapping[config['colors'][i]]}
                             for i in range(len(config['range']) - 1)],
                   }
        ), row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

    # Update subplot layout
    fig.update_layout(height=num_rows * 340, showlegend=False, width = 1000)

    # Display the plot
    st.plotly_chart(fig)


def aqi_range():
    categories = [
        {"name": "Good", "range": "0-50", "description": "The air is fresh and free from toxins. People are not exposed to any health risk.", "image_url": "good.png"},
        {"name": "Satisfactory", "range": "51-100", "description": "Acceptable air quality for healthy adults but mild threat to sensitive individuals.", "image_url":  "moderate.png"},
        {"name": "Moderate", "range": "101-200", "description": "Inhaling such air can cause slight discomfort and difficulty in breathing.", "image_url":  "poor.png"},
        {"name": "Poor", "range": "201-300", "description": "This could be typically problematic for children, pregnant women and the elderly.", "image_url":  "unhealthy.png"},
        {"name": "Very Poor", "range": "301-400", "description": "Exposure to air can cause chronic morbidities or even organ impairment.", "image_url":  "severe.png"},
        {"name": "Severe", "range": "401-500+", "description": "Beware! Your life is in danger. Prolonged exposure can lead to premature death.", "image_url":  "hazardous.png"}
    ]
    # Create cards for each category
    st.markdown("<h1 style='text-align: center;'>Air Quality Index Scale</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Know about the category of air quality index (AQI) your ambient air falls in and what it implies.</p>", unsafe_allow_html=True)

    # Define the number of columns
    num_columns = len(categories)

    column_width = 400


    # Display the cards in a single row
    row = st.columns(num_columns)

    for i in range(num_columns):
        with row[i]:
            category = categories[i]
            st.markdown(f"<h2 style='margin-bottom: 10px; text-align: center;'>{category['name']}</h2>", unsafe_allow_html=True)
            st.image(category['image_url'], width=200)
            st.markdown(f"<p style='margin-top: 10px; text-align: center;'><strong>Range:</strong> <span >{category['range']}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-weight: 33; text-align: center;'>{category['description']}</p>", unsafe_allow_html=True)
            st.markdown(f"<style>.st-ee {{ width: {column_width}px; }}</style>", unsafe_allow_html=True)
    for _ in range(5):
        st.empty()

def forcasting():
    # Assuming you have a DataFrame 'df' with 'Timestamp' and 'AQI' columns
    dfforecast = df_forecasted

    dfforecast =dfforecast.drop(['PM2.5', 'Ozone','PM10','CO','NO2', 'SO2'], axis=1)

    # Resample the data to daily frequency, taking the mean of AQI values for each day
    dfforecast = dfforecast.resample('D', on='Timestamp').mean().reset_index()
    dfforecast['Timestamp'] = dfforecast['Timestamp'].dt.date

    # Title of the app
    st.title('Forecasted Air Quality Index (AQI)')
    st.write('\n')
    st.write('\n')

    # Get current system date
    current_date = datetime.now().date()
    current_day_index = (current_date - dfforecast['Timestamp'].min()).days

    # Create a layout similar to the image
    cols = st.columns(7)
    for i in range(current_day_index, current_day_index + 7):
        if i >= len(dfforecast):  # If index exceeds available data
            break
        aqi = dfforecast["Predicted_AQI"][i]
        date = dfforecast['Timestamp'][i].strftime('%d %b')  # Format date as "day month"
        day = dfforecast['Timestamp'][i].strftime('%d')  # Extract day without leading zeros
        day_of_week = dfforecast['Timestamp'][i].strftime('%a')  # Abbreviated day of the week
        # Change color based on AQI value
        if aqi <= 50:
            color = '#00FF00'
        elif aqi <= 100:
            color = '#90EE90'
        elif aqi <= 200:
            color = '#FFFF00'
        elif aqi <= 300:
            color = '#FF6347'
        elif aqi <= 400:
            color = '#FF0000'
        else:
            color = '#8B0000'
        cols[i - current_day_index].markdown(f"""<div style='background-color: #3b5b8c; padding: 10px; border-radius: 10px;'>
                                             <h3 style='text-align: center; color: {color};font-size:5em'>{aqi:.0f}</h3>
                                             <h3 style='text-align: center; color: white; margin-bottom: 2px; padding-bottom: 2px'>{day_of_week}</h3>
                                             <h3 style='text-align: center; color: white; margin-top: 2px; padding-top: 2px'>{date}</h3></div>""", 
                                             unsafe_allow_html=True)

def health_advice_section():
    st.header("Health Advice for Air Pollution")

    # Add your health advice content here
    st.write("""
    Air pollution can have serious health effects, especially on vulnerable groups such as children, the elderly, and individuals with pre-existing health conditions. Here are some health advice tips to minimize the impact of air pollution:
    
    1. **Stay indoors during high pollution levels**: Limit outdoor activities, especially during times when air pollution levels are high.
    
    2. **Use air purifiers**: Consider using air purifiers indoors to reduce indoor air pollution levels.
    
    3. **Wear masks**: When outdoor activities are necessary, wear masks designed to filter out pollutants.
    
    4. **Keep indoor air clean**: Avoid smoking indoors and minimize the use of products that release pollutants indoors.
    
    5. **Stay informed**: Stay updated on air quality forecasts and take necessary precautions accordingly.
    
    6. **Seek medical advice**: If you experience symptoms such as coughing, shortness of breath, or chest tightness, seek medical advice promptly.
    """)

def plot_year_plot(df, year, background_color='black'):
    thresholds = [0, 50, 100, 150, 200, 250, 300]
    colors = ['darkgreen', 'green', 'yellow', 'orange', 'red', 'darkred']
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(20, 12))
    calmap.yearplot(df['AQI'], year=year, cmap=cmap, fillcolor='grey', linewidth=0.5, 
                    vmin=df['AQI'].min(), vmax=df['AQI'].max(), ax=ax)
    
    # Set background color
    fig.patch.set_facecolor(background_color)
    
    if background_color == 'black':
        text_color = 'white'
    else:
        text_color = 'black'
    
    plt.title(f'Calendar Plot of AQI for the Year {year}', color=text_color, fontweight='bold')
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(legend_handles, ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300'], 
                title='AQI Range', loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Set month and day names color and fontweight
    plt.setp(ax.get_xticklabels(), color=text_color, fontweight='bold')
    plt.setp(ax.get_yticklabels(), color=text_color, fontweight='bold')

    return fig

def historical_Analysis():
    st.title('Historical Analysis of AQI Data')

    # Sidebar: Year selection
    st.subheader('Select Year')
    selected_year = st.selectbox('', sorted(df['Timestamp'].dt.year.unique()))

    # Filter data for the selected year
    filtered_df = df[df['Timestamp'].dt.year == selected_year]

    # Horizontally arrange radio buttons
    st.subheader('Select Time Interval for Historical Analysis')

    time_intervals = ['15 min', '1 hr', '3 hr', '6 hr', '12 hr', '24 hr']
    selected_time_interval = st.radio (
        '',
        time_intervals,
        index=1,  # default index for 1 hour
        key='time_interval',
        horizontal=True
    )

    # Mapping time interval to Pandas resample format
    time_interval_map = {
        '15 min': '15T',
        '1 hr': '1H',
        '3 hr': '3H',
        '6 hr': '6H',
        '12 hr': '12H',
        '24 hr': '24H'
    }

    # Resample data based on selected time interval
    resampled_df = filtered_df.resample(time_interval_map[selected_time_interval], on='Timestamp').mean()

    # Line chart for historical analysis
    st.write("\n")
    st.write("\n")
    st.subheader('Historical Analysis')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resampled_df.index, y=resampled_df['AQI'], mode='lines', name='AQI'))
    fig.update_layout(title='AQI over Time', xaxis_title='Time', yaxis_title='AQI', width = 1200)
    st.plotly_chart(fig)

    # Calmap for the selected year
    st.subheader('Calmap for the Selected Year')
    filtered_df.set_index('Timestamp', inplace=True)
    calmap_fig = plot_year_plot(filtered_df, selected_year)
    st.pyplot(calmap_fig)

def plot_air_pollution_data(df):
    st.write('\n')
    st.title("AQI Forecasting Over a the Period of 2 Days: ")

    current_datetime = round_to_quarter_hour(datetime.now())
    # st.write(current_datetime)
    current_index = df[df['Timestamp'] == current_datetime].index[0]
    # st.write(current_index)

    # Get the previous and next 96 records
    start_index = max(0, current_index - 96)
    end_index = min(len(df), current_index + 96)
    df_filtered = df.iloc[start_index:end_index]

    # Define colors based on AQI levels
    colors = ['green' if aqi < 50 else 'lightgreen' if 50 <= aqi <= 100 else 'yellow' for aqi in df_filtered['Predicted_AQI']]

    # Define marker sizes
    marker_sizes = [10 if i == 96 else 5 for i in range(len(df_filtered))]

    # Create a scatter plot
    fig = go.Figure(data=go.Scatter(x=df_filtered['Timestamp'], y=df_filtered['Predicted_AQI'], mode='markers', marker=dict(color=colors, size=marker_sizes)))

    # Update layout
    fig.update_layout(title='Air Quality Index (AQI)',
                      xaxis_title='Timestamp',
                      yaxis_title='AQI',
                      hovermode='closest',
                      width=1200,
                      showlegend=False)
    
    # Set Y-axis range to start from 0
    fig.update_yaxes(range=[0, max(df_filtered['Predicted_AQI']) * 1.1])  # Adjust multiplier as needed

    st.plotly_chart(fig)
    return fig

def main():

    with st.sidebar.expander("About AQI", expanded=False):
        st.title('About AQI')
        st.header('What is AQI?')
        st.markdown('The Air Quality Index (AQI) is a measure used to communicate how polluted the air is and what associated health effects might be of concern. It is a numerical scale that ranges from 0 to 500, where lower values indicate better air quality.')
        st.header('How is it calculated?')
        st.markdown('AQI is calculated based on the concentrations of different pollutants in the air, including particulate matter (PM2.5 and PM10), ozone (O3), nitrogen dioxide (NO2), sulfur dioxide (SO2), and carbon monoxide (CO). The formula varies depending on the pollutant, but generally involves converting pollutant concentrations into a standardized scale and then selecting the highest value as the AQI.')

    #Speedometer Graph
    gaugeChart()

    forcasting()

    plot_air_pollution_data(df_forecasted)

    historical_Analysis()

    aqi_range()

    health_advice_section()


if __name__ == "__main__":
    main()
