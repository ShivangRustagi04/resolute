import pandas as pd

# Read raw data
raw_data = pd.read_csv("C:\\Users\\shiva\\OneDrive\\Desktop\\rawdata.csv")  # Replace with the actual path and filename

# Assuming your raw data has columns like 'date', 'time', 'sensor', 'location', 'number', 'activity', 'position'
# Adjust column names based on your dataset
raw_data['datetime'] = pd.to_datetime(raw_data['date'] + ' ' + raw_data['time'])

# Extract date from datetime
raw_data['Date'] = raw_data['datetime'].dt.date

# Task 1: Datewise total duration for each inside and outside
duration_summary = raw_data.groupby(['Date', 'location'])['number'].sum().unstack(fill_value=0).reset_index()

# Task 2: Datewise number of picking and placing activity done
activity_summary = raw_data.groupby(['Date', 'activity'])['activity'].count().unstack(fill_value=0).reset_index()

# Merge the two summaries on the 'Date' column
output = pd.merge(duration_summary, activity_summary, on='Date')
output.columns = ['date', 'pick_activities', 'place_activities', 'inside_duration', 'outside_duration']
# Save the output to a new CSV file
output.to_csv("C:\\Users\\shiva\\OneDrive\\Desktop\\taskkk3.csv", index=False)
output.columns = ['date', 'pick_activities', 'place_activities', 'inside_duration', 'outside_duration']