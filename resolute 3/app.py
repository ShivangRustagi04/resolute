from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Read raw data
raw_data = pd.read_csv("C:\\Users\\shiva\\OneDrive\\Desktop\\rawdata.csv")  # Replace with the actual path and filename

# Convert 'date' and 'time' columns to datetime
raw_data['datetime'] = pd.to_datetime(raw_data['date'] + ' ' + raw_data['time'])

# Extract date from Timestamp
raw_data['Date'] = raw_data['datetime'].dt.date

# Task: Count 'date', 'pick_activities', 'place_activities', 'inside_duration', 'outside_duration'
count_summary = raw_data.groupby(['Date']).agg(
    date_count=('date', 'count'),
    pick_activities_count=('activity', lambda x: (x == 'picked').sum()),  # Count 'pick' activities
    place_activities_count=('activity', lambda x: (x == 'placed').sum()),  # Count 'place' activities
    inside_count=('position', lambda x: (x == 'inside').sum()),  # Count 'inside' occurrences
    outside_count=('position', lambda x: (x == 'outside').sum())  # Count 'outside' occurrences
).reset_index()

# Save the output to a new CSV file with only the specified columns
count_summary.to_csv("C:\\Users\\shiva\\OneDrive\\Desktop\\output.csv", columns=['Date', 'date_count', 'pick_activities_count', 'place_activities_count', 'inside_count', 'outside_count'], index=False)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html', tables=[count_summary.to_html(classes='data')], titles=count_summary.columns)

if __name__ == '__main__':
    app.run(debug=True, port=7000)
