from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("C:\\Users\\shiva\\OneDrive\\Desktop\\train.csv")  # Replace with the actual path and filename

# Assume 'target' is the name of the target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Function to predict the class for new data point
def predict_class(new_data_point):
    predicted_class = model.predict([new_data_point])
    return predicted_class[0]

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json['data']

        # Predict class for the new data point
        predicted_class = predict_class(data)

        return jsonify({'predicted_class': predicted_class})
    except ValueError as ve:
        return jsonify({'error': f'ValueError: {str(ve)}'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Route for displaying reasons to choose Random Forest Classifier
@app.route('/reasons')
def reasons():
    return render_template('reasons_rf.html')

if __name__ == '__main__':
    app.run(debug=True, port=4000)
