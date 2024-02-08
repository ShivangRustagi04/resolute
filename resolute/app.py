from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load your data
your_data = np.genfromtxt("C:\\Users\\shiva\\OneDrive\\Desktop\\train.csv", delimiter=',')

# Impute missing values (replace NaNs with the mean of each feature)
imputer = SimpleImputer(strategy='mean')
your_data_imputed = imputer.fit_transform(your_data)

# Standardize the data
scaler = StandardScaler()
your_data_standardized = scaler.fit_transform(your_data_imputed)

# Apply K-Means clustering
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(your_data_standardized)

# Function to predict cluster for a new data point
def predict_cluster(new_data_point):
    new_data_point_imputed = imputer.transform(new_data_point)
    new_data_point_standardized = scaler.transform(new_data_point_imputed)
    predicted_cluster = kmeans.predict(new_data_point_standardized)
    return predicted_cluster[0]

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()['data']

        # Convert data to a list of floats
        new_data_point = [float(value) for value in data]

        # Convert the list to a NumPy array
        new_data_point_array = np.array([new_data_point])

        # Predict cluster for the new data point
        predicted_cluster = predict_cluster(new_data_point_array)

        return jsonify({'predicted_cluster': str(predicted_cluster)})

    except ValueError as ve:
        return jsonify({'error': f'ValueError: {str(ve)}'})
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)
