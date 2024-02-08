import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

# Get cluster labels for each data point in your dataset
labels = kmeans.labels_
print("Cluster Labels:", labels)

# Function to predict cluster for new data point
def predict_cluster(new_data_point):
    new_data_point_imputed = imputer.transform(new_data_point)
    new_data_point_standardized = scaler.transform(new_data_point_imputed)
    predicted_cluster = kmeans.predict(new_data_point_standardized)
    return predicted_cluster[0]

# Example new data point
new_data_point = np.array([[-67, -69, -64, -64, -59, -53, -71, -72, -71, -60, -61, -58, -54, -73, -60, -66, -72, -79, -72 ]])
# Predict cluster for the new data point
predicted_cluster = predict_cluster(new_data_point)

# Print the new data point
print("New Data Point:", new_data_point)

# Print the imputed and standardized new data point
new_data_point_imputed = imputer.transform(new_data_point)
new_data_point_standardized = scaler.transform(new_data_point_imputed)
print("Imputed and Standardized New Data Point:", new_data_point_standardized)

# Print the predicted cluster for the new data point
print("Predicted Cluster for New Data Point:", predicted_cluster)
