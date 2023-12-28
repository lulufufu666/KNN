import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Read the first chunk of the dataset
chunk_size = 1000  # Number of rows to read at a time
data_chunks = pd.read_csv('D:\\train.csv', chunksize=chunk_size)
first_chunk = next(data_chunks)
first_chunk.fillna(first_chunk.mean(), inplace=True)

# Split features and target variable
X = first_chunk.drop('target', axis=1)
y = first_chunk['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Create a standard scaler
scaler = StandardScaler()

# Fit the scaler to the training set of the first chunk
scaler.fit(X_train)

# Scale the testing set
X_test = scaler.transform(X_test)

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=15)

# Read and train through each chunk
for chunk in data_chunks:

    # Split features and target variable
    chunk.fillna(chunk.mean(), inplace=True)
    X = chunk.drop('target', axis=1)
    y = chunk['target']

    # Scale the features of each chunk's training set
    X = scaler.fit_transform(X)

    # Fit the KNN model
    knn.fit(X, y)

# Predict on the testing set
y_pred = knn.predict(X_test)

# Evaluate the model performance on the testing set
accuracy = round(knn.score(X_test, y_test), 5)
print("Accuracy on testing set: {:.3f}".format(roc_auc_score(y_test, y_pred)))
print("Accuracy:", accuracy)
print("classification report:\n", classification_report(y_test, knn.predict(X_test),
                                                        target_names=["non-5g", "5g"]))
