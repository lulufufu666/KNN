# KNN Model Training and Evaluation

This project demonstrates how to train and evaluate a dataset using the K-Nearest Neighbors (KNN) algorithm.

## Installation and Dependencies

Please ensure that the following libraries are installed:

- pandas
- scikit-learn

You can install the dependencies using the following command:

```
pip install pandas scikit-learn
```

## Usage

1. Prepare the dataset:
    - Place the dataset file (train.csv) in the same directory as the code.
2. Execute the code:
    - Run the code file (train.py).
3. View the results:
    - The output will be divided into two parts:
        - Accuracy Score: The classification accuracy of the model on the test set.
        - AUC Score: The area under the ROC curve of the model on the test set.

## Parameter Description

You can customize the following parameters in the code:

- chunk_size: The number of rows to be read from the dataset at each iteration.
- test_size: The proportion of the test set.
- random_state: The random seed used to generate a stable split of the training and test sets.
