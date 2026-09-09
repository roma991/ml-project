from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_classification_metrics(test_y, prediction):
    accuracy = accuracy_score(test_y, prediction)
    precision = precision_score(test_y, prediction, average='weighted', zero_division=0)
    recall = recall_score(test_y, prediction, average='weighted', zero_division=0)
    f1 = f1_score(test_y, prediction, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1


def calculate_regression_metrics(test_y, prediction):
    mae = mean_absolute_error(test_y, prediction)
    rmse = mean_squared_error(test_y, prediction) ** 0.5
    r2 = r2_score(test_y, prediction)

    return mae, rmse, r2