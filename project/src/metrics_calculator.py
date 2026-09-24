from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_classification_metrics(test_y, prediction, probability=None):

    accuracy = accuracy_score(test_y, prediction)

    precision = precision_score(
        test_y,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        test_y,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        test_y,
        prediction,
        average="weighted",
        zero_division=0
    )

    roc_auc = None

    if probability is not None:
        try:
            if probability.ndim == 2 and probability.shape[1] == 2:
                roc_auc = roc_auc_score(
                    test_y,
                    probability[:, 1]
                )
            else:
                roc_auc = roc_auc_score(
                    test_y,
                    probability,
                    multi_class="ovr"
                )
        except ValueError:
            roc_auc = None

    return accuracy, precision, recall, f1, roc_auc


def calculate_regression_metrics(test_y, prediction):

    mae = mean_absolute_error(
        test_y,
        prediction
    )

    rmse = mean_squared_error(
        test_y,
        prediction
    ) ** 0.5

    r2 = r2_score(
        test_y,
        prediction
    )

    return mae, rmse, r2