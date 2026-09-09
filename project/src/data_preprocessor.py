import pandas as pd


def show_missing_values(train_x):
    print("\nПропуски до обработки:")
    print(train_x.isnull().sum().loc[lambda x: x > 0])


def fill_numeric_na(train_x, test_x):
    numeric_columns = train_x.select_dtypes(include=['number']).columns

    for column in numeric_columns:
        median = train_x[column].median()
        train_x[column] = train_x[column].fillna(median)
        test_x[column] = test_x[column].fillna(median)

    return train_x, test_x


def fill_categorical_na(train_x, test_x):
    categorical_columns = train_x.select_dtypes(include=['object', 'category', 'bool']).columns

    for column in categorical_columns:
        mode = train_x[column].mode()
        if len(mode) > 0:
            value = mode[0]
            train_x[column] = train_x[column].fillna(value)
            test_x[column] = test_x[column].fillna(value)

    return train_x, test_x


def one_hot_encode(train_x, test_x):
    train_x = pd.get_dummies(train_x, drop_first=True)
    test_x = pd.get_dummies(test_x, drop_first=True)
    return train_x, test_x


def align_columns(train_x, test_x):
    train_x, test_x = train_x.align(test_x, join='left', axis=1, fill_value=0)
    return train_x, test_x


def check_missing_after(train_x):
    print("\nПропуски после обработки:")
    print(train_x.isnull().sum().loc[lambda x: x > 0])