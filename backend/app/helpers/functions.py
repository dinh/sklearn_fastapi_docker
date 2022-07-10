import os
import tempfile

import numpy as np
import pandas as pd
import joblib
import gzip
from io import BytesIO

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
# Not used in the code.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def predict(X, model):
    """
    It takes in a model and a data point, and returns the model's prediction for that data point

    :param X: The input data
    :param model: the model to use for prediction
    :return: The predicted value of the first row of the dataframe X
    """
    return model.predict(X)[0]


def get_churn_prediction(customer_data, model):
    """
    It takes a customer data object and a model, and returns a dictionary with the prediction, probability, and label

    :param customer_data: a Customer object
    :param model: the model we trained in the previous step
    :return: A dictionary with the label, prediction and probability of the customer churning.
    """
    X = pd.json_normalize(customer_data.__dict__)
    prediction = predict(X.values, model)
    probability = model.predict_proba(X.values)[0][prediction]
    label = "churner" if prediction == 1 else "non churner"
    return {
        'label': label,
        'prediction': int(prediction),
        'probability': round(probability, 2)
    }


def prepare_churn_dataset(df):
    """
    It takes a dataframe and returns a dataframe with the same columns, but with the following changes:

    - The customerID column is dropped
    - The gender, Partner, Dependents, PhoneService, Churn, and PaperlessBilling columns are converted to boolean values
    - The TotalCharges column is converted to a float
    - The remaining columns are converted to dummy variables
    - The resulting dataframe is reordered to match the column order in the selected_features list

    :param df: The dataframe to be processed
    :return: A dataframe with the selected features
    """
    # Preselected feature
    selected_features = [
        'tenure',
        'InternetService_No',
        'InternetService_Fiber optic',
        'OnlineSecurity_Yes',
        'DeviceProtection_Yes',
        'Contract_Month-to-month',
        'PaymentMethod_Electronic check',
        'PaperlessBilling',
        'Churn'
    ]
    # drop de customerID
    df.drop(columns='customerID', inplace=True)
    # df.set_index('customerID', inplace=True)
    # Transform labels to bools
    df.gender = pd.Series(np.where(df.gender.values == 'Male', 1, 0), df.index)
    df.Partner = pd.Series(np.where(df.Partner.values == 'Yes', 1, 0), df.index)
    df.Dependents = pd.Series(np.where(df.Dependents.values == 'Yes', 1, 0), df.index)
    df.PhoneService = pd.Series(np.where(df.PhoneService.values == 'Yes', 1, 0), df.index)
    df.Churn = pd.Series(np.where(df.Churn.values == 'Yes', 1, 0), df.index)
    df.PaperlessBilling = pd.Series(np.where(df.PaperlessBilling.values == 'Yes', 1, 0), df.index)
    # To fix TotalChurn data type
    # First replace empty fields in column TotalCharges by 0
    df.TotalCharges = pd.Series(np.where(df.tenure == 0, 0, df.TotalCharges), df.index)
    # Then change column TotalCharges to float type
    df.TotalCharges = pd.to_numeric(df.TotalCharges, downcast="float")

    df = pd.get_dummies(df)

    # Create required features columns if missing
    for feature in selected_features:
        if feature not in df:
            df[feature] = 0

    df = df.reindex(columns=selected_features)

    return df


def train_and_save_model(df, model_path, model_metric_path, model_version_path):
    """
    It takes a dataframe, trains a model, and saves the model to a file

    :param df: The dataframe containing the data to train the model on
    :param model_path: The path to the model file
    :param model_metric_path: The path to the file where we'll store the model's accuracy
    :param model_version_path: The path to the file that stores the model version
    """
    X, y = df.drop("Churn", axis=1), df.Churn
    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # define scaler
    scaler = MinMaxScaler()

    # Create an ensemble of 3 models
    estimators = [
        ('logistic', LogisticRegression()),
        ('cart', DecisionTreeClassifier()),
        ('gradient_booster', GradientBoostingClassifier(
            learning_rate=0.01,
            n_estimators=500,
            min_samples_split=3,
            max_features="log2",
            max_depth=3))
    ]

    # Create the Ensemble Model
    ensemble = VotingClassifier(estimators, voting='soft')

    # Make preprocess Pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', ensemble)
    ])

    pipeline.fit(X_train.values, y_train)

    # Test Accuracy
    print("Accuracy: %s%%" % str(round(pipeline.score(X_test.values, y_test), 3) * 100))

    # Export model to a temporary file
    temp_file_path = tempfile.NamedTemporaryFile(suffix='_churn_model', delete=False)
    with temp_file_path as f:
        joblib.dump(pipeline, gzip.open(f, "wb"))
        # replace temporary model with
        os.replace(f.name, model_path)

    # Persists the model metrics in model_metric_path, for comparisons with future iterations
    with open(model_metric_path, 'a') as f:
        f.write(str(round(pipeline.score(X_test.values, y_test), 3)) + '\n')
        f.close()

    # save model version
    version = 1
    if os.path.exists(model_version_path):
        with open(model_version_path) as f:
            line_value = f.readlines()
            if line_value:
                version = int(int(line_value[0])) + 1
        f.close()

    with open(model_version_path, 'w') as f:
        f.write(str(version))
        f.close()


def batch_file_predict(customer_data, model):
    """
    It takes a CSV file as input, and returns a CSV file as output

    :param customer_data: The data to be predicted
    :param model: The model object that you created in the previous step
    :return: A dataframe with the customerID, Churn, and Prediction columns.
    """
    buffer = BytesIO(customer_data)
    df = pd.read_csv(buffer)
    buffer.close()
    expected_columns = [
        'customerID',
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'tenure',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod',
        'MonthlyCharges',
        'TotalCharges',
        'Churn'
    ]

    # Create required features columns if missing
    for column in expected_columns:
        if column not in df:
            return f"Missing required column '{column}'"

    X = prepare_churn_dataset(df.copy())

    X = X.drop("Churn", axis=1)
    # Get batch prediction

    prediction = model.predict(X.values)  # use X.values to remove the headers to avoid the warning
    df_prediction = pd.DataFrame(prediction, columns=["Prediction"])
    df_prediction = df_prediction.replace({1: 'Yes', 0: 'No'})

    return pd.concat([df[["customerID", "Churn"]], df_prediction], axis=1)


def execute_pipeline(dataset_path, model_path, model_metric_path, model_version_path, message=""):
    """
    It reads the dataset, prepares it, trains a model and saves it

    :param dataset_path: the path to the dataset
    :param model_path: the path to save the model
    :param model_metric_path: The path to the model metrics file
    :param model_version_path: The path to the model version file
    :param message: a string that will be printed to the console when the task is finished
    """
    df = pd.read_csv(dataset_path)
    df = prepare_churn_dataset(df)
    train_and_save_model(df, model_path, model_metric_path, model_version_path)
    # TODO: send notification when the task has finished
    print(message)
