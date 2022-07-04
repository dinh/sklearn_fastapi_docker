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
    return model.predict(X)[0]


def get_churn_prediction(customer_data, model):
    X = pd.json_normalize(customer_data.__dict__)
    prediction = predict(X.values, model)
    probability = model.predict_proba(X.values)[0][prediction]
    label = "churner" if prediction == 1 else "non churner"
    return {
        'status_code': 200,
        'label': label,
        'prediction': int(prediction),
        'probability': round(probability, 2)
    }


def prepare_churn_dataset(df):
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
    buffer = BytesIO(customer_data)
    df = pd.read_csv(buffer)
    buffer.close()

    X = prepare_churn_dataset(df.copy())

    X = X.drop("Churn", axis=1)
    # Get batch prediction

    prediction = model.predict(X.values)  # use X.values to remove the headers to avoid the warning
    df_prediction = pd.DataFrame(prediction, columns=["Prediction"])
    df_prediction = df_prediction.replace({1: 'Yes', 0: 'No'})

    return pd.concat([df[["customerID", "Churn"]], df_prediction], axis=1)


def execute_pipeline(dataset_path, model_path, model_metric_path, model_version_path, message=""):
    df = pd.read_csv(dataset_path)
    df = prepare_churn_dataset(df)
    train_and_save_model(df, model_path, model_metric_path, model_version_path)
    # TODO: send notification when task finish
    print(message)
