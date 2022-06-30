import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def load_model():
    """
    Loads and returns the pretrained model
    """
    model = joblib.load("model/model_binary.dat.gz")
    print("Model loaded")
    return model


def predict(X, model):
    return model.predict(X)[0]

model = joblib.load("model/model_binary.dat.gz")

def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    #model=load_model()

    prediction = predict(X.values, model)
    probability = model.predict_proba(X.values)[0][prediction]
    label = "churner" if prediction == 1 else "non churner"
    return {
        'label': label,
        'prediction': int(prediction),
        'probability': round(probability, 2)
    }
	
def prepare_data(df):
    #if file is not None:
    # importing the dataset
    #file="data/batch_churn.csv"
    #df = pd.read_csv(file,sep=',')
    #df = pd.read_csv(file)
    # drop de customerID qui n'est pas une information utile
    #df.drop(columns='customerID', inplace=True)
    # Transformation des labels en données binaires
    df.gender = pd.Series(np.where(df.gender.values == 'Male', 1, 0), df.index)
    df.Partner = pd.Series(np.where(df.Partner.values == 'Yes', 1, 0), df.index)
    df.Dependents = pd.Series(np.where(df.Dependents.values == 'Yes', 1, 0), df.index)
    df.PhoneService = pd.Series(np.where(df.PhoneService.values == 'Yes', 1, 0), df.index)
    df.SeniorCitizen = pd.Series(np.where(df.SeniorCitizen.values == 'Yes', 1, 0), df.index)
    df.PaperlessBilling = pd.Series(np.where(df.PaperlessBilling.values == 'Yes', 1, 0), df.index)
    # Correction du type de données de TotalChurn
    # on commence par remplacer les champs vide de la colonne TotalCharges par 0 
    df.TotalCharges = pd.Series(np.where(df.tenure == 0, 0, df.TotalCharges), df.index)
    # On change ensuite le type de la colonne TotalCharges en type numérique float 
    # df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.TotalCharges = pd.to_numeric(df.TotalCharges, downcast="float")
    #feature scaling
    sc = StandardScaler()
    #df['tenure'] = sc.fit_transform(df[['tenure']])
    #df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    #df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])
    df = df[['tenure','PaperlessBilling','InternetService','OnlineSecurity','DeviceProtection','Contract','PaymentMethod']]
    columns=['tenure', 
   	       'PaperlessBilling',
   	       'InternetService_Fiber optic',
   	       'InternetService_No',
               'OnlineSecurity_Yes',
  	       'DeviceProtection_Yes',
   	       'Contract_Month-to-month',
    	       'PaymentMethod_Electronic check']
    df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    #Encoding the other categorical categoric features with more than two categories
    return df
def batch_file_predict(clean_data,df_initial):    
    model=load_model()
    #Get batch prediction
    prediction = model.predict(clean_data)
    prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
    prediction_df = prediction_df.replace({1:'Yes',
                                           0:'No'})
    return pd.concat([df_initial,prediction_df], axis = 1)
	 
       

