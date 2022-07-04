# API de prédiction du churn.

# Entrainement et création du modèle de ML

## Données d'entrainement
Les données d'entrainement doivent être dans un fichier nommé churn.csv.
Le contenu du fichier churn.csv est au format suivant :

```
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7010-BRBUU,Male,0,Yes,Yes,72,Yes,Yes,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Two year,No,Credit card (automatic),24.1,1734.65,No
9688-YGXVR,Female,0,No,No,44,Yes,No,Fiber optic,No,Yes,Yes,No,Yes,No,Month-to-month,Yes,Credit card (automatic),88.15,3973.2,No
 ```

### Copier le fichier dans le répertoire /app/datasets/ du conteneur

`$ cat churn.csv | docker exec churn-api bash -c 'cat > /app/datasets/churn.tmp; mv -f /app/datasets/churn.tmp datasets/churn.csv`

## Entrainement et sauvegarde du modèle

L'exécution de :

`$ docker exec churn-api python train.py` 

ou de 

```
$ curl -X 'GET' \
'http://127.0.0.1:8000/train' \
-H 'accept: application/json'
```

génères trois fichiers :

* Le modèle ML: models/churn-model.dat.gz
* Historique des mesures de performance du modèle à des fins de supervision: models/churn-model-metrics.txt
* La dernière version du modèle : model/churn-model-latest-version.txt. Le numéro de version est incrémenté à chaque entrainement

Pour récupérer ces fichiers sur la machine host :

```
$ docker exec churn-api cat models/churn-model.dat.gz > churn-model.dat.gz
$ docker exec churn-api cat models/churn-model-metrics.txt > churn-model-metrics.txt
$ docker exec churn-api cat models/churn-model-latest-version.txt > churn-model-latest-version.txt
```

> La création du fichier du modèle ML est "safe". Lors de l'opération d'entrancement, un fichier temporaire est créé.
> Une fois le fichier est créé, on remplace le fichier churn-model.dat.gz par une opération atomique.

# Prédiction par bacth

Il est possible de faire des prédictions par lot (batch) en utilsant l'API `/bacth-predict` :

```
curl -X 'POST' \
'http://127.0.0.1:8000/batch-predict' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@churn.csv;type=text/csv'
```

où le fichier churn.csv est au format suivant :
```
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7010-BRBUU,Male,0,Yes,Yes,72,Yes,Yes,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Two year,No,Credit card (automatic),24.1,1734.65,No
9688-YGXVR,Female,0,No,No,44,Yes,No,Fiber optic,No,Yes,Yes,No,Yes,No,Month-to-month,Yes,Credit card (automatic),88.15,3973.2,No
 ```