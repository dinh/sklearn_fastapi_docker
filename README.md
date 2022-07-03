# Description
Ce projet est un exemple de déploiement du modèle de machine learning de prédiction de churn.
Le projet conntient deux applications dockerisées: 
* une API avec le framework Fastapi
* une application Web avec le framework Flask

# Démarage rapide
## Exécution de l'application dans un container 

* Installer ou mettre à jour Docker et Docker Compose pour disposer des dernières versions
* Installer make avec `sudo apt-get install make`
* Cloner le repo et aller dans le répertoire racine et exécuter `make up`

L'interface Web pour la prédiction de churn est à présent accessible à l'adresse suivante: ```http://127.0.0.1:8020/```

La documentation de l'API est accessible à l'adresse suivante: `http://127.0.0.1:8000/docs`

* Pour arrêter les services: `make stop`
* Pour arrêter les services et supprimer les volumes: `make down`

# Structure des répertoires


    sklearn_fastapi_docker        # racine du projet  
    ├─ backend/                   # code de l'API et de du script train.py qui entraine et sauvegarde le modèle  
    │  ├─ app/  
    │  │  ├─ core/  
    │  │  │  ├─ schemas/          # schémas pydendic  
    │  │  │  ├─ settings.py       # fichier de configuration  
    │  │  ├─ datasets/            # datasets pour l'entrainement  
    │  │  ├─ helpers/             # fonctions utilitaires  
    │  │  ├─ modeles/             # contient le modèle de ML généré, la dernière version du modèle, et l'historique du scoring  
    │  │  ├─ tests/  
    │  │  │  ├─ fixtures/         # données de tests  
    │  │  │  │  ├─ datasets/  
    │  │  │  ├─ modeles/          # contient le modèle de ML généré, la dernière version du modèle, et l'historique du scoring  
          
    ├─ frontend/                  # code de la Webapp (UI)  
    │  ├─ app/  
    │  │  ├─ statics/             # éléments statiques: css, javascript, images ...  
    │  │  │  ├─ styles/  
    │  │  ├─ templates/           # templates flask


# Développement

## Environnement virtuel

Installer virtualenv: `$ pip install virtualenv`

Vous pouvez installer un environnement virtuel pour chacune des applications frontend et backend.
Lorsque vous êtes dans l'une des répertoires, faire:

`$ virtualenv venv`

Pour pour activer un environnement virtuel:

`$ source venv/bin/activate`

# Installation des dépendances 

`$ pip install -r requirements.txt`

## Backend

### Entrainement et sauvegarde du modèle

```
cd backend/app
python train/train.py
```

Le modèle sera sauvegardé dans le répertoire model.

### API endpoints

Lancer le serveur:

`$ uvicorn main:app`

Vous pouvez tester les différents endpoints ici : http://localhost:8000/doc

### Test auto

Exécuter la commande `pytest` n'importe où dans le répertoire backend