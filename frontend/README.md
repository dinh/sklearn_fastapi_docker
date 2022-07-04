# Description

Une simple Webapp utilisant l'API de prédiction
URL: ```http://127.0.0.1:8020/```

## Requirements

L'API a étté configuré pour accepter les requêtes en provenance des IPs, hostname et ports suivants :  

```
BACKEND_CORS_ORIGINS = [
"http://localhost",
"http://localhost:5000",
"http://127.0.0.1:5000",
"http://127.0.0.1:8020",
"http://localhost:8020"
]
```

Si vous utilisez une adresse IP, un hostname ou un port différent, il faudra le rajouter dans 
le répertoire `backend/app/core/settings.py`