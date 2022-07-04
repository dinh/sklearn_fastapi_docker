# Description

API de prédiction du churn.

# Choix techniques

## Gestion des erreurs

Toutes les erreurs applicatives renvoient un code HTTP 200.
Le code de l'erreur est dans le champ "status_code" de la réponse JSON.