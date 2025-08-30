#!/bin/bash

# Ce script vérifie les logs du conteneur et tente de le démarrer
echo "Vérification des logs du conteneur..."
docker logs fastapi-app

echo "Tentative de démarrage du conteneur..."
docker start fastapi-app

echo "Vérification du statut après tentative de démarrage..."
docker ps -a | grep fastapi-app
