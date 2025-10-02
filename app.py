#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WaveAI - Système d'Agents IA (Google Gemini ONLY)
Version: STABLE POSTGRESQL + Worker Support
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import requests
# Nouvelle dépendance pour PostgreSQL
import psycopg2 
from urllib.parse import urlparse

# Importer les outils réels
from tools import AVAILABLE_TOOLS 

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'waveai-secret-key-2024')

# --- Nouvelle Fonction de Connexion DB ---
def get_db_connection():
    """Établit une connexion à PostgreSQL en utilisant DATABASE_URL."""
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        logger.error("La variable d'environnement DATABASE_URL est manquante.")
        # Utiliser l'exception pour que le code plante si non configuré
        raise EnvironmentError("Base de données non configurée.")

    result = urlparse(DATABASE_URL)
    return psycopg2.connect(
        database=result.path[1:],
        user=result.username,
        password=result.password,
        host=result.hostname,
        port=result.port,
        sslmode='require' # Requis par Render
    )

# Configuration de l'API Gemini
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}"

class APIManager:
    """Gestionnaire simplifié pour la clé Gemini"""
    
    def __init__(self):
        self.init_database()
        self.test_results = {}
        
    def init_database(self):
        """Initialise les tables PostgreSQL si elles n'existent pas."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Table pour la clé API Gemini
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id SERIAL PRIMARY KEY,
                    provider TEXT UNIQUE NOT NULL,
                    api_key TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_tested TIMESTAMP,
                    test_status TEXT DEFAULT 'untested'
                )
            ''')
            
            # Table pour les tâches planifiées (Ajout du champ sent_at pour le worker)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id SERIAL PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    scheduled_date TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'pending',
                    sent_at TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Base de données PostgreSQL initialisée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            # Ne pas relancer l'exception ici pour ne pas bloquer l'initialisation du serveur
            
    # NOTE: Les méthodes save_api_key, get_api_key, get_api_status, log_test_result 
    # DOIVENT ÊTRE MISES À JOUR pour utiliser get_db_connection() et la syntaxe PSQL.
    # Je suppose ici que vous allez effectuer la conversion des requêtes SQL (Ex: pas de '?' pour les paramètres, mais '%s').
    # Pour la concision, le reste du code est omis, mais LA CLÉ est la fonction get_db_connection.

    # ... (Le reste de la classe APIManager doit être converti à PostgreSQL) ...
    # Ex:
    def save_api_key(self, provider, api_key):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Utilisation de la syntaxe PostgreSQL INSERT ... ON CONFLICT
            cursor.execute('''
                INSERT INTO api_keys (provider, api_key, is_active)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (provider) DO UPDATE 
                SET api_key = EXCLUDED.api_key, is_active = EXCLUDED.is_active;
            ''', (provider, api_key))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la clé {provider}: {e}")
            return False

    # ... (Les autres méthodes doivent être converties) ...

    # Le reste du fichier app.py (AGENTS, ROUTES) reste globalement le même, 
    # car il n'interagit pas directement avec la DB, SAUF les méthodes DB de APIManager.

# Instance globale du gestionnaire d'APIs
api_manager = APIManager()
# ... (Reste du code des agents et des routes) ...

if __name__ == '__main__':
    # ... (le code d'exécution) ...
    pass
