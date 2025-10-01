#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WaveAI - Système d'Agents IA avec Tests Complets et Corrections HF
Version: FIXED FINAL - Problème de persistance DB résolu
"""

import os
import sqlite3
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import openai
import anthropic
import requests
from functools import wraps

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# S'assurer que SECRET_KEY est bien défini pour les sessions Flask
app.secret_key = os.environ.get('SECRET_KEY', 'waveai-secret-key-2024')

# Configuration de la base de données
DATABASE_PATH = 'waveai.db'

class APIManager:
    """Gestionnaire centralisé des APIs avec persistance et tests"""
    
    def __init__(self):
        self.init_database()
        self.test_results = {}
        # Modèles Hugging Face de fallback
        self.hf_models = [
            {
                'name': 'microsoft/DialoGPT-medium',
                'url': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
                'description': 'DialoGPT Medium - Conversationnel'
            },
            {
                'name': 'facebook/blenderbot-400M-distill',
                'url': 'https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill',
                'description': 'BlenderBot - Conversationnel distillé'
            },
            {
                'name': 'microsoft/DialoGPT-small',
                'url': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-small',
                'description': 'DialoGPT Small - Version légère'
            },
            {
                'name': 'gpt2',
                'url': 'https://api-inference.huggingface.co/models/gpt2',
                'description': 'GPT-2 - Génération de texte classique'
            },
            {
                'name': 'distilgpt2',
                'url': 'https://api-inference.huggingface.co/models/distilgpt2',
                'description': 'DistilGPT-2 - Version légère de GPT-2'
            }
        ]

    def init_database(self):
        """Initialise la table api_keys si elle n'existe pas"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                provider TEXT PRIMARY KEY,
                api_key TEXT,
                test_status TEXT DEFAULT 'untested',
                last_tested TEXT,
                working_model TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get_api_key(self, provider):
        """
        Récupère une clé API par fournisseur.
        PRIORITÉ CORRIGÉE : 1. Variable d'Environnement > 2. Base de Données
        """
        # Mapping des variables d'environnement aux noms des fournisseurs
        env_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'huggingface': 'HUGGINGFACE_TOKEN'
        }

        # 1. Tenter de lire depuis la variable d'environnement
        env_var_name = env_map.get(provider)
        if env_var_name:
            env_key = os.getenv(env_var_name)
            if env_key:
                # La clé a été trouvée dans les variables d'environnement, elle est prioritaire
                return env_key

        # 2. Si non trouvée, lire depuis la base de données
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT api_key FROM api_keys WHERE provider = ?", (provider,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        
        return None # Retourne None si la clé n'est nulle part

    def save_api_key(self, provider, api_key, test_status='untested', working_model=None):
        """Sauvegarde/Met à jour une clé API et son statut de test"""
        # Note: Si la clé vient de l'ENV, elle n'est pas sauvegardée ici, mais celle-ci
        # permet de sauvegarder les clés entrées manuellement dans l'interface.
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO api_keys (provider, api_key, test_status, last_tested, working_model)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                api_key = excluded.api_key,
                test_status = excluded.test_status,
                last_tested = excluded.last_tested,
                working_model = excluded.working_model
        """, (provider, api_key, test_status, timestamp, working_model))
        
        conn.commit()
        conn.close()

    def test_huggingface_api(self, token):
        """Teste séquentiellement les modèles HF de fallback"""
        logger.info("Démarrage du test Hugging Face séquentiel...")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": "Bonjour, dis un seul mot.",
            "options": {"wait_for_model": True}
        }
        
        working_model = None
        
        for model in self.hf_models:
            url = model['url']
            name = model['name']
            
            logger.info(f" -> Test de: {name}")
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    response.json()
                    working_model = name
                    logger.info(f"✅ Succès pour {name}!")
                    break
                
                elif response.status_code == 503:
                    logger.warning(f"503 pour {name}. Modèle en cours de chargement. Continue...")
                    
                elif response.status_code == 403:
                    logger.error(f"403 pour {name}. Problème de token. Abandon.")
                    break
                    
                else:
                    logger.error(f"Erreur HTTP {response.status_code} pour {name}. Réponse: {response.text}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur de connexion pour {name}: {e}")
            except json.JSONDecodeError:
                logger.error(f"Erreur de décodage JSON pour {name}. Réponse non valide.")
            
        
        # Mise à jour du statut dans la base de données
        status = 'success' if working_model else 'failed'
        self.save_api_key('huggingface', token, status, working_model=working_model)
        
        return status, working_model
    
    def test_openai_api(self, api_key):
        """Teste l'API OpenAI"""
        logger.info("Démarrage du test OpenAI...")
        status = 'failed'
        
        try:
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Dit bonjour en français."}],
                max_tokens=10
            )
            
            if response.choices[0].message.content:
                status = 'success'
                logger.info("✅ Succès du test OpenAI!")
            
        except openai.APIError as e:
            # Capture les erreurs spécifiques (AuthenticationError, RateLimitError, Quota)
            logger.error(f"❌ Erreur API OpenAI (Clé/Quota/Auth): {e}")
        except Exception as e:
            logger.error(f"❌ Erreur générale OpenAI: {e}")
            
        self.save_api_key('openai', api_key, status)
        return status

    def test_anthropic_api(self, api_key):
        """Teste l'API Anthropic"""
        logger.info("Démarrage du test Anthropic...")
        status = 'failed'

        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Dis un seul mot."}]
            )

            if response.content[0].text:
                status = 'success'
                logger.info("✅ Succès du test Anthropic!")

        except anthropic.APIError as e:
            logger.error(f"❌ Erreur API Anthropic (Clé/Quota/Auth): {e}")
        except Exception as e:
            logger.error(f"❌ Erreur générale Anthropic: {e}")

        self.save_api_key('anthropic', api_key, status)
        return status

api_manager = APIManager()


class AIAgent:
    """Agent IA générique qui utilise le meilleur fournisseur disponible."""
    def __init__(self, name, description, system_prompt):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.api_manager = api_manager

    def generate_response(self, user_prompt):
        """Génère une réponse en utilisant la meilleure API disponible (priorité)"""
        
        # 1. Tenter avec OpenAI (Priorité 1)
        openai_key = self.api_manager.get_api_key('openai')
        if openai_key:
            try:
                client = openai.OpenAI(api_key=openai_key)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                return {'success': True, 'response': response.choices[0].message.content, 'provider': 'OpenAI', 'model': 'gpt-3.5-turbo'}
                
            except openai.APIError as e:
                logger.error(f"Échec OpenAI API (Quota/Auth): {e}")
            except Exception as e:
                logger.error(f"Échec OpenAI Général: {e}")

        # 2. Tenter avec Anthropic (Priorité 2)
        anthropic_key = self.api_manager.get_api_key('anthropic')
        if anthropic_key:
            try:
                client = anthropic.Anthropic(api_key=anthropic_key)

                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": f"{self.system_prompt}\n\n{user_prompt}"}
                    ]
                )
                
                return {'success': True, 'response': response.content[0].text, 'provider': 'Anthropic', 'model': 'Claude 3 Haiku'}
            
            except anthropic.APIError as e:
                logger.error(f"Échec Anthropic API (Quota/Auth): {e}")
            except Exception as e:
                logger.error(f"Échec Anthropic Général: {e}")


        # 3. Tenter avec Hugging Face (Priorité 3)
        hf_key = self.api_manager.get_api_key('huggingface')
        if hf_key:
            # Récupérer le modèle fonctionnel sauvegardé
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT working_model FROM api_keys WHERE provider = 'huggingface'")
            result = cursor.fetchone()
            working_model = result[0] if result else None
            conn.close()

            if working_model:
                # Trouver l'URL du modèle de travail
                model_info = next((m for m in self.api_manager
