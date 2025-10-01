#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WaveAI - Système d'Agents IA avec Tests Complets et Corrections HF
Version: FIXED FINAL - Tous les problèmes résolus
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
# Utilisez une variable d'environnement SECRET_KEY en production
app.secret_key = os.environ.get('SECRET_KEY', 'waveai-secret-key-2024')

# Configuration de la base de données
DATABASE_PATH = 'waveai.db'

class APIManager:
    """Gestionnaire centralisé des APIs avec persistance et tests"""
    
    def __init__(self):
        self.test_results = {}
        self.openai_api_key = None
        self.anthropic_api_key = None
        self.hf_working_model = None
        
        # Modèles Hugging Face de fallback (du plus accessible au moins accessible)
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
                'description': 'DialoGPT Small - Conversationnel léger'
            },
            {
                'name': 'gpt2',
                'url': 'https://api-inference.huggingface.co/models/gpt2',
                'description': 'GPT-2 - Génération de texte classique'
            },
            {
                'name': 'google/flan-t5-base',
                'url': 'https://api-inference.huggingface.co/models/google/flan-t5-base',
                'description': 'FLAN-T5 Base - Question-Réponse/Instruction'
            },
        ]

    def init_database(self):
        """Initialise la table des clés API si elle n'existe pas."""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    api_key_openai TEXT,
                    api_key_anthropic TEXT,
                    api_key_hf TEXT,
                    hf_working_model TEXT,
                    test_status TEXT,
                    test_details TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Erreur d'initialisation de la base de données: {e}")

    # Fonction CRITIQUE : CHARGEMENT DE L'ÉTAT
    def load_initial_state(self):
        """Charge l'état de la clé et du modèle fonctionnel le plus récent de la DB."""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            # Récupère la dernière configuration sauvegardée
            cursor.execute("SELECT api_key_openai, api_key_anthropic, hf_working_model FROM api_keys ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
            
            if result:
                # Met à jour les variables du gestionnaire
                self.openai_api_key = result[0] if result[0] else None
                self.anthropic_api_key = result[1] if result[1] else None
                self.hf_working_model = result[2] if result[2] else None
                logger.info(f"État initial chargé: OpenAI={bool(self.openai_api_key)}, Anthropic={bool(self.anthropic_api_key)}, HF Model={self.hf_working_model}")
            
            conn.close()
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état initial: {e}")

    def save_api_key(self, openai_key, anthropic_key, hf_key, hf_working_model=None, status="pending", details=""):
        """Sauvegarde les clés et le modèle fonctionnel dans la base de données."""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            # Sauvegarde de la nouvelle configuration
            cursor.execute("""
                INSERT INTO api_keys (timestamp, api_key_openai, api_key_anthropic, api_key_hf, hf_working_model, test_status, test_details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, openai_key, anthropic_key, hf_key, hf_working_model, status, details))
            
            conn.commit()
            conn.close()

            # Mettre à jour l'état du gestionnaire immédiatement après la sauvegarde
            self.load_initial_state() 
            
            return True
        except Exception as e:
            logger.error(f"Erreur de sauvegarde des clés: {e}")
            return False

    def get_api_status(self):
        """Récupère le statut de la dernière configuration sauvegardée."""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            # On récupère toutes les clés et le modèle fonctionnel sauvegardés
            cursor.execute("SELECT api_key_openai, api_key_anthropic, api_key_hf, hf_working_model, test_status, test_details FROM api_keys ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    # Les clés sont lues directement dans la base de données pour avoir le token HF pour l'appel
                    'api_key_openai': result[0],
                    'api_key_anthropic': result[1],
                    'api_key_hf': result[2], 
                    
                    'openai': {'configured': bool(result[0]), 'working': result[4] == 'success' and bool(result[0])},
                    'anthropic': {'configured': bool(result[1]), 'working': result[4] == 'success' and bool(result[1])},
                    'huggingface': {'configured': bool(result[2]), 'working_model': result[3], 'working': bool(result[3])},
                    'last_status': result[4],
                    'last_details': result[5]
                }
            return {'openai': {'configured': False, 'working': False}, 'anthropic': {'configured': False, 'working': False}, 'huggingface': {'configured': False, 'working_model': None, 'working': False}, 'last_status': 'none', 'last_details': 'No configuration found'}
        except Exception as e:
            logger.error(f"Erreur de lecture du statut: {e}")
            return {'error': str(e)}

    def test_openai_api(self, key):
        """Test la clé OpenAI."""
        if not key:
            return True, "No key provided. Skipping test."
        try:
            client = openai.OpenAI(api_key=key)
            client.models.list()
            return True, "Success: Models list accessible."
        except openai.AuthenticationError:
            return False, "Authentication Error: The key is invalid."
        except openai.APITimeoutError:
            return False, "Timeout Error: API is too slow or unreachable."
        except Exception as e:
            return False, f"Unknown OpenAI Error: {e}"

    def test_anthropic_api(self, key):
        """Test la clé Anthropic."""
        if not key:
            return True, "No key provided. Skipping test."
        try:
            client = anthropic.Anthropic(api_key=key)
            client.models.list() 
            return True, "Success: Models list accessible."
        except anthropic.AuthenticationError:
            return False, "Authentication Error: The key is invalid."
        except anthropic.APIStatusError:
            return False, "API Status Error: Service unavailable or key restricted."
        except Exception as e:
            return False, f"Unknown Anthropic Error: {e}"

    def test_hf_api_fallback(self, token):
        """Teste séquentiellement les modèles HF de fallback."""
        if not token:
            return True, None, "No token provided. Skipping test."
        
        for model in self.hf_models:
            url = model['url']
            headers = {"Authorization": f"Bearer {token}"}
            payload = {"inputs": "Salut, qui es-tu?"}
            
            logger.info(f"Tentative avec modèle Hugging Face: {model['name']}")
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10) 
                
                if response.status_code == 200:
                    logger.info(f"✅ Succès! Modèle trouvé: {model['name']}")
                    return True, model['name'], f"Success: Working model is {model['name']}"
                
                elif response.status_code == 403:
                    logger.warning(f"❌ Erreur 403 (Permission) pour {model['name']}. Essai du prochain modèle.")
                
                elif response.status_code == 503:
                    logger.info(f"⏳ Modèle {model['name']} en cours de chargement (503). Essai du prochain modèle.")

                else:
                    logger.warning(f"❌ Erreur {response.status_code} pour {model['name']}. Essai du prochain modèle. Détails: {response.text[:100]}...")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"❌ Timeout pour {model['name']}. Essai du prochain modèle.")
            except Exception as e:
                logger.error(f"❌ Erreur de connexion pour {model['name']}: {e}")

        logger.error("❌ Aucun modèle Hugging Face fonctionnel trouvé.")
        return False, None, "Failed: None of the fallback models are working with the provided token."

api_manager = APIManager()


class Agent:
    """Classe de base pour un agent IA."""
    
    def __init__(self, name, persona, model_name, system_prompt):
        self.name = name
        self.persona = persona
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.demo_response = f"Je suis l'agent {name} et je fonctionne en mode démo. Mon rôle est : {persona}. Veuillez configurer une API pour une réponse complète."
        self.demo_provider = "Demo Mode"

    def _call_openai(self, prompt):
        """Appel à l'API OpenAI."""
        # Utilise la clé chargée au démarrage ou sauvegardée
        if not api_manager.openai_api_key:
            return None, None
        try:
            client = openai.OpenAI(api_key=api_manager.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content, "OpenAI"
        except (openai.AuthenticationError, openai.APIError, requests.exceptions.RequestException) as e:
            logger.error(f"Erreur OpenAI pour l'agent {self.name}: {e}")
            return None, None 

    def _call_anthropic(self, prompt):
        """Appel à l'API Anthropic (Claude)."""
        # Utilise la clé chargée au démarrage ou sauvegardée
        if not api_manager.anthropic_api_key:
            return None, None
        try:
            client = anthropic.Anthropic(api_key=api_manager.anthropic_api_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307", 
                max_tokens=500,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text, "Anthropic"
        except (anthropic.AuthenticationError, anthropic.APIError, requests.exceptions.RequestException) as e:
            logger.error(f"Erreur Anthropic pour l'agent {self.name}: {e}")
            return None, None 

    def _call_huggingface(self, prompt):
        """Appel à l'API Hugging Face (modèle fonctionnel sauvegardé)."""
        model_name = api_manager.hf_working_model
        
        # Récupère le token HF le plus récent de la base de données
        status = api_manager.get_api_status()
        hf_token = status.get('api_key_hf')

        if not model_name or not hf_token:
            return None, None

        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": prompt}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status() 

            data = response.json()
            
            # Logique d'extraction de la réponse
            if isinstance(data, list) and data and 'generated_text' in data[0]:
                text = data[0]['generated_text']
            elif isinstance(data, dict) and 'generated_text' in data:
                 text = data['generated_text']
            else:
                # Fallback pour d'autres formats de réponse de l'API HF
                text = str(data)

            # Nettoyage pour les modèles de chat
            if model_name.startswith("microsoft/DialoGPT"):
                text = text.replace(prompt, "", 1).strip()
                
            return text, f"Hugging Face ({model_name})"

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur Hugging Face pour {model_name}: {e}")
            return None, None 
        except Exception as e:
            logger.error(f"Erreur de parsing HF pour {model_name}: {e}")
            return None, None

    def generate_response(self, prompt):
        """Génère la réponse en utilisant le meilleur fournisseur disponible (Fallback)."""
        
        # 1. Tente OpenAI
        response, provider = self._call_openai(prompt)
        if response:
            return {'response': response, 'provider': provider, 'success': True}

        # 2. Tente Anthropic
        response, provider = self._call_anthropic(prompt)
        if response:
            return {'response': response, 'provider': provider, 'success': True}

        # 3. Tente Hugging Face (avec le modèle fonctionnel sauvegardé)
        response, provider = self._call_huggingface(prompt)
        if response:
            return {'response': response, 'provider': provider, 'success': True}

        # 4. Mode Démo (Fallback ultime)
        return {'response': self.demo_response, 'provider': self.demo_provider, 'success': False}


# Initialisation des Agents
agents = {
    'kai': Agent(
        name="Kai",
        persona="Un assistant expert en codage Python et déploiement cloud (GitHub, Render).",
        model_name="GPT-3.5-Turbo",
        system_prompt="Tu es Kai, un assistant expert en Python, Flask, et déploiement cloud. Ton rôle est de fournir des solutions de code concises et prêtes à l'emploi. Réponds toujours en français."
    ),
    'elara': Agent(
        name="Elara",
        persona="Une spécialiste de la finance et de l'analyse de marché.",
        model_name="Claude-3-Haiku",
        system_prompt="Tu es Elara, une spécialiste de la finance et de l'analyse de marché. Fournis des informations claires et objectives sur l'économie, les actions et les cryptomonnaies. Réponds toujours en français."
    )
}


# --- Routes Flask ---

@app.route('/')
def index():
    """Page d'accueil et d'interaction avec les agents."""
    status = api_manager.get_api_status()
    # Statut pour affichage dans l'interface
    is_openai_working = status.get('openai', {}).get('working', False)
    is_anthropic_working = status.get('anthropic', {}).get('working', False)
    is_hf_working = status.get('huggingface', {}).get('working', False)
    
    # Détermine le fournisseur principal
    if is_openai_working:
        primary_provider = "OpenAI (GPT-3.5)"
    elif is_anthropic_working:
        primary_provider = "Anthropic (Claude-3)"
    elif is_hf_working:
        primary_provider = f"Hugging Face ({status.get('huggingface', {}).get('working_model')})"
    else:
        primary_provider = "Mode Démo Intelligent (API non configurée)"
        
    return render_template('index.html', 
        agents=agents, 
        primary_provider=primary_provider, 
        status=status)


@app.route('/settings')
def settings():
    """Page de configuration des clés API."""
    status = api_manager.get_api_status()
    return render_template('settings.html', status=status)

@app.route('/api/save_key', methods=['POST'])
def save_key():
    """Endpoint pour sauvegarder les clés API et déclencher les tests."""
    data = request.get_json()
    openai_key = data.get('openai_key')
    anthropic_key = data.get('anthropic_key')
    hf_key = data.get('hf_key')

    # Test des clés
    openai_success, openai_details = api_manager.test_openai_api(openai_key)
    anthropic_success, anthropic_details = api_manager.test_anthropic_api(anthropic_key)
    hf_success, hf_working_model, hf_details = api_manager.test_hf_api_fallback(hf_key)
    
    # Déterminer le statut global
    global_status = "success" if openai_success or anthropic_success or hf_success else "failure"
    
    details_list = []
    if openai_key and not openai_success: details_list.append(f"OpenAI: {openai_details}")
    if anthropic_key and not anthropic_success: details_list.append(f"Anthropic: {anthropic_details}")
    if hf_key and not hf_success: details_list.append(f"Hugging Face: {hf_details}")
    
    global_details = "; ".join(details_list) if details_list else "All provided keys passed their respective tests or were not provided."

    # Sauvegarde de la configuration
    api_manager.save_api_key(
        openai_key, 
        anthropic_key, 
        hf_key, 
        hf_working_model, 
        global_status, 
        global_details
    )
    
    return jsonify({
        'success': True,
        'message': f"Clés sauvegardées. Test global: {global_status.upper()}",
        'openai_status': 'Working' if openai_success else 'Failed',
        'anthropic_status': 'Working' if anthropic_success else 'Failed',
        'hf_model': hf_working_model if hf_working_model else 'Failed'
    })

@app.route('/api/get_api_status', methods=['GET'])
def get_api_status_endpoint():
    """Endpoint pour obtenir le statut actuel des APIs."""
    return jsonify(api_manager.get_api_status())

@app.route('/api/generate', methods=['POST'])
def generate():
    """Endpoint pour générer une réponse d'agent."""
    data = request.get_json()
    prompt = data.get('prompt')
    agent_name = data.get('agent', 'kai').lower()
    
    if not prompt or agent_name not in agents:
        return jsonify({'response': 'Erreur : prompt ou agent invalide.', 'provider': 'System Error', 'success': False})

    agent = agents[agent_name]
    response_data = agent.generate_response(prompt)
    
    return jsonify(response_data)


@app.route('/api/test_agent', methods=['POST'])
def test_agent():
    """Test spécifique d'un agent"""
    try:
        data = request.get_json()
        agent_name = data.get('agent', 'kai').lower()
        
        if agent_name not in agents:
            return jsonify({'success': False, 'message': f'Agent {agent_name} non trouvé'})
        
        # Test avec un message standard
        test_message = "Bonjour, peux-tu te présenter et expliquer ton rôle ?"
        agent = agents[agent_name]
        response_data = agent.generate_response(test_message)
        
        return jsonify({
            'success': True,
            'agent': agent_name,
            'test_message': test_message,
            'response': response_data['response'],
            'provider': response_data['provider'],
            'api_working': response_data['success']
        })
        
    except Exception as e:
        logger.error(f"Erreur test agent: {e}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    try:
        logger.info("Démarrage de WaveAI...")
        
        # 1. Initialiser la base de données
        api_manager.init_database()
        
        # 2. CHARGEMENT CRITIQUE : Charger l'état des clés et modèles au démarrage
        api_manager.load_initial_state() 
        
        logger.info("Système initialisé avec succès")
        
        # Démarrer l'application
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True) 

    except Exception as e:
        logger.error(f"Erreur fatale au démarrage: {e}")
