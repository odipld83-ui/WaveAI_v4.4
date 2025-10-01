#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test spécifique pour corriger le problème Hugging Face
"""

import requests
import sys

def test_hf_models(token):
    """Test tous les modèles HF disponibles"""
    
    models = [
        {
            'name': 'microsoft/DialoGPT-small',
            'url': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-small',
            'description': 'DialoGPT Small - Conversationnel léger'
        },
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
            'name': 'gpt2',
            'url': 'https://api-inference.huggingface.co/models/gpt2',
            'description': 'GPT-2 - Génération de texte classique'
        },
        {
            'name': 'distilgpt2',
            'url': 'https://api-inference.huggingface.co/models/distilgpt2',
            'description': 'DistilGPT-2 - Version légère de GPT-2'
        },
        {
            'name': 'google/flan-t5-small',
            'url': 'https://api-inference.huggingface.co/models/google/flan-t5-small',
            'description': 'FLAN-T5 Small - Question-réponse'
        }
    ]
    
    headers = {"Authorization": f"Bearer {token}"}
    working_models = []
    
    print("🧪 Test des modèles Hugging Face disponibles...")
    print("=" * 60)
    
    for model in models:
        print(f"\n📊 Test: {model['name']}")
        print(f"   📝 {model['description']}")
        
        try:
            # Adapter le payload selon le modèle
            if 'flan-t5' in model['name'].lower():
                payload = {
                    "inputs": "Question: Comment allez-vous? Réponse:",
                    "parameters": {"max_new_tokens": 30}
                }
            elif 'blenderbot' in model['name'].lower():
                payload = {
                    "inputs": "Hello",
                    "parameters": {"max_new_tokens": 30}
                }
            else:  # GPT-2, DialoGPT
                payload = {
                    "inputs": "Hello, how are you?",
                    "parameters": {"max_new_tokens": 30, "temperature": 0.7}
                }
            
            response = requests.post(model['url'], headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                
                generated_text = ""
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    generated_text = result.get('generated_text', '')
                
                if generated_text and len(generated_text.strip()) > 5:
                    print(f"   ✅ FONCTIONNEL")
                    print(f"   📤 Réponse: {generated_text[:80]}...")
                    working_models.append(model)
                else:
                    print(f"   ❌ Réponse vide ou trop courte")
                    
            elif response.status_code == 503:
                print(f"   ⏳ Modèle en cours de chargement")
            elif response.status_code == 403:
                print(f"   🔒 Permissions insuffisantes")
            else:
                print(f"   ❌ Erreur HTTP {response.status_code}")
                print(f"      {response.text[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📈 RÉSULTATS: {len(working_models)}/{len(models)} modèles fonctionnels")
    
    if working_models:
        print("\n✅ Modèles disponibles avec votre token:")
        for model in working_models:
            print(f"   • {model['name']} - {model['description']}")
        return working_models[0]['name']  # Retourner le premier qui fonctionne
    else:
        print("\n❌ Aucun modèle accessible avec ce token")
        print("\n💡 Solutions possibles:")
        print("   1. Vérifiez que votre token HF est correct")
        print("   2. Créez un nouveau token avec les permissions 'Inference API'")
        print("   3. Attendez quelques minutes (modèles en cours de chargement)")
        print("   4. Utilisez OpenAI ou Anthropic à la place")
        return None

def main():
    print("🤗 TEST DE CORRECTION HUGGING FACE")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = input("\n🔑 Entrez votre token Hugging Face: ").strip()
    
    if not token:
        print("❌ Token requis")
        return
    
    working_model = test_hf_models(token)
    
    if working_model:
        print(f"\n🎉 SOLUTION TROUVÉE!")
        print(f"   Modèle recommandé: {working_model}")
        print(f"   WaveAI utilisera automatiquement ce modèle.")
    else:
        print(f"\n⚠️  Aucun modèle HF accessible")
        print(f"   Recommandation: Utilisez OpenAI ou Anthropic")

if __name__ == "__main__":
    main()