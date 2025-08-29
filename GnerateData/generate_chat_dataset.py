import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import json
from tqdm import tqdm
from groq import Groq
import random

"""
Script pour générer un dataset de FAQ sur les cartes de fidélité
en utilisant deux modèles:
- Un modèle pour générer des questions réalistes basées sur les FAQ existantes
- Un modèle pour générer des réponses officielles basées sur les sites d'entreprises
"""

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv("/home/anas-nouri/chatBotAPP/config/.env")

# Modèles disponibles
MODEL_LLAMA = "llama3-70b-8192"
MODEL_LLAMA_8B = "llama3-8b-8192"
MODEL_OPENAI = "openai/gpt-oss-120b"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------- PARAMÈTRES -----------------

# OPTIONS DE GÉNÉRATION
GENERATE_CARD_ACQUISITION = True
GENERATE_CARD_COST = True
GENERATE_POINTS_ACCUMULATION = True
GENERATE_POINTS_BALANCE = True
GENERATE_CARD_LOSS = True
GENERATE_BENEFITS_ADVANTAGES = True
GENERATE_CARD_USAGE = True
GENERATE_ACCOUNT_MANAGEMENT = True
GENERATE_ELIGIBILITY = True
GENERATE_EXPIRATION = True

# MODÈLES UTILISÉS
QUESTION_MODEL = MODEL_LLAMA  # Modèle pour générer les questions
ANSWER_MODEL = MODEL_OPENAI  # Modèle pour générer les réponses

# PARAMÈTRES DU MODÈLE
temperature_questions = 1.2  # Plus de créativité pour les questions
temperature_answers = 0.8    # Plus de cohérence pour les réponses

# NOMBRE DE QUESTIONS PAR CATÉGORIE
QUESTIONS_PER_CATEGORY = 15

# ----------------- CONTEXTES ET PROMPTS -----------------

# Contexte sur les cartes de fidélité au Maroc
LOYALTY_CARD_CONTEXT = """
Contexte sur les cartes de fidélité au Maroc:
- Entreprises populaires: Carrefour Maroc, Marjane, Atacadao, BIM, Aswak Assalam
- Monnaie: Dirham marocain (DH)
- Points généralement: 1 point pour 20 DH d'achat
- Cartes généralement gratuites
- Avantages: réductions, offres spéciales, points cadeaux
- Problèmes courants: perte de carte, oubli de carte, points qui expirent
"""

# Contextes spécifiques par catégorie
CATEGORY_CONTEXTS = {
    "card_acquisition": {
        "context": "Questions sur comment obtenir une carte de fidélité - inscription, documents nécessaires, lieux d'inscription, délais de réception",
        "examples": [
            "Comment puis-je obtenir une carte de fidélité ?",
            "Où puis-je m'inscrire pour avoir ma carte ?",
            "Quels documents dois-je apporter ?",
            "Est-ce que je peux faire ma demande en ligne ?"
        ]
    },
    "card_cost": {
        "context": "Questions sur les coûts et frais de la carte de fidélité - gratuite ou payante, frais cachés, frais de renouvellement",
        "examples": [
            "Est-ce que la carte est gratuite ?",
            "Y a-t-il des frais d'inscription ?",
            "Combien coûte le renouvellement de la carte ?",
            "Y a-t-il des frais cachés ?"
        ]
    },
    "points_accumulation": {
        "context": "Questions sur comment gagner et accumuler des points - taux de conversion, produits éligibles, promotions spéciales",
        "examples": [
            "Comment gagner des points avec ma carte ?",
            "Combien de points j'obtiens pour 100 DH d'achat ?",
            "Est-ce que tous les produits donnent des points ?",
            "Comment profiter des points bonus ?"
        ]
    },
    "points_balance": {
        "context": "Questions sur la consultation du solde de points - où vérifier, historique des points, relevé de compte",
        "examples": [
            "Comment consulter mon solde de points ?",
            "Où puis-je voir l'historique de mes points ?",
            "Comment recevoir un relevé de mes points ?",
            "Y a-t-il une application mobile ?"
        ]
    },
    "card_loss": {
        "context": "Questions sur la perte ou vol de carte - procédure de remplacement, récupération des points, coût de remplacement",
        "examples": [
            "J'ai perdu ma carte, que dois-je faire ?",
            "Comment remplacer ma carte volée ?",
            "Est-ce que je garde mes points avec une nouvelle carte ?",
            "Combien coûte le remplacement de carte ?"
        ]
    },
    "benefits_advantages": {
        "context": "Questions sur les avantages et bénéfices de la carte - réductions, offres spéciales, cadeaux, privilèges",
        "examples": [
            "Quels sont les avantages de la carte ?",
            "Comment utiliser mes points pour des réductions ?",
            "Y a-t-il des offres spéciales pour les porteurs de carte ?",
            "Comment échanger mes points contre des cadeaux ?"
        ]
    }
}

# Prompts pour générer les questions
QUESTION_GENERATION_PROMPT = """
Tu es un expert en expérience client au Maroc. Génère une question réaliste qu'un client marocain poserait fréquemment sur les cartes de fidélité.

{context}

La question doit être:
- Naturelle et dans le style d'un vrai client marocain
- Différente des exemples donnés
- Courte et directe
- En français (language utilisé au Maroc pour ce type de service)

Exemples de questions similaires: {examples}

{loyalty_context}

GÉNÈRE SEULEMENT LA QUESTION, RIEN D'AUTRE:
"""

# Prompts pour générer les réponses
ANSWER_GENERATION_PROMPT = """
Tu es un agent du service client officiel d'une grande chaîne de distribution au Maroc (type Carrefour, Marjane).

Réponds de manière professionnelle et officielle à cette question de client sur la carte de fidélité:

QUESTION: {question}

Ta réponse doit être:
- Professionnelle et courtoise
- Précise et informative
- Adaptée au contexte marocain (DH, magasins au Maroc)
- Basée sur les pratiques standard des entreprises marocaines
- En français
- Complète mais concise

{loyalty_context}

GÉNÈRE SEULEMENT LA RÉPONSE OFFICIELLE, RIEN D'AUTRE:
"""

# ----------------- FONCTIONS UTILITAIRES -----------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=100),
)
def ask_groq(prompt, model, temperature=1.0):
    """Fonction pour interroger Groq API avec gestion des erreurs"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à Groq: {e}")
        raise

def generate_question_for_category(category, category_info):
    """Génère une question pour une catégorie spécifique"""
    examples_str = "\n- ".join(category_info["examples"])
    
    prompt = QUESTION_GENERATION_PROMPT.format(
        context=category_info["context"],
        examples=examples_str,
        loyalty_context=LOYALTY_CARD_CONTEXT
    )
    
    return ask_groq(prompt, QUESTION_MODEL, temperature_questions)

def generate_answer_for_question(question):
    """Génère une réponse officielle pour une question"""
    prompt = ANSWER_GENERATION_PROMPT.format(
        question=question,
        loyalty_context=LOYALTY_CARD_CONTEXT
    )
    
    return ask_groq(prompt, ANSWER_MODEL, temperature_answers)

def generate_qa_pairs_for_category(category, category_info, count=10):
    """Génère des paires question-réponse pour une catégorie"""
    conversations = []
    
    logger.info(f"Génération de {count} paires Q&A pour la catégorie: {category}")
    
    for i in tqdm(range(count), desc=f"Génération {category}"):
        try:
            # Générer la question
            question = generate_question_for_category(category, category_info)
            
            # Générer la réponse
            answer = generate_answer_for_question(question)
            
            # Créer la conversation
            conversation = {
                "intent": category,
                "question": question,
                "answer": answer,
                "metadata": {
                    "category": category,
                    "generated_by": {
                        "question_model": QUESTION_MODEL,
                        "answer_model": ANSWER_MODEL
                    }
                }
            }
            
            conversations.append(conversation)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération pour {category}: {e}")
            continue
    
    return conversations

def save_conversations_to_jsonl(conversations, output_dir, file_name):
    """Sauvegarde les conversations au format JSONL"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, "w", encoding="utf-8") as file:
        for conversation in conversations:
            json_string = json.dumps(conversation, ensure_ascii=False)
            file.write(f"{json_string}\n")
    
    logger.info(f"Sauvegardé {len(conversations)} conversations dans {file_path}")

def generate_complete_dataset():
    """Génère le dataset complet pour toutes les catégories"""
    all_conversations = []
    output_dir = "loyalty_card_dataset"
    
    # Génération pour chaque catégorie activée
    categories_to_generate = {
        "card_acquisition": GENERATE_CARD_ACQUISITION,
        "card_cost": GENERATE_CARD_COST,
        "points_accumulation": GENERATE_POINTS_ACCUMULATION,
        "points_balance": GENERATE_POINTS_BALANCE,
        "card_loss": GENERATE_CARD_LOSS,
        "benefits_advantages": GENERATE_BENEFITS_ADVANTAGES,
    }
    
    for category, should_generate in categories_to_generate.items():
        if should_generate and category in CATEGORY_CONTEXTS:
            category_conversations = generate_qa_pairs_for_category(
                category, 
                CATEGORY_CONTEXTS[category], 
                QUESTIONS_PER_CATEGORY
            )
            all_conversations.extend(category_conversations)
            
            # Sauvegarder chaque catégorie séparément
            save_conversations_to_jsonl(
                category_conversations, 
                output_dir, 
                f"loyalty_card_{category}.jsonl"
            )
    
    # Sauvegarder le dataset complet
    if all_conversations:
        save_conversations_to_jsonl(
            all_conversations, 
            output_dir, 
            "loyalty_card_complete_dataset.jsonl"
        )
        
        # Générer aussi le format d'entraînement standard
        training_format = []
        for conv in all_conversations:
            training_format.append({
                "conversations": [
                    {"role": "user", "content": conv["question"]},
                    {"role": "assistant", "content": conv["answer"]}
                ]
            })
        
        save_conversations_to_jsonl(
            training_format,
            output_dir,
            "loyalty_card_training_format.jsonl"
        )
    
    logger.info(f"Dataset complet généré avec {len(all_conversations)} conversations")
    return all_conversations

def main():
    """Fonction principale"""
    logger.info("Début de la génération du dataset FAQ carte de fidélité")
    logger.info(f"Modèle pour questions: {QUESTION_MODEL}")
    logger.info(f"Modèle pour réponses: {ANSWER_MODEL}")
    
    try:
        conversations = generate_complete_dataset()
        logger.info("Génération terminée avec succès!")
        return conversations
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        raise

if __name__ == "__main__":
    main()