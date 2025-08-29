import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import json
from tqdm import tqdm
from groq import Groq
import random
from difflib import SequenceMatcher
import hashlib

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

# Modèles disponibles (mis à jour pour 2025)
MODEL_LLAMA_70B = "openai/gpt-oss-20b"  # Production model
MODEL_LLAMA_8B = "llama-3.1-8b-instant"      # Production model  
MODEL_GEMMA = "gemma2-9b-it"                  # Production model
MODEL_QWEN = "qwen/qwen3-32b"                 # Preview model
MODEL_KIMI = "moonshotai/kimi-k2-instruct"    # Preview model
MODEL_OPENAI = "qwen/qwen3-32b"


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
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
QUESTION_MODEL = MODEL_LLAMA_70B   # Modèle pour générer les questions
ANSWER_MODEL = MODEL_OPENAI      # Modèle pour générer les réponses

# PARAMÈTRES DU MODÈLE
temperature_questions = 1.4  # Plus de créativité pour les questions (augmenté)
temperature_answers = 0.8    # Plus de cohérence pour les réponses

# CONTRÔLE DE LA DIVERSITÉ
MAX_SIMILARITY_THRESHOLD = 0.7  # Seuil de similarité (0-1)
MAX_RETRY_FOR_UNIQUE = 5        # Nombre max de tentatives pour question unique
ENABLE_SIMILARITY_CHECK = True   # Activer la vérification de similarité
ENABLE_VARIATION_PROMPTS = True  # Activer les prompts de variation

# NOMBRE DE QUESTIONS PAR CATÉGORIE
QUESTIONS_PER_CATEGORY = 15

# ----------------- CONTEXTES ET PROMPTS -----------------

# Contexte sur les cartes de fidélité au Maroc
LOYALTY_CARD_CONTEXT = """
Contexte sur les cartes de fidélité au Maroc:
- Entreprises populaires: Carrefour Maroc, Marjane, Atacadao, BIM, Aswak Assalam
- Monnaie: Dirham marocain (DH)
- Points généralement: 1 point pour 10 DH d'achat
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

# Prompts pour générer les questions avec variation
QUESTION_GENERATION_PROMPTS = [
    """
Tu es un expert en expérience client au Maroc. Génère une question ORIGINALE et UNIQUE qu'un client marocain poserait fréquemment sur les cartes de fidélité.

{context}

CONTRAINTES IMPORTANTES:
- La question doit être DIFFÉRENTE de toutes les questions existantes
- Utilise une formulation CRÉATIVE et NATURELLE
- Varie le style: question directe, informelle, préoccupée, curieuse
- Évite les formulations répétitives

Questions déjà générées à éviter: {existing_questions}

{loyalty_context}

GÉNÈRE SEULEMENT UNE QUESTION UNIQUE ET ORIGINALE:
""",
    """
Un client marocain cherche des informations sur sa carte de fidélité. Génère une question DIFFÉRENTE des autres, avec un angle nouveau.

{context}

VARIANTES À EXPLORER:
- Question avec contexte personnel ("J'ai un problème avec...")
- Question comparative ("Est-ce que c'est mieux que...")
- Question situationnelle ("Dans le cas où...")
- Question d'urgence ("J'ai besoin de savoir rapidement...")

Questions à éviter: {existing_questions}

{loyalty_context}

QUESTION ORIGINALE:
""",
    """
Imagine différents profils de clients marocains (jeune, âgé, famille, professionnel) qui posent des questions sur les cartes de fidélité.

{context}

PROFILS DE CLIENTS:
- Étudiant/jeune: questions sur économies, facilité
- Parent: questions sur famille, avantages enfants
- Professionnel: questions sur efficacité, temps
- Personne âgée: questions sur simplicité, aide

Évite ces formulations: {existing_questions}

{loyalty_context}

NOUVELLE QUESTION SELON UN PROFIL:
"""
]

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

def calculate_similarity(text1, text2):
    """Calcule la similarité entre deux textes (0-1)"""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()

def is_question_unique(new_question, existing_questions, threshold=0.7):
    """Vérifie si une question est suffisamment unique"""
    if not ENABLE_SIMILARITY_CHECK:
        return True
        
    for existing_q in existing_questions:
        similarity = calculate_similarity(new_question, existing_q)
        if similarity > threshold:
            return False
    return True

def generate_question_hash(question):
    """Génère un hash unique pour une question"""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def clean_question_text(question):
    """Nettoie le texte de la question"""
    question = question.strip()
    # Enlever les préfixes courants
    prefixes_to_remove = [
        "Question:", "QUESTION:", "Q:", 
        "Voici la question:", "La question est:",
        "Question originale:", "QUESTION ORIGINALE:"
    ]
    for prefix in prefixes_to_remove:
        if question.startswith(prefix):
            question = question[len(prefix):].strip()
    
    # S'assurer que ça finit par un point d'interrogation
    if not question.endswith('?'):
        question += ' ?'
    
    return question

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

def generate_question_for_category(category, category_info, existing_questions=None, attempt=0):
    """Génère une question unique pour une catégorie spécifique"""
    if existing_questions is None:
        existing_questions = []
    
    # Choisir un prompt de variation aléatoire
    if ENABLE_VARIATION_PROMPTS:
        prompt_template = random.choice(QUESTION_GENERATION_PROMPTS)
    else:
        prompt_template = QUESTION_GENERATION_PROMPTS[0]
    
    # Limiter le nombre de questions existantes montrées pour éviter des prompts trop longs
    existing_sample = random.sample(existing_questions, min(5, len(existing_questions)))
    existing_text = "\n- " + "\n- ".join(existing_sample) if existing_sample else "Aucune"
    
    prompt = prompt_template.format(
        context=category_info["context"],
        existing_questions=existing_text,
        loyalty_context=LOYALTY_CARD_CONTEXT
    )
    
    # Ajouter de la randomité avec des paramètres variables
    temp_variation = random.uniform(1.2, 1.6)  # Température variable
    
    question = ask_groq(prompt, QUESTION_MODEL, temp_variation)
    question = clean_question_text(question)
    
    return question

def generate_answer_for_question(question):
    """Génère une réponse officielle pour une question"""
    prompt = ANSWER_GENERATION_PROMPT.format(
        question=question,
        loyalty_context=LOYALTY_CARD_CONTEXT
    )
    
    return ask_groq(prompt, ANSWER_MODEL, temperature_answers)

def generate_qa_pairs_for_category(category, category_info, count=10):
    """Génère des paires question-réponse uniques pour une catégorie"""
    conversations = []
    existing_questions = []
    question_hashes = set()  # Pour vérification rapide des doublons
    
    logger.info(f"Génération de {count} paires Q&A UNIQUES pour la catégorie: {category}")
    
    for i in tqdm(range(count), desc=f"Génération {category}"):
        question = None
        attempts = 0
        
        # Tenter de générer une question unique
        while attempts < MAX_RETRY_FOR_UNIQUE:
            try:
                # Générer la question
                question = generate_question_for_category(
                    category, 
                    category_info, 
                    existing_questions, 
                    attempts
                )
                
                # Vérifier l'unicité
                question_hash = generate_question_hash(question)
                
                if (question_hash not in question_hashes and 
                    is_question_unique(question, existing_questions, MAX_SIMILARITY_THRESHOLD)):
                    break  # Question unique trouvée
                else:
                    logger.debug(f"Question similaire détectée, tentative {attempts + 1}")
                    attempts += 1
                    question = None
                    
            except Exception as e:
                logger.error(f"Erreur génération question (tentative {attempts + 1}): {e}")
                attempts += 1
        
        if question is None:
            logger.warning(f"Impossible de générer une question unique après {MAX_RETRY_FOR_UNIQUE} tentatives")
            continue
        
        try:
            # Générer la réponse
            answer = generate_answer_for_question(question)
            
            # Ajouter à la liste des questions existantes
            existing_questions.append(question)
            question_hashes.add(generate_question_hash(question))
            
            # Créer la conversation
            conversation = {
                "intent": category,
                "question": question,
                "answer": answer,
                "metadata": {
                    "category": category,
                    "question_hash": generate_question_hash(question),
                    "generation_attempt": attempts + 1,
                    "generated_by": {
                        "question_model": QUESTION_MODEL,
                        "answer_model": ANSWER_MODEL
                    }
                }
            }
            
            conversations.append(conversation)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse pour {category}: {e}")
            continue
    
    # Statistiques finales
    unique_count = len(conversations)
    logger.info(f"✅ {unique_count}/{count} questions uniques générées pour {category}")
    
    return conversations

def save_conversations_to_jsonl(conversations, output_dir, file_name):
    """Sauvegarde les conversations au format JSONL avec statistiques"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, file_name)
    
    # Vérifier l'unicité finale avant sauvegarde
    unique_questions = set()
    duplicates_found = 0
    
    filtered_conversations = []
    for conv in conversations:
        question_hash = generate_question_hash(conv.get("question", ""))
        if question_hash not in unique_questions:
            unique_questions.add(question_hash)
            filtered_conversations.append(conv)
        else:
            duplicates_found += 1
    
    if duplicates_found > 0:
        logger.warning(f"🚨 {duplicates_found} doublons supprimés lors de la sauvegarde")
    
    with open(file_path, "w", encoding="utf-8") as file:
        for conversation in filtered_conversations:
            json_string = json.dumps(conversation, ensure_ascii=False)
            file.write(f"{json_string}\n")
    
    logger.info(f"✅ Sauvegardé {len(filtered_conversations)} conversations UNIQUES dans {file_path}")
    return len(filtered_conversations)

def generate_complete_dataset():
    """Génère le dataset complet pour toutes les catégories"""
    all_conversations = []
    output_dir = "loyalty_card_datasets"
    
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
    logger.info("🚀 Début de la génération du dataset FAQ carte de fidélité")
    logger.info(f"Modèle pour questions: {QUESTION_MODEL}")
    logger.info(f"Modèle pour réponses: {ANSWER_MODEL}")
    logger.info("Modèles disponibles mis à jour pour 2025")
    logger.info(f"🎯 Contrôles d'unicité activés:")
    logger.info(f"   - Seuil de similarité: {MAX_SIMILARITY_THRESHOLD}")
    logger.info(f"   - Tentatives max par question: {MAX_RETRY_FOR_UNIQUE}")
    logger.info(f"   - Température questions: {temperature_questions}")
    
    try:
        conversations = generate_complete_dataset()
        logger.info("🎉 Génération terminée avec succès!")
        return conversations
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération: {e}")
        raise

if __name__ == "__main__":
    main()