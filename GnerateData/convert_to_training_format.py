import json
import os

def convert_to_training_format(input_dir="/home/anas-nouri/chatBotAPP/datasets/loyalty_card_datasets"):
    """Convertit tous les fichiers JSONL au format d'entraînement"""
    
    # Fichiers à traiter
    files = [
        "loyalty_card_benefits_advantages.jsonl",
        "loyalty_card_card_acquisition.jsonl", 
        "loyalty_card_card_cost.jsonl",
        "loyalty_card_card_loss.jsonl",
        "loyalty_card_points_accumulation.jsonl",
        "loyalty_card_points_balance.jsonl"
    ]
    
    all_conversations = []
    
    # Lire tous les fichiers
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    conv = json.loads(line.strip())
                    all_conversations.append(conv)
    
    # Convertir au format d'entraînement

    training_format = []
    for conv in all_conversations:
    # Extraire seulement les champs voulus des métadonnées
      filtered_metadata = {
        'category': conv["metadata"].get('category'),
        'question_hash': conv["metadata"].get('question_hash'),
        'generation_attempt': conv["metadata"].get('generation_attempt', 1)
    }
    
      training_format.append({
        "conversations": [
            {"role": "user", "content": conv["question"]},
            {"role": "assistant", "content": conv["answer"]}
        ],
        "metadata": filtered_metadata
      })
    # training_format = []
    # for conv in all_conversations:
    #     training_format.append({
    #         "conversations": [
    #             {"role": "user", "content": conv["question"]},
    #             {"role": "assistant", "content": conv["answer"]},],
    #         "metadata":conv["metadata"]
                
            
    #     })
        # if "metadata" in conv:
        #     training_format.append(conv["metadata"])
        
    # Sauvegarder
    output_file = os.path.join(input_dir, "loyalty_card_training_format_1.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_format:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ {len(training_format)} conversations converties dans {output_file}")

# Utilisation
convert_to_training_format()