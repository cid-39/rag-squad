import pickle
from tqdm import tqdm

# Usage:
# docs, questions = load_processed_data()
def load_processed_data(file_path="../data/processed/squad_processed.pkl"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["docs_for_splitter"], data["questions_ground_truth"]

def load_mini_question_set(file_path="../data/processed/squad_processed_mini.pkl"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["questions_ground_truth"]

def evaluate_retrieval(questions, retriever, k_values=[1, 3, 5, 7, 10]):
    hits = {k: 0 for k in k_values}
    total_mrr = 0.0
    total_questions = len(questions)
    max_k = max(k_values)

    print(f"Starting evaluation on {total_questions} questions...")

    for q_data in tqdm(questions):
        question = q_data['question']
        target_id = q_data['ground_truth_para_id']
        
        retrieved_docs = retriever.invoke(question, k=max_k) 
        retrieved_ids = [doc.metadata.get('para_id') for doc in retrieved_docs]

        # Calculating MRR and Hit Rates
        if target_id in retrieved_ids:
            # MRR 
            rank = retrieved_ids.index(target_id) + 1
            total_mrr += 1.0 / rank
            
            # HitRate
            for k in k_values:
                if rank <= k:
                    hits[k] += 1
        else:
            # target_id not in retrieved list / no hit no rank
            total_mrr += 0.0

    results = {
        "mrr": total_mrr / total_questions,
        "hit_rates": {k: (hits[k] / total_questions) for k in k_values}
    }

    # summary
    print("\n--- Evaluation Results ---")
    print(f"MRR: {results['mrr']:.4f}")
    for k in k_values:
        print(f"Hit Rate@{k}: {results['hit_rates'][k]*100:.2f}%")

    return results