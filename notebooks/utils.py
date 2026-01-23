import pickle

# Usage:
# docs, questions = load_processed_data()
def load_processed_data(file_path="../data/processed/squad_processed.pkl"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["docs_for_splitter"], data["questions_ground_truth"]