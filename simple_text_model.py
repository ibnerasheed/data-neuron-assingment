import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SimpleTextModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.sentence_transformer_model = SentenceTransformer('bert-base-nli-mean-tokens')

    def preprocess_and_encode(self, data):
        if isinstance(data, str):
            data = [data]  
        embeddings = self.sentence_transformer_model.encode(data)
        return embeddings

    def evaluate_similarity(self, text1, text2):
        embeddings1 = self.preprocess_and_encode(text1)
        embeddings2 = self.preprocess_and_encode(text2)
        concatenated_embeddings = self.model.predict([embeddings1, embeddings2])
        similarity_scores = cosine_similarity(concatenated_embeddings[:, :768], concatenated_embeddings[:, 768:])
        return similarity_scores[0][0]

if __name__ == "__main__":
    model_path = './best_siamese_model.keras'  # Path to your pre-trained Siamese model
    simple_text_model = SimpleTextModel(model_path)

    
    new_data_text1 = "nuclear body seeks new tech ......."
    new_data_text2 = "terror suspects face arrest ......"

    similarity_scores = simple_text_model.evaluate_similarity(new_data_text1, new_data_text2)
   
