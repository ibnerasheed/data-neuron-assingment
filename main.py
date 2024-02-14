from fastapi import FastAPI
from pydantic import BaseModel
from simple_text_model import SimpleTextModel

class Texts(BaseModel):
    text1: str
    text2: str

app = FastAPI()

simple_text_model = SimpleTextModel('./best_siamese_model.h5')


@app.post("/")
async def create_item(texts: Texts):
    text1 = texts.text1
    text2 = texts.text2
    
    similarity_score = simple_text_model.evaluate_similarity(text1, text2)
    similarity_score = round(similarity_score.item(), 3)
  
    return {"similarity score": similarity_score}



