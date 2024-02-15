from fastapi import FastAPI
from pydantic import BaseModel
from simple_text_model import SimpleTextModel
from fastapi.responses import HTMLResponse

class Texts(BaseModel):
    text1: str
    text2: str

app = FastAPI(title="Data Neuron Assingment API")

simple_text_model = SimpleTextModel('./best_siamese_model.h5')

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Data Neuron Assingment</title>
        </head>
        <body>
            <h1>Data Neuron Assingment</h1>
            <a href="/docs">Click here for documentation.</a>
        </body>
    </html>
    """



@app.post("/")
async def calculate_similarity(texts: Texts):
    text1 = texts.text1
    text2 = texts.text2
    
    similarity_score = simple_text_model.evaluate_similarity(text1, text2)
    similarity_score = round(similarity_score.item(), 3)
  
    return {"similarity score": similarity_score}



