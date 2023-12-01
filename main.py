from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, validator

app = FastAPI()

class Sentiment(BaseModel):
    text:str
    sentiment:str

#path 1 (test)
@app.get('/')
def get_root():
    return {"message" : "Test"}

#path 2
@app.post('sentiment')
def post_sentiment(s: Sentiment, background_tasks: BackgroundTasks):
    pass
