from fastapi import FastAPI
from typing import Optional
import tensorflow as tf
from transformers import BertTokenizer
import uvicorn
from pydantic import BaseModel
import numpy as np

app = FastAPI()

bert_tokenizer_path = "/Users/kakhan/Desktop/saved models/BERT"
model_path = "/Users/kakhan/Desktop/saved models/bert_yelp_model"

bert_model = None
bert_tokenizer = None

def preprocess_data(text, tokenizer):
    """"
    :param text: text based review to preprocess
    :param tokenizer: BERT tokenizer object
    :return : text data in BERT inpur format
    """
    tokens = tokenizer.batch_encode_plus(text, max_length=128, padding='max_length', truncation=True)
    input_word_ids = np.asarray(tokens['input_ids'])
    input_mask = np.asarray(tokens['attention_mask'])
    input_type_ids = np.asarray(tokens['token_type_ids'])
    return ({'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids})

class review(BaseModel):
    text: str
    stars: Optional[str] = None

# load the model and the tokenizer on startup.
@app.on_event("startup")
def on_startup():
    global bert_model, bert_tokenizer

    bert_model = tf.keras.models.load_model(model_path)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    print(type(bert_tokenizer))

@app.post('/predict')
def prediction_on_review(user_review: review):
    global bert_model, bert_tokenizer
    text = []
    text.append(user_review.text)
    text_tokens = preprocess_data(text, bert_tokenizer)
    preds = bert_model.predict(text_tokens)
    preds = preds.tolist()[0]
    stars = preds.index(max(preds)) + 1
    user_review.stars = str(stars)
    return user_review


if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port= 8000, reload=True)