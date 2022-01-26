import numpy as np
def preprocess_data(text, tokenizer):
    tokens = tokenizer.batch_encode_plus(text, max_length=128, padding='max_length', truncation=True)
    input_word_ids = np.asarray(tokens['input_ids'])
    input_mask = np.asarray(tokens['attention_mask'])
    input_type_ids = np.asarray(tokens['token_type_ids'])
    return ({'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids})


if __name__== '__main__':
    from transformers import BertTokenizer
    import tensorflow as tf
    #add tokenizer path here
    bert_tokenizer_path = "/Users/kakhan/Desktop/saved models/BERT"
    #add model path here
    model_path = "/Users/kakhan/Desktop/saved models/bert_yelp_model"
    print('Loading tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    print('Loading model...')
    keras_model = tf.keras.models.load_model(model_path)


    while True:
        review = input('Enter a review: ')
        text = []
        text.append(review)
        tokens = preprocess_data(text,tokenizer)
        preds = keras_model.predict(tokens)  # model.predict_classes is only available for Sequential models and not for Model class
        # single review by review predictions uses the below steps
        # for batch prediction refer to the colab notebook
        preds = preds.tolist()[0]
        print(preds)
        print('Review:', review, '\tThe rating is :', preds.index(max(preds)) + 1)

        another = input('Do you wish to continue ? y/n: ')
        if another.lower() == 'y':
            continue
        else:
            break


