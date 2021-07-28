# Press the green button in the gutter to run the script.
def reviews_preprocessing(file_path):
    # the reviews are present as a json object
    # read the first 1000000 reviews into a pandas df
    reviews = pd.read_json(file_path, lines=True, chunksize=1000000, nrows=1000000)
    for review in reviews:
        df = review
        break
    df.drop(['business_id', 'date', 'review_id', 'user_id', 'cool', 'funny', 'useful'], axis=1, inplace=True)
    text = df.text
    stars = df.stars
    x_train, x_val, y_train, y_val = train_test_split(text, stars, train_size=0.9, random_state=69)
    data = {
        'train': (x_train, y_train),
        'validation': (x_val, y_val)

    }
    return data


def bert_data_pipeline(data, label_list, max_seq_length=128):
    """

    :param data: a dict containing the training and validation data,
    :param label_list: list of prediction labels
    :param max_seq_length: max length of inputs to the bert layer
    :return: inputs suitable for a BERT layer
    """

    (x_train, y_train) = data['train']
    (x_val, y_val) = data['validation']

    train_data = tf.data.Dataset.from_tensor_slices(
        (x_train.values, y_train))  # df.Series.values and y_train already is np ndarray
    val_data = tf.data.Dataset.from_tensor_slices((x_val.values, y_val))

    train_data = (train_data.map(to_feature_map,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
                  .shuffle(1000)
                  .batch(64, drop_remainder=True)
                  .prefetch(tf.data.experimental.AUTOTUNE))

    # valid
    val_data = (val_data.map(to_feature_map,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(64, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

    return (train_data, val_data)


def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_length, tokenizer=tokenizer):
    """

    :param text:
    :param label:
    :param label_list:
    :param max_seq_length:
    :param tokenizer:
    :return:
    """
    example = classifier_data_lib.InputExample(guid=None,
                                               text_a=text.numpy(),
                                               label=label.numpy())

    feature = classifier_data_lib.convert_single_example(0, example, label_list, max_seq_length, tokenizer)
    return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)


def to_feature_map(text, label):
    """
    Wrap a Python Function into a TensorFlow op for Eager Execution
    :param text:
    :param label:
    :return:
    """
    input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text, label],
                                                                  Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

    input_ids.set_shape([max_seq_length])  # set_shape is a tf.Tensor 's method
    input_mask.set_shape([max_seq_length])
    segment_ids.set_shape([max_seq_length])
    label_id.set_shape([])

    x = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    }

    return x, label_id


def create_model(bert_layer):
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_type_ids")
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])

    drop = tf.keras.layers.Dropout(0.4)(pooled_output)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(drop)
    output = tf.keras.layers.Dense(5, activation='softmax', name='op')(
        dense1)  # the ouput units remain 5 even when using sparsecategoricalcrossentropy
    # instead of CategoricalCrossentropy

    model = tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs=output

    )

    return model


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub
    import os
    import pandas as pd
    import sys

    # clone the tensorflow models garden
    sys.path.append('models')  # this step is required to use the next step
    from official.nlp.data import classifier_data_lib  # models/official
    from official.nlp.bert import tokenization
    from official.nlp import optimization
    from sklearn.model_selection import train_test_split

    sys.path.append(os.path.dirname(__file__))
    file_path = os.path.join(os.path.dirname(__file__), "yelp_academic_dataset_review.json")
    reviews_data = reviews_preprocessing(file_path)

    # convert the text and validation data to a form suitable for feeding into a BERT model
    # Label categories
    # maximum length of (token) input sequences
    label_list = [1, 2, 3, 4, 5]
    max_seq_length = 128
    batch_size = 64
    (train_data, val_data) = bert_data_pipeline(reviews_data, label_list, max_seq_length)

    # Get BERT layer and tokenizer:
    # More details here: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()  # will return true if the bert model is uncased else false

    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    model = create_model(bert_layer)
    model.compile(tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  # SparseCategoricalCrossentropy if the labels weren't in one hot encoded form
                  metrics=[
                      tf.keras.metrics.SparseCategoricalAccuracy()])  # SparseCategoricalAccuracy if the labels weren't in ohe
    print(model.summary())

