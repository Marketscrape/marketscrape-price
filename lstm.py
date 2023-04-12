import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam
from keras import backend as K

# Based on histogram distribution train.seq_name and train.seq_item_description respectively
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = 0
MAX_CONDITION = 0

BATCH_SIZE = 512 * 3
EPOCHS = 15

def load_data():
    train = pd.read_table("./input/train.tsv")
    test = pd.read_table("./input/test.tsv")

    return train, test

def fill_missing(frame):
    frame["item_description"].fillna(value="missing", inplace=True)
    frame["item_description"].replace(to_replace="No description yet", value="missing", inplace=True)

    return frame

def extract_features(train, test):
    text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    
    global MAX_TEXT
    MAX_TEXT = len(tokenizer.word_index) + 100

    train["seq_item_description"] = tokenizer.texts_to_sequences(train.item_description.str.lower())
    test["seq_item_description"] = tokenizer.texts_to_sequences(test.item_description.str.lower())
    train["seq_name"] = tokenizer.texts_to_sequences(train.name.str.lower())
    test["seq_name"] = tokenizer.texts_to_sequences(test.name.str.lower())

    return train, test

def format_data(frame):
    data = {
        "name": pad_sequences(frame.seq_name, maxlen=MAX_NAME_SEQ),
        "item_description": pad_sequences(frame.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        "item_condition_id": np.array(frame.item_condition_id).astype(np.int32)
    }

    return data

def rmsle(y_true, y_pred) :
    return K.abs(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))

# Source 1: https://www.kaggle.com/code/valkling/mercari-rnn-2ridge-models-with-notes-0-42755/notebook
# Source 2: https://www.kaggle.com/code/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl/notebook
def build_model(x_train, x_valid, y_train, y_valid):
    # Input layers
    name = Input(shape=[x_train["name"].shape[1]], name="name")
    item_description = Input(shape=[x_train["item_description"].shape[1]], name="item_description")
    item_condition_id = Input(shape=[1], name="item_condition_id")

    # Embedding layers
    emb_layer_name = Embedding(input_dim=MAX_TEXT, output_dim=50)(name)
    emb_layer_item_description = Embedding(input_dim=MAX_TEXT, output_dim=50)(item_description)
    emb_layer_item_condition_id = Embedding(input_dim=MAX_CONDITION, output_dim=5)(item_condition_id)

    # RNN layers
    rnn_layer_1 = GRU(units=64, recurrent_dropout=0.1)(emb_layer_item_description)
    rnn_layer_2 = GRU(units=32, recurrent_dropout=0.1)(emb_layer_name)
    rnn_layer_3 = GRU(units=16, recurrent_dropout=0.1)(emb_layer_item_condition_id)

    # Main layers
    main_layer = concatenate([
        rnn_layer_1,
        rnn_layer_2,
        rnn_layer_3
    ])
    main_layer = Dense(units=512, kernel_initializer="normal", activation="relu")(main_layer)
    main_layer = BatchNormalization()(main_layer)
    main_layer = Dense(units=256, kernel_initializer="normal", activation="relu")(main_layer)
    main_layer = BatchNormalization()(main_layer)
    main_layer = Dense(units=128, kernel_initializer="normal", activation="relu")(main_layer)
    main_layer = BatchNormalization()(main_layer)
    main_layer = Dense(units=64, kernel_initializer="normal", activation="relu")(main_layer)
    main_layer = BatchNormalization()(main_layer)

    # Output layer
    output = Dense(1, activation="linear")(main_layer)

    model = Model([name, item_description, item_condition_id], output)
    optimzer = Nadam()
    model.compile(loss="mse", optimizer=optimzer, metrics=[rmsle, "mae"])

    checkpoint = ModelCheckpoint(
        filepath="model.h5", 
        monitor="val_loss", 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False, 
        mode="min",
        patience=5
    )

    model.fit(
        x=x_train, 
        y=y_train.target, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(x_valid, y_valid.target), 
        verbose=1, 
        callbacks=[checkpoint]
    )

    return model

def main():
    train, test = load_data()
    train, test = fill_missing(train), fill_missing(test)
    train, test = extract_features(train, test)

    train["target"] = np.log1p(train.price)

    global MAX_CONDITION
    MAX_CONDITION = np.max([train.item_condition_id.max(), test.item_condition_id.max()]) + 1

    y_train, y_valid = train_test_split(train, random_state=123, train_size=0.99)
    x_train, x_valid = format_data(y_train), format_data(y_valid)

    model = build_model(x_train, x_valid, y_train, y_valid)

    predictions = model.predict(x_valid, batch_size=BATCH_SIZE)
    predictions = np.expm1(predictions)
    y_true = np.array(y_valid.price.values)
    y_predictions = predictions[:, 0]

    error = rmsle(y_true, y_predictions)
    print(f"RMSLE on validation: {error}")

if __name__ == "__main__":
    main()