import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import pad_sequences
from keras.optimizers import Nadam
from keras.preprocessing.text import Tokenizer

# Constants
BATCH_SIZE = 512 * 3
EPOCHS = 15
MAX_BRAND = 4810 + 1
MAX_CONDITION = 6
MAX_CATEGORY = 1288
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = 259187

def handle_missing(dataset):
    """
    Fills missing values in the specified dataset columns with the default value "missing".

    Args:
        dataset: The dataset to handle missing values for.

    Returns:
        The modified dataset with filled missing values.
    """

    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    
    return (dataset)

def vectorized_dataset(train, test):
    """
    Vectorizes the given train and test datasets using TF-IDF vectorization. It combines the 'name', 'item_description', and 'category_name' columns from each dataset and creates TF-IDF vectors.

    Args:
        train: The training dataset.
        test: The testing dataset.

    Returns:
        - X_test_tfidf: The TF-IDF vector representation of the combined columns in the test dataset.
        - X_train_tfidf: The TF-IDF vector representation of the combined columns in the train dataset.
        - vectorizer: The trained TF-IDF vectorizer object.
    """

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(train['name'] + ' ' + train['item_description'] + ' ' + train['category_name'])
    X_test_tfidf = vectorizer.transform(test['name'] + ' ' + test['item_description'] + ' ' + test['category_name'])
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return (X_test_tfidf, X_train_tfidf, vectorizer)

def predict_brand_names_for_test( X_test_tfidf):
    """
    Loads a pre-trained SVM model and predicts brand names for the given TF-IDF vectors of the test dataset.

    Args:
        X_test_tfidf: The TF-IDF vector representation of the test dataset.

    Returns:
        None
    """

    svm = joblib.load('svm_model.pkl')

    return svm.predict(X_test_tfidf)

def labelling(train, test):
    """
    Performs label encoding on the 'category_name' and 'brand_name' columns of the train and test datasets.

    Args:
        train: The training dataset.
        test: The testing dataset.

    Returns:
        - train: The modified training dataset with label encoded 'category_name' and 'brand_name' columns.
        - test: The modified testing dataset with label encoded 'category_name' and 'brand_name' columns.
    """

    le = LabelEncoder()

    le.fit(np.hstack([train.category_name, test.category_name]))
    train.category_name = le.transform(train.category_name)
    test.category_name = le.transform(test.category_name)
    joblib.dump(le, 'category_label_encoder.pkl')

    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train.brand_name = le.transform(train.brand_name)
    test.brand_name = le.transform(test.brand_name)
    joblib.dump(le, 'brand_label_encoder.pkl')

    del le

    return (train, test)

def extract_features(train, test):
    """
    Extracts features from the 'item_description' and 'name' columns of the train and test datasets.

    Args:
        train: The training dataset.
        test: The testing dataset.

    Returns:
        - train: The modified training dataset with extracted features.
        - test: The modified testing dataset with extracted features.
    """

    raw_text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])

    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)

    train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
    test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
    train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
    test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())

    return(train, test)

def format_data(dataset):
    """
    Formats the given dataset into a dictionary format suitable for model training or prediction.

    Args:
        dataset: The dataset to format.

    Returns:
        X: A dictionary containing the formatted data, including:
            - 'name': Padded sequences of the 'name' column.
            - 'item_desc': Padded sequences of the 'item_description' column.
            - 'brand_name': NumPy array of the 'brand_name' column.
            - 'category_name': NumPy array of the 'category_name' column.
            - 'item_condition_id': NumPy array of the 'item_condition_id' column.
            - 'num_vars': NumPy array of the 'shipping' column.
    """

    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand_name': np.array(dataset.brand_name)
        ,'category_name': np.array(dataset.category_name)
        ,'item_condition_id': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }

    return X

def rmsle(y_true, y_pred) :
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE) between the true and predicted values.

    Args:
        y_true: The true values.
        y_pred: The predicted values.

    Returns:
        The RMSLE value.
    """

    return K.abs(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))

def build_model(X_train, dtrain, X_valid, dvalid):
    """
    Builds a deep learning model for training and prediction.

    Args:
        X_train: The training data inputs.
        dtrain: The training data targets.
        X_valid: The validation data inputs.
        dvalid: The validation data targets.

    Returns:
        The trained deep learning model.
    """
    
    checkpoint = ModelCheckpoint(
        filepath="model.h5", 
        monitor="val_loss", 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False, 
        mode="min",
        patience=5
    )

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition_id = Input(shape=[1], name="item_condition_id")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    # Embeddings layers
    emb_name = Embedding(MAX_TEXT, 50)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition_id)
    
    # RNN layers
    rnn_layer1 = GRU(units=64, recurrent_dropout=0.1)(emb_item_desc)
    rnn_layer2 = GRU(units=32, recurrent_dropout=0.1)(emb_name)
    rnn_layer3 = GRU(units=16, recurrent_dropout=0.1)(emb_item_condition)
    
    # Main layer
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_category_name)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        , num_vars
    ])
    
    main_l = Dense(units=512, kernel_initializer="normal", activation="relu")(main_l)
    main_l = BatchNormalization()(main_l)
    main_l = Dense(units=256, kernel_initializer="normal", activation="relu")(main_l)
    main_l = BatchNormalization()(main_l)
    main_l = Dense(units=128, kernel_initializer="normal", activation="relu")(main_l)
    main_l = BatchNormalization()(main_l)
    main_l = Dense(units=64, kernel_initializer="normal", activation="relu")(main_l)
    main_l = BatchNormalization()(main_l)
    
    # Output
    output = Dense(1, activation="linear") (main_l)
    optimzer = Nadam()

    # Model
    model = Model(
        [name, item_desc, brand_name, category_name, item_condition_id, num_vars],
        output
    )
    model.compile(loss="mse", optimizer=optimzer, metrics=[rmsle, "mae"])
    
    model.fit(
        X_train,
        dtrain.target,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_data=(X_valid, dvalid.target), 
        verbose=1, callbacks=[checkpoint]
    )
    
    return model

def main():
    # Loading data
    data = pd.read_table("./input/data.tsv")

    train, test = train_test_split(data, test_size=0.2)

    test = test[:1000]
    train, test = handle_missing(train), handle_missing(test)
    train, test = extract_features(train, test)

    train["target"] = np.log1p(train.price)

    X_test_tfidf, _, _ = vectorized_dataset(train, test)

    test['brand_name'] = predict_brand_names_for_test(X_test_tfidf)

    train, test = labelling(train, test)

    # Extracting development test
    dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)
    X_train, X_valid, X_test = format_data(dtrain), format_data(dvalid), format_data(test)

    model = build_model(X_train, dtrain, X_valid, dvalid)

    # Model evaluation
    predictions = model.predict(X_valid)
    predictions = np.expm1(predictions)

    y_true = np.array(dvalid.price.values)
    y_predictions = predictions[:, 0]

    error = rmsle(y_true, y_predictions)
    print(f"RMSLE on validation: {error}")

    # Model prediction
    predictions_on_test = model.predict(X_test, batch_size=BATCH_SIZE)
    predictions_on_test = np.expm1(predictions_on_test)

    y_true = np.array(test.price.values)
    y_predictions = predictions_on_test[:, 0]

    error = rmsle(y_true, y_predictions)
    print(f"RMSLE on test: {error}")

if __name__ == "__main__":
    main()