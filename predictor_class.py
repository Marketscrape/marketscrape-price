from keras.models import load_model
from gru_model import *

# Constants
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_CATEGORY = 1288
BATCH_SIZE = 20000

class PricePredictor:
    def __init__(self):
        self.model = load_model("model copy.h5", custom_objects={"rmsle": rmsle})
    
    def __repr__(self):
        """
        Returns a string representation of the model summary.

        Returns:
            A string representation of the model summary.
        """

        return f"{self.model.summary()}"

    def create_item(self, name, item_description, item_condition_id, category_name, shipping):
        """
        Creates a new item using the provided details.

        Args:
            name: The name of the item.
            item_description: The description of the item.
            item_condition_id: The condition ID of the item.
            category_name: The category name of the item.
            shipping: The shipping information of the item.

        Returns:
            A DataFrame containing the details of the created item.
        """

        item = pd.DataFrame({
            "name": [name],
            "item_description": [item_description],
            "item_condition_id": [item_condition_id],
            "category_name": [category_name],
            "shipping": [shipping]
        })

        return item
    
    def extract_features(self, frame):
        """
        Extracts features from the given DataFrame.

        Args:
            frame: The DataFrame to extract features from.

        Returns:
            The modified DataFrame with extracted features.
        """

        text = np.hstack([frame.item_description.str.lower(), frame.name.str.lower()])
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)

        frame["seq_item_description"] = tokenizer.texts_to_sequences(frame.item_description.str.lower())
        frame["seq_name"] = tokenizer.texts_to_sequences(frame.name.str.lower())

        return frame
    
    def format_data(self, frame):
        """
        Formats the given DataFrame into a dictionary format suitable for model training or prediction.

        Args:
            frame: The DataFrame to format.

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
            'name': pad_sequences(frame.seq_name, maxlen=MAX_NAME_SEQ)
            ,'item_desc': pad_sequences(frame.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ)
            ,'brand_name': np.array(frame.brand_name)
            ,'category_name': np.array(frame.category_name)
            ,'item_condition_id': np.array(frame.item_condition_id)
            ,'num_vars': np.array(frame[["shipping"]])
        }

        return X
    
    def predict(self, data):
        """
        Predicts the price of an item using the trained model.

        Args:
            data: The formatted data for the item to predict the price for.

        Returns:
            The predicted price of the item in CAD (Canadian Dollars), formatted as a string with two decimal places.
        """

        prediction = self.model.predict(data, batch_size=BATCH_SIZE)
        print("prediction is: ", prediction)
        prediction = np.expm1(prediction)[0][0]

        # Cumulative inflation rate from 2018 to 2023 (as of 2023-04-11)
        cumulative_rate = 0.2066
        prediction += prediction * cumulative_rate

        # Current USD to CAD exchange rate (as of 2023-04-11)
        exchange_rate = 1.35
        cad_price = prediction * exchange_rate

        return f"{cad_price:.2f}"

# Load models
vectorizer = joblib.load('vectorizer.pkl')
svm = joblib.load('svm_model.pkl')
brand_le = joblib.load('brand_label_encoder.pkl')
category_le = joblib.load('category_label_encoder.pkl')

# Test item
name = "Mens US10 NIKE Blazer Mid 77"
item_description = "EUC mens size 10 nike blazer mids in white w/ jumbo university blue swoosh. Worn once, mostly. indoors. Price firm. Cash only please, in town meet ups. (Located im Duncan, travel to Victoria regularly.)"
item_condition_id = 2
category_name = "missing"
shipping = 0

# Predict price
object = PricePredictor()
item = object.create_item(name, item_description, item_condition_id, category_name, shipping)
item_tfidf = vectorizer.transform(item['name'] + ' ' + item['item_description'] + ' ' + item['category_name'])

feature = object.extract_features(item)
feature['brand_name'] = svm.predict(item_tfidf)
feature.category_name = category_le.transform(feature.category_name)
feature.brand_name = brand_le.transform(feature.brand_name)

data = object.format_data(feature)
output = object.predict(data)

print(f"Predicted price: ${output} CAD")