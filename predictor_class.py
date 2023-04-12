from keras.models import load_model
from lstm import *

class PricePredictor:
    def __init__(self):
        self.model = load_model("model.h5", custom_objects={"rmsle": rmsle})
    
    def __repr__(self):
        return f"{self.model.summary()}"

    def create_item(self, name, item_description, item_condition_id):
        item = pd.DataFrame({
            "name": [name],
            "item_description": [item_description],
            "item_condition_id": [item_condition_id]
        })

        return item
    
    def extract_features(self, frame):
        text = np.hstack([frame.item_description.str.lower(), frame.name.str.lower()])
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)

        frame["seq_item_description"] = tokenizer.texts_to_sequences(frame.item_description.str.lower())
        frame["seq_name"] = tokenizer.texts_to_sequences(frame.name.str.lower())

        return frame
    
    def format_data(self, frame):
        data = {
            "name": pad_sequences(frame.seq_name, maxlen=MAX_NAME_SEQ),
            "item_description": pad_sequences(frame.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
            "item_condition_id": np.array(frame.item_condition_id).astype(np.int32)
        }

        return data
    
    def predict(self, data):
        prediction = self.model.predict(data, batch_size=BATCH_SIZE)
        prediction = np.expm1(prediction)[0][0]

        # Cumulative inflation rate from 2018 to 2023 (as of 2023-04-11)
        cumulative_rate = 0.2066
        prediction += prediction * cumulative_rate
        # Current USD to CAD exchange rate (as of 2023-04-11)
        exchange_rate = 1.35
        cad_price = prediction * exchange_rate

        return f"{cad_price:.2f}"

# Example: https://www.facebook.com/marketplace/item/727969295788995/?ref=browse_tab&referral_code=marketplace_top_picks&referral_story_type=top_picks
# Actual price: $100
name = "Mens US10 NIKE Blazer Mid 77"
item_description = "EUC mens size 10 nike blazer mids in white w/ jumbo university blue swoosh. Worn once, mostly. indoors. Price firm. Cash only please, in town meet ups. (Located im Duncan, travel to Victoria regularly.)"
# 1: New, 2: Like New, 3: Good, 4: Fair, 5: Poor
item_condition_id = 2

object = PricePredictor()
item = object.create_item(name, item_description, item_condition_id)
feature = object.extract_features(item)
data = object.format_data(feature)
output = object.predict(data)

print(output)
