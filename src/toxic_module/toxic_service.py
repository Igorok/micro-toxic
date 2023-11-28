import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

class ToxicService:
    empty_message = 'Empty message!'
    model_path = './src/toxic_module/toxic_model_lstm'

    def __init__(self):
        self.model = load_model(self.model_path)

    def format(self, prediction):
        keys = ['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        return dict(zip(keys, prediction))

    def is_toxic(self, message):
        [prediction] = self.model.predict([message])
        prediction = [bool(round(value)) for value in prediction]

        return {
            'is_toxic': any(prediction),
            'prediction': self.format(prediction),
        }
