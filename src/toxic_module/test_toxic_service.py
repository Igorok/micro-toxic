import unittest
from src.toxic_module.toxic_service import ToxicService


toxic_quotes = [
    "Maybe If The Fat Lump Had Given This A Squeeze, He'd Have Remembered To Fall On His Fat Ass.",
    "No One Asked Your Opinion, You Filthy Little Mudblood.",
    "Didn't Mummy Ever Tell You It Was Rude To Eavesdrop, Potter? Oh, Yeah. She Was Dead Before You Could Wipe The Drool Off Your Chin.",
    "You're Gonna Regret This! You And Your Bloody Chicken!",
    "I'm going to kill you, Harry Potter. I'm going to destroy you.",
]
toxic_quotes_predictions = [
    {'toxic': True, 'severe_toxic': False, 'obscene': True, 'threat': False, 'insult': True, 'identity_hate': False},
    {'toxic': True, 'severe_toxic': False, 'obscene': False, 'threat': False, 'insult': False, 'identity_hate': False},
    {'toxic': True, 'severe_toxic': False, 'obscene': False, 'threat': False, 'insult': False, 'identity_hate': False},
    {'toxic': True, 'severe_toxic': False, 'obscene': True, 'threat': False, 'insult': True, 'identity_hate': False},
    {'toxic': True, 'severe_toxic': False, 'obscene': False, 'threat': False, 'insult': False, 'identity_hate': False},
]

not_toxic_quotes = [
    "It takes a great deal of bravery to stand up to your enemies, but a great deal more to stand up to your friends.",
    "No, no. This kind of mark cannot be seen. It lives in your very skin… Love, Harry. Love.",
    "I would trust Hagrid with my own life.",
    "Also, our caretaker, Mr. Filch, has asked me to remind you that the third-floor corridor on the right-hand side is out of bounds to everyone who does not wish to die a most painful death.",
    "It is not our abilities that show what we truly are. It is our choices.",
]
not_toxic_quotes_predictions = [
    {'toxic': False, 'severe_toxic': False, 'obscene': False, 'threat': False, 'insult': False, 'identity_hate': False},
    {'toxic': False, 'severe_toxic': False, 'obscene': False, 'threat': False, 'insult': False, 'identity_hate': False},
    {'toxic': False, 'severe_toxic': False, 'obscene': False, 'threat': False, 'insult': False, 'identity_hate': False},
    {'toxic': False, 'severe_toxic': False, 'obscene': False, 'threat': False, 'insult': False, 'identity_hate': False},
    {'toxic': False, 'severe_toxic': False, 'obscene': False, 'threat': False, 'insult': False, 'identity_hate': False},
]

class Testing(unittest.TestCase):
    def setUp(self):
        self.toxic_service_instance = ToxicService()

    def test_toxic(self):
        for i in range(len(toxic_quotes)):
            prediction = self.toxic_service_instance.is_toxic(toxic_quotes[i])
            self.assertTrue(prediction['is_toxic'])
            self.assertDictEqual(prediction['prediction'], toxic_quotes_predictions[i])

    def test_not_toxic(self):
        for i in range(len(not_toxic_quotes)):
            prediction = self.toxic_service_instance.is_toxic(not_toxic_quotes[i])
            self.assertFalse(prediction['is_toxic'])
            self.assertDictEqual(prediction['prediction'], not_toxic_quotes_predictions[i])

if __name__ == '__main__':
    unittest.main()
