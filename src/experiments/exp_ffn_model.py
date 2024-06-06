from utils.utils import Vectorizer
from lms.models import FFNLM

def test_probability():
    texto = ['Ana Beto Carlos David']
    vec = Vectorizer(texto)
    parameters = {"vectorizer":vec,
                "window_length": 2#,
                #"hidden_size":10
    }
    lm = FFNLM(**parameters)

    words = ['carlos', 'david']
    contexts = [['ana', 'beto'], ['beto', 'carlos']]
    prob = lm.probability(words=words, contexts=contexts)
    for i, word in enumerate(words):
        print(f'P({word}|{contexts[i]}) = {prob[i].item()}')