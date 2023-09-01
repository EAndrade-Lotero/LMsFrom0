from utils import Vectorizer
from models import FFNLM

texto = ['Qué linda que está la luna, colgada como una fruta.', 'Si se llegara a caer, que golpe tan tenaz.']

vec = Vectorizer(texto)

parameters = {"vectorizer":vec,
              "window_length": 2,
              "hidden_size":10
}
lm = FFNLM(**parameters)

lm.probability(word='luna', context=[['está', 'la'], ['una', 'fruta']])