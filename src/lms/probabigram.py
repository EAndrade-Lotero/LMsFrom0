import nltk
from nltk import bigrams, trigrams
from nltk.probability import ConditionalFreqDist
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#nltk.download('punkt')

text = ['Star Wars, conocida también en español como La guerra de las galaxias,[1]​[2]​[3]​ es una franquicia y universo compartido de fantasía compuesta primordialmente de una serie de películas concebidas por el cineasta estadounidense George Lucas en la década de 1970, y producidas y distribuidas inicialmente por 20th Century Fox y posteriormente por The Walt Disney Company a partir de 2012. Su trama describe las vivencias de un grupo de personajes que habitan en una galaxia ficticia e interactúan con elementos como «la Fuerza», un campo de energía metafísico y omnipresente[4]​ que posee un «lado luminoso» impulsado por la sabiduría, la nobleza y la justicia y utilizado por los Jedi, y un «lado oscuro» usado por los Sith y provocado por la ira, el miedo, el odio y la desesperación.']


def get_ngrams(text, ngram_from=2, ngram_to=2, n=None, max_features=20000):
    
    vec = CountVectorizer(ngram_range = (ngram_from, ngram_to), 
                          max_features = max_features, 
                          stop_words=None).fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis = 0) 
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
   
    return words_freq[:n]

bigrams = get_ngrams(text, ngram_from=2, ngram_to=2, n=None)
trigrams = get_ngrams(text, ngram_from=3, ngram_to=3, n=15)
trigrams_df = pd.DataFrame(trigrams)


# Crear distribución de frecuencia condicional para bigramas y trigramas
cfd_bigramas = ConditionalFreqDist(bigrams)
cfd_trigramas = ConditionalFreqDist((trigrama[:-1], trigrama[-1]) for trigrama in trigrams)

trigrams_df = pd.DataFrame(trigrams)
trigrams_df.columns=["Trigram", "Frequency"]
#print(trigrams_df)

bigrams_df = pd.DataFrame(bigrams)
bigrams_df.columns=["Bigram", "Frequency"]
#print(bigrams_df)

# Calcular la probabilidad condicional
ultima_palabra = 'universo'
probabilidad_condicional = cfd_trigramas[trigrams[-1]][ultima_palabra]
print(probabilidad_condicional)

print(trigrams_df)