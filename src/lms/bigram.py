import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


texto = ['ana beto carlos ana beto daniel ana beto edgar ana beto felipe ana beto gabriela']

def get_ngrams(texto, ngram_from=2, ngram_to=2, n=None, max_features=20000):
    
    vec = CountVectorizer(ngram_range = (ngram_from, ngram_to), 
                          max_features = max_features, 
                          stop_words=None).fit(texto)
    bag_of_words = vec.transform(texto)
    sum_words = bag_of_words.sum(axis = 0) 
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
   
    return words_freq[:n]

trigrams = get_ngrams(texto, ngram_from=3, ngram_to=3, n=None)
trigrams_df = pd.DataFrame(trigrams)
trigrams_df.columns=["Trigram", "Frequency"]

# print(trigrams)

lista_trigramas = trigrams_df['Trigram'].tolist()

# Extraer los primeros dos elementos de cada trígrama
primeros_dos_elementos = [' '.join(trigrama.split()[:2]) for trigrama in lista_trigramas]

# Contar la frecuencia de cada trígrama y de sus dos primeros elementos
frecuencia_trigramas = Counter(lista_trigramas)
frecuencia_primeros_dos_elementos = Counter(primeros_dos_elementos)

# Calcular la frecuencia relativa
probabilidad_trigramas = {trigrama: frecuencia_trigramas[trigrama] / frecuencia_primeros_dos_elementos[' '.join(trigrama.split()[:2])] for trigrama in lista_trigramas}

# Crear un DataFrame a partir del diccionario de resultados
df_resultados = pd.DataFrame(list(probabilidad_trigramas.items()), columns=['Trigrama', 'P(w_2 | w_0 w_1)'])

# Mostrar el DataFrame
print(df_resultados)