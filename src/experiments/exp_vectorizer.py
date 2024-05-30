from utils import Vectorizer

texto = ['Qué linda que está la luna, colgada como una fruta.', 'Si se llegara a caer, que golpe tan tenaz.']
vec = Vectorizer(texto)


def test_vectorizer():

    print('Lista de tokens:', vec.tokens)
    print('Cantidad de tokens:', len(vec.tokens))

    oracion = 'qué golpe tan'
    print(f'Considerando la oración \"{oracion}\"')

    tokens = vec.get_tokens(oracion)
    indices = vec.token_to_index(tokens)
    print(f'Índices de los tokens de la oración en el vocabulario:')
    print(indices)
    print('')

    one_hot_oracion = vec.one_hot(oracion)
    print(f'One hot encoding:')
    print(one_hot_oracion)
    print(one_hot_oracion.shape)
    print('Observe que usamos siempre un batch')
    print('')

    de_vuelta = vec.one_hot_to_token(one_hot_oracion)
    print('De regreso a tokens:')
    print(de_vuelta)


def test_batch():

    print('Cantidad de tokens:', len(vec.tokens))

    oracion1 = 'qué luna tan linda'
    oracion2 = 'una fruta no está colgada'
    oraciones = [oracion1, oracion2]
    print('Lista de oraciones:')
    print(oraciones)

    one_hot_batch = vec.one_hot(oraciones)
    print(f'One hot encoding:')
    print(one_hot_batch)
    print(one_hot_batch.shape)

    de_vuelta = vec.one_hot_to_token(one_hot_batch)
    print('De regreso a tokens:')
    print(de_vuelta)
