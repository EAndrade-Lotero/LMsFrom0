from pathlib import Path
import stanza
import json
from temaswiki import topics

stanza.download(lang='es')

NLP = stanza.Pipeline(lang='es', processors='tokenize', use_gpu=True, download_method=None)
DIRECTORIO_CLEAN = Path("..").resolve() / Path("data", "clean", "wiki","hola")
DIRECTORIO_CLOZE_MEMORIA = Path("..").resolve() / Path("data", "cloze", "memoria")
DIRECTORIO_CLOZE_MEMORIA.mkdir(exist_ok = True, parents = True)

def create_cloze(file_name):
    '''
    Creates the cloze questions where last word is masked.
    '''
    # --------------------------------------
    # Cargando archivo
    # --------------------------------------
    file_path = DIRECTORIO_CLEAN /  Path(file_name)
    print(f'Intentando leer {file_name}...')
    with open(file_path, encoding = "utf-8") as fp:
        text = fp.read()
    print('¡Ok!')

    # --------------------------------------
    # Pre-procesando
    # --------------------------------------
    # Segmentación de oraciones
    doc = NLP(text)
    #sentences = [sentence.text.lower() for sentence in doc.sentences]
    sentences = [[token.text.lower() for token in sentence.tokens] for sentence in doc.sentences ]
    
    # --------------------------------------
    # Creando preguntas
    # --------------------------------------
    preguntas = []
    for i, sentence in enumerate(sentences):
        # Seleccionamos oraciones cortas
        if 10 < len(sentence) < 20:
            cloze = sentence[:-2] 
            respuesta = sentence[-2]
            preguntas.append({'id':f'{file_name}_{i}', 'pregunta':cloze, 'respuesta':[respuesta]})
            print(f'{sentence} -> \n\t{cloze}')
    # --------------------------------------
    # Guardando en archivo
    # --------------------------------------
    new_name = file_name.split(".")[0] + "_cloze.json"
    cloze_path = DIRECTORIO_CLOZE_MEMORIA /  Path(new_name)
    with open(cloze_path, 'w', encoding='utf-8') as fp:
        json.dump(preguntas, fp, ensure_ascii=False)
    print('¡Listo!')
    print(f'Preguntas guardadas en {cloze_path}')

for tema in topics:
    create_cloze(tema + '.txt')
