import pretty_errors
pretty_errors.configure(
    lines_before=3,
    display_locals=True,  # Enable the display of local variables
)

import codecs
from pathlib import Path

from utils.utils import Vectorizer
from lms.models import FFNLM
from questions.questions import question_solver


DIRECTORIO_CLEAN = Path("..").resolve() / Path("data", "clean", "wiki")

def eval_modelo():

    # texto = []
    # for file in list(DIRECTORIO_CLEAN.iterdir())[:1]:
    #     with codecs.open(file,"r",encoding='utf-8') as archivo:
    #         p = archivo.read()
    #         texto.append(p)
    # vec = Vectorizer(texto)
    # vec.save()
    vec = Vectorizer(None)

    print(len(vec.tokens))
    parameters = {
        "vectorizer":vec,
        "window_length": 3,
        "hidden_size":20
    }
    lm = FFNLM(**parameters)
    lm.load_model()

    q = question_solver()

    mrr = q.evaluate_model(
        lm,\
        max_rank = 8,\
        max_words = 1,\
        verbose=True
    )
    print('El MRR es :', mrr)

