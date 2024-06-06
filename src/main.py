import pretty_errors
pretty_errors.configure(
    lines_before=3,
    display_locals=True,  # Enable the display of local variables
)

from experiments import exp_vectorizer as ExpV
from experiments import exp_ffn_model as ExpFFN
from experiments import exp_dataset as ExpDS
from experiments import exp_training as ExpTr
from experiments import exp_path_patching_noobie as noobie
from questions.eval_modelo import eval_modelo
from experiments import exp_transformer as TRM

if __name__ == '__main__':

    # Prueba de vectorizador
    # ExpV.test_vectorizer()
    # ExpV.test_batch()
    # ExpFFN.test_probability()
    # ExpDS.test_dataset()
    # ExpTr.test_cross_entropy()
    # ExpTr.test_training()
    # eval_modelo()
    # noobie.test_path()
    # noobie.test_perplexity()
    # ExpTr.test_corpus()
	TRM.test_training()