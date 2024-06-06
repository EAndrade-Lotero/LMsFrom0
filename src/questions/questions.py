from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import stanza
import re
from copy import deepcopy
from lms.models import FFNLM
import torch

DIRECTORIO_CLOZE_MEMORIA = Path("..").resolve() / Path("data", "cloze", "memoria")

def preprocess_text(text:str) -> list:
    '''
    Pequeño preprocesamiento de sentence segmentation:
        * Elimina puntos de abreviaciones (cm, kg, g)
        * Elimina caracteres extraños
        * Espacios alrededor de símbolos de puntuación
        * Todo a minúsculas
    Input:
        - text
    Output:
        - List of sentences
    '''
    nlp = stanza.Pipeline(
        lang='es', 
        processors='tokenize', 
        use_gpu=False, 
        download_method=None,
        verbose=False
    )
    doc = nlp(text)
    sentences = [str(sentence) for sentence in doc.sentences]
    new_sentences = []
    for sentence in sentences:
        # Elimina puntos de abreviaciones (cm, kg, g)
        sentence = re.sub(r"\bcm.\b", r"cm", sentence)
        sentence = re.sub(r"\bkg.\b", r"kg", sentence)
        sentence = re.sub(r"\bg.\b", r"g", sentence)
        # Elimina caracteres extraños
        sentence = re.sub(r"[^a-zA-Z0-9.\-,;¡!¿?áéíóúüñ\(\)]+", r" ", sentence)
        sentence = re.sub(r"([,;:¡!¿?\(\)])", r" \1 ", sentence)
        # Incluimos espacio antes del punto final
        sentence = re.sub(r"\.$", r" .", sentence)
        # Todo a minúsculas
        sentence = sentence.lower()
        new_sentences.append(sentence)
    return new_sentences

class Text :
    '''
    Class for containing the training text.
    Input:
        - text, a string with the text.
    '''

    def __init__(self, text:str, window_length:int=2):
        self._text = text
        self._sentences = self.sentences()
        self._tokens = self.tokens()
        self.window_length = window_length
        self.end_final = True

    def sentences(self) -> list:
        '''
        Returns the list of sentences from the text.
        '''
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences
        sentences = preprocess_text(self._text)
        self._sentences = sentences
        return sentences

    def tokens(self) -> list:
        '''
        Returns the list of tokens from the text.
        '''
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens
        tokens = [sentence.split() for sentence in self.sentences()] 
        tokens = [word for sentence in tokens for word in sentence]
        tokens = list(set(tokens))
        tokens.sort()
        self._tokens = tokens
        return tokens
    
    def word_stream(self, reps:int, sentences:list=None) -> list:
        '''
        Takes a list of preprocessed sentences and returns a list of words representing the text.
        Input:
            - sentences, a list of sentences as lists of words
            - reps, an integer representing the number of <BEGIN> tokens to
              include at the begining of each sentence.
                  include at the beginning of the stream.
        '''
        if sentences is None:
            sentences = deepcopy(self._sentences)
        '''
        # Includes begin and end tokens for each sentence
        if self.end_final:
            sentences = ['<BEGIN> '*reps + sentence + ' <END>' for sentence in self._sentences]
        else:
            sentences = ['<BEGIN> '*reps + sentence for sentence in self._sentences]
        '''
        # Includes begin and end tokens for each sentence
        if self.end_final:
            sentences = ['<BEGIN> ' + sentence + ' <END>' for sentence in self._sentences]
        else:
            sentences = ['<BEGIN> ' + sentence for sentence in self._sentences]
        sentences[0] = '<BEGIN> '*(reps-1) + sentences[0]
        # Splits stences tokens into lists
        word_stream = [sentence.split() for sentence in sentences] 
        # Flattens the list of lists
        word_stream = [word for sentence in word_stream for word in sentence]
        return word_stream

    def window_sentences(self, accumulated=False) -> list:
        '''
        Returns the list of sentences from the text.
        '''
        if hasattr(self, "_window_sentences") and self._window_sentences:
            return self._window_sentences
        window_sentences = []
        print('Creating window sentences...')
        word_stream = self.word_stream(reps=1)
        if accumulated:
            windows = [word_stream[:i+self.window_length] for i in range(len(word_stream)-self.window_length)]
        else:
            windows = [word_stream[i:i+self.window_length] for i in range(len(word_stream)-self.window_length)]
        for i, w in enumerate(windows):
            window_sentences.append({"x_window":w, "y_next_word":word_stream[i+self.window_length]})
        print('Done!')
        self._window_sentences = window_sentences
        return window_sentences

    def __str__(self):
        return self._text

class ClozeLastQuestions :
    '''
    Class containing cloze questions (last word missing).
    Input:
        - file, string with file name.
        Assumes file is a list of dictionaries
        of the form:
            'id':id(str)
            'pregunta':pregunta(str)
            'respuesta':respuestas(list)
    '''

    def __init__(self, file:str=None):
        self.questions = []
        if file is not None:
            self.load_questions(file)

    def load_questions(self, file:str):
        '''
        Loads question from file
        '''
        suffix = str(file).split('.')[-1]
        if suffix != 'json':
            raise Exception(f'Extensión de archivo invalida ({suffix}), se esperaba json')
        with open(file, 'r', encoding='utf-8') as fp:
            self.questions += json.load(fp)
        fp.close()

    def answer_question(self, question:dict, model:FFNLM, max_rank:int=3 ,verbose:bool=False) -> tuple:
        '''
        Answers a given question with the language model.
        Input:
            - question, dict from self.questions.
            - model, a language model.
            - max_rank, an integer defining the maximum number of words generated by the model.
            - verbose, a boolean to print information about the result.
        Output:
            - result, a boolean for whether the answer is correct.
            - answer, a string with the answer.
            - probability, the probality output by the model conditioned on the question 
        '''
        pregunta = question['pregunta']
        probas = model.probabilities([pregunta])
        values, indices = torch.topk(probas,max_rank)
        if len(indices.shape) > 1:
            assert (indices.shape[0] == 1), 'Ay mk'
            indices = indices.squeeze()
        if len(values.shape) > 1:
            assert (values.shape[0] == 1), 'Ay mk'
            values = values.squeeze()
        respuestas = model.vectorizer.index_to_token(indices.tolist())
        results = list()
        for indice, w in enumerate(respuestas):
            if w in question['respuesta']:
                result = True
            else:
                result = False
            results.append((result, w, values[indice]))
        if verbose:
            for result, w, p in results:
                print(f'La respuesta {w} con probabilidad {p} es {result}')
        return results


    def evaluate_model(self, model:any, max_rank=5, max_words:int=1, verbose:bool=False) -> list:
        '''
        Evaluates a language model with respect to the questions.
        Input:
            - model, a language model.
            - max_rank, an integer defining the maximum number of answers.
            - max_words, an integer defining the maximum number of words generated by the model.
            - verbose, a boolean to print information about the result.
        Output:
            - MRR, float with mean reciprocal rank.
        '''
        contador = 0
        mrr = 0
        for question in tqdm(self.questions):
            if verbose:
                print('\n\n' + '-'*30)
                print(f'Pregunta: {question["pregunta"]}')
                print(f'Respuesta correcta: {question["respuesta"]}\n')
            results = self.answer_question(
                question=question,
                model=model,
                max_rank=max_rank,
                verbose=verbose
            )
            results.sort(key=lambda x:x[2], reverse=True)
            rank_ = 0
            for answer in results:
                if answer[0]:
                    rank_ += 1
                    break
                else:
                    rank_ += 1
            recip_rank = 1./rank_ if rank_ < max_rank else 0
            if verbose:
                marca = '<----' if recip_rank != 0 else ''
                print(f'\nEl rank de pregunta {question["id"]} es {recip_rank}  {marca}')
                print('-'*30 + '\n')
            mrr += recip_rank 
            contador += 1
        return mrr / contador
    
def question_solver() -> ClozeLastQuestions:
    '''
    Returns the questions in data/close as an object of class ClozeLastQuestions
    '''
    q = ClozeLastQuestions()
    for file in DIRECTORIO_CLOZE_MEMORIA.iterdir():
        suffix = str(file).split('.')[-1]
        if suffix == 'json':
            print(f'\tReading {file}...')
            q.load_questions(file)
            print('¡Ok!')
    return q