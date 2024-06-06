import os
import torch
from pathlib import Path

from utils.utils import Vectorizer
from lms.models import Transformer

DATA_FOLDER = Path.cwd() / Path('..').resolve() / Path('data', 'clean', 'wiki')


def test_training():
	# --------------------------------------
	# Loading corpus
	# --------------------------------------
	print('Loading corpus...')
	texto = ''
	lista_textos = [f for f in os.listdir(DATA_FOLDER) if f.split('.')[-1] == 'txt']
	for wiki_txt in lista_textos:
		print(f'\tReading {wiki_txt}...')
		with open(DATA_FOLDER / Path(wiki_txt), encoding='utf-8') as fp:
			texto += fp.read()
		fp.close()
		print(f'¡Ok! Texto de longitud {len(texto.split())}')
	# --------------------------------------
	# Loading Language Model
	# --------------------------------------
	window_length = 3
	embedding_dim = 4
	batch_size = 1
	# ---------------------------------------
	# Descomentar cada cambio de parámetros
	vec = Vectorizer(texto)
	vec.save_vocabulary()
	vocabulary_size = len(vec)
	vec.embeddings = torch.nn.Embedding(
		num_embeddings=vocabulary_size,
		embedding_dim=embedding_dim
	)
	# vec = Vectorizer(None)
	# ---------------------------------------

	lm = Transformer(
		vectorizer=vec,
		window_length=window_length,
		d_model=embedding_dim,
		nhead=1,
		num_encoder_layers=1,
		num_decoder_laters=1,
		dim_feedforward=8,
	)
	print('vocabulary_size: ',lm.vocabulary_size)
	#print('vocabulary: ',lm.vectorizer.tokens)
	words = lm.vectorizer.tokens[-2:]
	print(f'word => {words}')
	word_embeddings = lm.vectorizer.tokens_to_embedding(words)
	print(f'embedding => {word_embeddings}')
	# --------------------------------------
	# Training
	# --------------------------------------
	parameters = {
		"learning_rate":1e-4,
		"window_length":window_length,
		"batch_size":batch_size,
		"num_epochs":5
	}
	print(lm.model.device)
	lm.summary()
	# --------------------------------------
	# Prueba de salida
	# --------------------------------------
	src = torch.rand((batch_size, window_length, embedding_dim))
	print('-'*50)
	print('src.shape', src.shape)
	print('src:')
	print(src)
	# src_mask = torch.ones(1, 1, 3)
	# print('src_mask.shape', src_mask.shape)
	# print(src_mask)
	tgt = torch.rand((batch_size, 1, embedding_dim))
	print('-'*50)
	print('tgt.shape', tgt.shape)
	print('tgt:')
	print(tgt)
	output = lm.model(src, tgt)
	print('-'*50)
	print('output:')
	print(output)
	print('-'*50)
	# # --------------------------------------
	# # Prueba de entrenamiento
	# # --------------------------------------
	# print('Training...')
	# lm.train(texto=texto, parametros=parameters)
	# lm.save_model() 
	# # --------------------------------------
	# # Finding perplexity
	# # --------------------------------------
	# print('Text perplexity:', lm.perplexity(texto))

