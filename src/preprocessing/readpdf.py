from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import re

pdf_folder = Path.cwd() / Path('..').resolve() / Path('data', 'dataoriginal')
pdf_name = 'DocMaestria.pdf'
file_name = pdf_folder / Path(pdf_name)


loader = PyPDFLoader(str(file_name))
pages = loader.load_and_split()





def clip_cadena(cadena_clippear,inicio_quitar):
    fin = re.search(inicio_quitar, cadena_clippear).end()
    return cadena_clippear[fin+1:]
    
texto = []
for page in pages:
    texto.append(clip_cadena(page.page_content,"Página [0-9]+ de [0-9]+"))
    print(page)



for page in texto:
    page = re.sub(" á", "á", page)
    page = re.sub(" é", "é", page)
    page = re.sub(" í", "í", page)
    page = re.sub(" ó", "ó", page)
    page = re.sub(" ú", "ú", page)


text=''.join(texto)

with open("DocMaestria.txt", "w", encoding="utf-8") as f:
    f.write(text)

