#pip install wikipedia

import wikipedia
from temaswiki import topics

def descargar_texto_wikipedia(articulo, idioma='es', nombre='output.txt'):
    wikipedia.set_lang(idioma)
    try:
        page = wikipedia.page(articulo)
        contenido = page.content
        #print(f"Descargando contenido de '{articulo}' en la Wikipedia en {idioma}:\n")
        #print(contenido)
        # Guardar el contenido en un archivo de texto
        with open(nombre, 'w', encoding='utf-8') as archivo:
            archivo.write(contenido)
        print(f"Contenido de '{articulo}' guardado exitosamente en '{nombre}'. ")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"El término '{articulo}' es ambiguo. Selecciona una opción más específica.")
    except wikipedia.exceptions.PageError as e:
        print(f"El artículo '{articulo}' no fue encontrado en la Wikipedia en {idioma}.")

for tema in topics:
    descargar_texto_wikipedia(tema, 'es',tema + '.txt')

