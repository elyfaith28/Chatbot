
# IMPORTACIONES DE LIBRERÍAS

from fastapi import FastAPI                  # Para crear la API
from pydantic import BaseModel               # Para definir el esquema de entrada (JSON)
import pandas as pd                          # Para manejar dataset en forma de tabla
import random                                # Para generar números aleatorios
from fractions import Fraction               # Para trabajar con fracciones exactas
from sklearn.feature_extraction.text import TfidfVectorizer   # Para convertir texto a vectores numéricos
from sklearn.naive_bayes import MultinomialNB                # Modelo de Machine Learning para clasificar errores

# 1. CREAR LA API
# Se inicializa la aplicación FastAPI con un título descriptivo
app = FastAPI(title="Chatbot Educativo de Operaciones con Fracciones")

# ==============================
# 2. FUNCIÓN PARA GENERAR DATASET
# ==============================
def generar_dataset(n=100):   # n = cantidad de ejemplos a generar
    data = []
    operaciones = ["suma", "resta", "multiplicación", "división"]

    # Generamos n ejemplos de preguntas
    for _ in range(n):
        # Fracciones aleatorias
        a, b = random.randint(1, 9), random.randint(1, 9)  
        c, d = random.randint(1, 9), random.randint(1, 9)  

        # Crear fracciones
        fr1 = Fraction(a, b)
        fr2 = Fraction(c, d)

        # Escoger operación aleatoria
        operacion = random.choice(operaciones)

        # Dependiendo de la operación, calcular resultado real
        if operacion == "suma":
            resultado = fr1 + fr2
            simbolo = "+"
        elif operacion == "resta":
            resultado = fr1 - fr2
            simbolo = "-"
        elif operacion == "multiplicación":
            resultado = fr1 * fr2
            simbolo = "*"
        else:  # división
            resultado = fr1 / fr2
            simbolo = "/"

        # Crear pregunta en texto
        pregunta = f"¿Cuánto es {fr1} {simbolo} {fr2}?"

        # Simulación de posibles errores de un estudiante
        posibles_errores = [
            str(float(resultado)),          # Convertir a decimal en lugar de fracción
            f"{resultado.numerator}/{resultado.denominator + 1}",  # Error en el denominador
            f"{resultado.numerator + 1}/{resultado.denominator}",  # Error en el numerador
            f"{fr1.numerator + fr2.numerator}/{fr1.denominator + fr2.denominator}",  # Sumar directo numeradores/denominadores
        ]

        # Caso correcto
        data.append({
            "pregunta": pregunta,
            "respuesta_estudiante": str(resultado),
            "respuesta_correcta": str(resultado),
            "tipo_error": "Ninguno",
            "retroalimentacion": "¡Excelente! Tu respuesta es correcta."
        })

        # Caso con error (se toma uno aleatorio de la lista de errores)
        error_resp = random.choice(posibles_errores)
        tipo_error = "Decimal en lugar de fracción" if "." in error_resp else "Error de cálculo"
        retro = "Recuerda dejar el resultado como fracción simplificada." if "." in error_resp else "Revisa bien las reglas para operar fracciones."

        data.append({
            "pregunta": pregunta,
            "respuesta_estudiante": error_resp,
            "respuesta_correcta": str(resultado),
            "tipo_error": tipo_error,
            "retroalimentacion": retro
        })

    # Retornar dataset como DataFrame de Pandas
    return pd.DataFrame(data)

# Generamos dataset con 120 ejemplos
data = generar_dataset(120)

# ==============================
# 3. ENTRENAR MODELO DE ML
# ==============================
# X = respuestas de estudiantes
X = data["respuesta_estudiante"]

# y = tipo de error (etiquetas)
y = data["tipo_error"]

# Convertir texto a vectores numéricos
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Entrenar modelo Naive Bayes para clasificar errores
model = MultinomialNB()
model.fit(X_vec, y)

# ==============================
# 4. DEFINIR ESQUEMA DE ENTRADA
# ==============================
# Esto define cómo debe llegar la información en formato JSON
class RespuestaEntrada(BaseModel):
    pregunta: str
    respuesta_estudiante: str

# ==============================
# 5. RUTA PRINCIPAL DE LA API
# ==============================
@app.post("/clasificar/")
def clasificar_error(entrada: RespuestaEntrada):
    # Buscar si la pregunta existe en el dataset
    fila = data[data["pregunta"] == entrada.pregunta]

    if fila.empty:
        return {"error": "Pregunta no encontrada en el dataset."}

    # Respuesta correcta de esa pregunta
    respuesta_correcta = fila["respuesta_correcta"].iloc[0]

    # Si la respuesta del estudiante es igual a la correcta
    if entrada.respuesta_estudiante.strip() == respuesta_correcta.strip():
        return {
            "tipo_error": "Ninguno",
            "retroalimentacion": "¡Excelente! Tu respuesta es correcta."
        }

    # Si la respuesta es incorrecta, predecimos el tipo de error
    entrada_vec = vectorizer.transform([entrada.respuesta_estudiante])
    tipo_error = model.predict(entrada_vec)[0]

    # Buscar retroalimentación asociada a ese tipo de error
    retro = data[data["tipo_error"] == tipo_error]["retroalimentacion"].iloc[0]

    return {
        "tipo_error": tipo_error,
        "retroalimentacion": retro,
        "respuesta_correcta": respuesta_correcta
    }
