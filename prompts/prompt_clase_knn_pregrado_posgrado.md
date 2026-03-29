# Prompt de Generación: Notebook Clase KNN (Pregrado y Posgrado)

**Rol:** Eres un profesor titular de Ciencias de la Computación experto en Machine Learning y Estructuras de Datos.

**Tarea:** Generar un Jupyter Notebook completo (en formato Markdown con bloques de código Python) para una clase de dos horas sobre K-Nearest Neighbors (KNN), utilizando el esquema proporcionado al final de este prompt.

---

## Contexto de Uso

* El notebook reemplaza las slides tradicionales y se proyecta en pantalla durante la clase.
* Existen **dos instancias de clase separadas**: ~20 estudiantes de pregrado y ~3 estudiantes de posgrado.
* En la clase de pregrado el profesor omite las celdas marcadas como `[POSGRADO]`. En la clase de posgrado se recorre el notebook completo.
* El entorno de ejecución es **Jupyter Notebook clásico** (VS Code, Colab o GitHub). Los widgets interactivos deben funcionar en VS Code y Colab.
* **La conexión a internet en el aula no es confiable.** Todo dataset debe poder cargarse desde una copia local en `data/`. El código debe intentar descarga online y caer en fallback local si falla.

---

## Restricciones de Audiencia y Formato Visual

1. **Nivel Base (Pregrado):** Intuición geométrica, implementaciones de alto nivel usando `scikit-learn` y visualización de fronteras de decisión.
   * **Regla de Formato:** Todos los bloques de texto explicativo (Markdown) dirigidos a pregrado deben estar formateados en color azul utilizando HTML:
     `<div style="color: #0056b3;"> [Texto explicativo pregrado] </div>`

2. **Nivel Avanzado (Posgrado):** Derivaciones matemáticas (Distancia de Mahalanobis), implementación algorítmica desde cero evaluando complejidad computacional ($O(N \cdot d)$ vs $O(d \log N)$), y estructuras de partición espacial (KD-Trees/Ball Trees).
   * **Regla de Formato:** Todos los bloques de texto explicativo (Markdown) dirigidos a posgrado deben estar resaltados con fondo amarillo utilizando HTML:
     `<div style="background-color: #fff9c4; padding: 10px; border-radius: 5px; color: #000;"> [Texto explicativo posgrado] </div>`

3. **Regla de delimitación de secciones de posgrado:** Cada bloque de posgrado debe iniciarse con una celda Markdown con el header:
   `## [POSGRADO] Título de la sección`
   y finalizarse con una celda Markdown que diga:
   `---  <!-- fin bloque posgrado -->`
   Esto permite al profesor identificar visualmente el inicio y fin del bloque a omitir.

4. **Regla de granularidad visual:** Cada celda Markdown debe ser concisa (máximo 8-10 líneas visibles) para su correcta proyección en pantalla. Usar `##` y `###` para jerarquía de secciones. La primera celda del notebook debe ser un índice navegable con anclas HTML a cada sección principal.

---

## Datasets

### Patrón estándar de carga (aplicar a todos los datasets externos)

```python
try:
    # Intento de descarga online
    import seaborn as sns
    df = sns.load_dataset('penguins').dropna()
except Exception:
    # Fallback: copia local
    import pandas as pd
    df = pd.read_csv('data/penguins.csv').dropna()
```

Los archivos locales deben residir en una carpeta `data/` en el mismo directorio del notebook. El notebook debe incluir al inicio una celda de verificación que compruebe si los archivos locales existen e imprima un aviso si no están presentes.

### Datasets utilizados

| Dataset | Fuente | Usado en | Propósito |
| --- | --- | --- | --- |
| `make_moons` | `sklearn.datasets` | Pregrado + Posgrado | Dataset sintético para aislar el efecto geométrico de K. No requiere internet. |
| Palmer Penguins | `seaborn.load_dataset('penguins')` / `data/penguins.csv` | Pregrado | Clasificación con 3 clases y variables continuas. Reemplaza Iris (ya usada en clase anterior). |
| `load_digits` | `sklearn.datasets` | Pregrado + Posgrado | Pregrado: clasificación de dígitos con KNN, motivación visual. Posgrado: benchmark de complejidad y demostración de la Maldición de la Dimensionalidad. No requiere internet. |

---

## Interactividad

* Implementar un widget interactivo con `ipywidgets.interact()` en la Sección 3 que permita variar K (1 a 30) con un slider y visualice en tiempo real la frontera de decisión sobre `make_moons`.
* Incluir obligatoriamente una **versión estática de fallback**: una grilla de gráficos para K ∈ {1, 3, 5, 10, 20} que se muestre si `ipywidgets` no está disponible o el entorno no soporta widgets (ej. render en GitHub).

```python
try:
    import ipywidgets as widgets
    # versión interactiva
except ImportError:
    # versión estática con grilla de subplots
    pass
```

---

## Instrucciones de Ejecución (Chain of Reasoning)

Antes de generar el contenido del notebook, estructura tu razonamiento en un bloque `<thought>` siguiendo estos pasos:

* **Paso 1: Asignación de Tiempos.** Distribuye los 120 minutos de la clase entre las 6 secciones del esquema, asignando tiempos específicos para explicaciones teóricas y ejecución de código.
* **Paso 2: Verificación de datasets.** Confirma que todos los datasets se cargan sin internet usando el patrón de fallback local. Incluye la celda de verificación de archivos `data/` al inicio del notebook.
* **Paso 3: Diseño de Código Pregrado.** Esboza la estructura del código para:
  * Clasificación de Penguins con KNN de sklearn.
  * Clasificación de dígitos con `load_digits` (motivación visual, sin análisis de complejidad).
  * Visualización de fronteras de decisión sobre `make_moons` con widget interactivo.
* **Paso 4: Diseño de Código Posgrado.** Esboza la arquitectura para:
  * Implementación manual de KNN con Distancia de Mahalanobis.
  * Benchmark `%timeit` sobre `load_digits`: búsqueda bruta vs. KD-Tree vs. Ball Tree.
  * Demostración de Maldición de la Dimensionalidad: aplicar PCA a `load_digits` reduciéndolo a 2, 10, 20 y 40 componentes y medir cómo varía la precisión KNN.
* **Paso 5: Ensamblaje.** Traduce el razonamiento a celdas de Markdown (aplicando estrictamente las reglas de formato azul/amarillo y los delimitadores `[POSGRADO]`) y celdas de código Python.

---

## Esquema Base para la Clase

### 1. Índice y Configuración del Entorno

* Tabla de contenidos con anclas HTML.
* Celda de instalación/importación de dependencias.
* Celda de verificación de archivos locales en `data/`.
* Carga de todos los datasets con patrón online/fallback.

### 2. Introducción y Fundamentos (Pregrado)

* Definición: Algoritmo de aprendizaje supervisado, no paramétrico y basado en instancias (lazy learning).
* Intuición: Clasificación por consenso espacial.
* Fases del algoritmo: Almacenamiento, consulta, cálculo de distancias, selección de $K$ vecinos, votación/promedio.
* *Inserto Posgrado:* Contraste detallado entre Aprendizaje Basado en Instancias vs. Modelos Paramétricos ($O(1)$ entrenamiento vs $O(N \cdot d)$ inferencia).

### 3. Métricas de Similitud y Distancia

* Espacio Vectorial: Representación en $\mathbb{R}^d$.
* Métrica de Minkowski y casos especiales (Pregrado): Manhattan ($p=1$), Euclidiana ($p=2$), Chebyshev ($p \to \infty$).
* *Inserto Posgrado:* Aprendizaje de Métricas (Metric Learning) y derivación de la Distancia de Mahalanobis.

### 4. El Hiperparámetro K y Fronteras de Decisión

* Rol de $K$ y análisis de Sesgo-Varianza: $K=1$ (alta varianza, sobreajuste, celdas de Voronoi) vs $K=N$ (alto sesgo, subajuste).
* Widget interactivo: slider de K sobre `make_moons` + fallback estático.
* Estrategias de selección (Pregrado): Validación cruzada.
* *Inserto Posgrado:* Ponderación por Distancia (Distance Weighting) y Regresión Local (LOESS/LOWESS) usando funciones de kernel.

### 5. Estructuras de Datos y Complejidad Algorítmica

* Implementación Naive (Fuerza Bruta) sobre `load_digits`: Complejidad $O(N \cdot d)$. Limitaciones. (Pregrado: ejecución y observación del tiempo; Posgrado: análisis formal de complejidad.)
* *Inserto Posgrado:* Indexación Espacial y Algoritmos de Búsqueda. KD-Trees ($O(d \log N)$ para baja dimensionalidad), Ball Trees (invarianza a rotaciones en alta dimensionalidad) y mención a LSH (Locality-Sensitive Hashing). Benchmark `%timeit` sobre `load_digits`.

### 6. Preprocesamiento Crítico

* Escalado de Características (Pregrado): Normalización y Estandarización sobre Penguins. Demostración del impacto en accuracy con y sin escalado.
* *Inserto Posgrado:* La Maldición de la Dimensionalidad (Curse of Dimensionality). Demostración con `load_digits` + PCA: reducir a 2, 10, 20 y 40 componentes y medir cómo varía la precisión KNN. Justificación de reducción de dimensionalidad previa a KNN.

### 7. Ejercicio Integrador

* Dataset: `load_wine()` de sklearn (sin internet, 3 clases, 13 features).
* Pregrado: escalar datos, buscar K óptimo por cross-validation, comparar accuracy con y sin escalado.
* *Inserto Posgrado:* Repetir con Distancia de Mahalanobis. Comparar con métrica Euclidiana.
