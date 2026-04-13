# Por qué Naive Bayes funciona aunque mienta

**Unidad 2 · Lectura complementaria 6 · Audiencia: todos**
versión: 2025-1 | modificado: 2026-04-06 | lectura estimada: 8 min

---

Hay algo desconcertante en Naive Bayes. Su nombre lleva la palabra "naïve" — ingenuo — porque asume algo que casi nunca es verdad: que todas las features son independientes entre sí, dada la clase. En diagnóstico médico, el nivel de colesterol y la presión arterial están correlacionados. En clasificación de texto, "machine" y "learning" co-ocurren mucho más de lo esperado por azar. En cualquier dataset real, las correlaciones existen.

Y aun así, Naive Bayes funciona. A veces sorprendentemente bien.

Este texto explora por qué, y cuándo ese "funcionamiento a pesar de todo" tiene un límite.

---

## El supuesto y su violación

El supuesto de NB es que, dado que un correo es spam, la probabilidad de que contenga "oferta" y la probabilidad de que contenga "gratis" son **independientes**:

$$P(\text{oferta}, \text{gratis} \mid \text{spam}) = P(\text{oferta} \mid \text{spam}) \cdot P(\text{gratis} \mid \text{spam})$$

Si en la realidad estas palabras co-ocurren frecuentemente en el mismo correo, NB **subestima** la probabilidad conjunta para la clase spam, porque la descompone como producto de probabilidades marginales que son individualmente altas. La probabilidad resultante no es la correcta.

Pero aquí está la clave: a NB no le interesa que la probabilidad sea correcta en términos absolutos. Solo necesita que **el orden relativo entre clases** sea correcto. Si la probabilidad estimada de spam es mayor que la de no-spam, el modelo clasifica bien, aunque ambas estimaciones sean inexactas.

---

## La condición suficiente: correlaciones simétricas

Hay una condición matemática específica bajo la cual la violación del supuesto no afecta la clasificación. Si las correlaciones entre features son **iguales en todas las clases**, los errores de estimación se cancelan en la comparación.

Pensémoslo así. Sea $\rho_k$ la correlación entre dos features en la clase $k$. Al calcular la log-probabilidad posterior para cada clase, NB comete un error proporcional a $\rho_k$. Si $\rho_A \approx \rho_B$ para todas las clases A y B, ese error es similar en todos los términos y no cambia cuál clase tiene mayor probabilidad.

En la práctica, muchos datasets tienen correlaciones que, aunque altas, son **similares entre clases**. Eso explica por qué NB puede funcionar bien incluso con correlaciones de 0.7 u 0.8.

El problema real ocurre cuando las correlaciones son **asimétricas**: muy altas en una clase y bajas en otra. En ese caso, NB sobreestima la probabilidad de la clase con baja correlación (porque no está siendo "penalizada" por la dependencia) y el clasificador se sesga.

---

## El caso del texto: correlaciones altas y NB exitoso

La clasificación de texto es el dominio donde NB es más famoso y donde también debería fallar más. Las palabras en un texto están lejos de ser independientes. "Machine" y "learning" co-ocurren. "Buenos" y "Aires" casi siempre van juntas. "No" niega lo que sigue, creando una dependencia semántica fuerte.

Y sin embargo, Multinomial NB fue durante décadas el estado del arte en clasificación de texto. ¿Cómo?

La respuesta está en el número de features. Un corpus típico tiene decenas de miles de palabras únicas. La inmensa mayoría de las correlaciones entre palabras son débiles o cero (dos palabras aleatorias raramente co-ocurren). Las correlaciones fuertes afectan a un porcentaje pequeño del vocabulario. Y como el modelo promedia sobre todas las features, el impacto de esas correlaciones fuertes queda diluido.

Hay también un efecto de regularización implícita: al ignorar correlaciones, NB tiene menos parámetros que estimar. Con vocabularios enormes y corpus finitos, eso reduce la varianza de estimación suficientemente como para compensar el sesgo introducido por el supuesto.

---

## Cuándo el supuesto importa: las señales de alerta

Hay tres situaciones en las que el supuesto de independencia sí causa problemas serios:

**Features altamente correlacionadas con correlaciones asimétricas entre clases.** Si la clase A tiene features con $\rho \approx 0.9$ y la clase B tiene $\rho \approx 0.1$, NB sistemáticamente favorecerá a B. LDA o QDA capturan esas diferencias de covarianza y clasifican mejor.

**Pocas features pero altamente informativas y dependientes.** Con 4 o 5 features que cargan toda la información del problema, una correlación fuerte entre ellas no queda diluida. El supuesto de independencia afecta directamente la región de mayor densidad probabilística.

**Necesidad de probabilidades calibradas, no solo etiquetas.** Si el sistema downstream necesita confianzas reales (por ejemplo, para tomar decisiones con umbral variable), las probabilidades de NB son poco confiables. NB produce probabilidades extremas — cercanas a 0 o 1 — con más frecuencia de la esperada, porque las dependencias ignoradas inflan artificialmente la certeza del modelo.

---

## La prueba empírica más simple

Antes de descartar o adoptar NB para un problema concreto, hay una prueba empírica simple: calcular la matriz de correlación de las features **dentro de cada clase** y verificar si las correlaciones son similares entre clases. Si lo son, NB probablemente funcionará bien. Si hay asimetrías grandes, LDA o QDA son mejores candidatos.

También es útil comparar directamente en validación cruzada. Si NB y LDA tienen F1 similar, las correlaciones no están perjudicando a NB. Si hay una brecha consistente en favor de LDA, el supuesto de independencia es el culpable.

---

## Para reflexionar

1. NB asume independencia condicional dado $Y$. ¿Eso equivale a asumir que las features son independientes sin condición? ¿Por qué sí o por qué no?

2. Imagina que tienes un dataset de 1000 features generadas así: la mitad son copia exacta de la otra mitad (correlación perfecta). ¿Cómo afectaría eso a NB? ¿Y a LDA?

3. ¿Se te ocurre un preprocesamiento que pudiera reducir la correlación entre features antes de aplicar NB, sin eliminar información útil?

---

## Para ir más lejos

- Zhang, H. (2004). *The Optimality of Naive Bayes*. FLAIRS Conference 2004.
- Rish, I. (2001). *An empirical study of the naive Bayes classifier*. IJCAI 2001 Workshop on Empirical Methods in AI.
- Hand, D. J. & Yu, K. (2001). *Idiot's Bayes — Not So Stupid After All?* International Statistical Review 69(3). doi:10.1111/j.1751-5823.2001.tb00465.x

---

*Lectura relacionada con la Clase ML_U2_C03 · Sección 2 (Naive Bayes Gaussiano) y el experimento PhD de correlación*
