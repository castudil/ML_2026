# Regularización: El Arte de Saber Cuándo Parar de Aprender

**Unidad 3 · Lectura complementaria 04 · Audiencia: todos**
versión: 2025-1 | modificado: 2026-05-09 | lectura estimada: 6 min

---

Existe una paradoja en el corazón del aprendizaje automático. Los modelos más poderosos —los que tienen más parámetros, más capas, más capacidad— son también los más propensos a fallar de una manera específica: aprenden demasiado bien. Memorizan no solo los patrones genuinos en los datos, sino también el ruido, las coincidencias y las peculiaridades de ese conjunto particular de ejemplos. El resultado es un modelo que obtiene 99% de accuracy en entrenamiento y 70% en producción.

Este fenómeno se llama sobreajuste, y es quizás el problema más importante de la práctica del machine learning. Regularización es el conjunto de técnicas diseñadas para prevenirlo —no evitando que el modelo aprenda, sino guiando qué aprende.

## Por Qué los Modelos Memorizan

Para entender por qué ocurre el sobreajuste, ayuda pensar en un modelo con suficientes parámetros para memorizar cada ejemplo del conjunto de entrenamiento individualmente. Si tienes 100 puntos y un modelo con 1000 parámetros, matemáticamente existe una solución que pasa exactamente por cada uno de esos 100 puntos. Esa solución puede tener pérdida cero en entrenamiento —y ser completamente inútil para datos nuevos.

El problema no es que el modelo sea "demasiado inteligente". Es que el objetivo de entrenamiento —minimizar el error en los datos disponibles— no es el mismo que el objetivo real —generalizar a datos no vistos. El sobreajuste es la brecha entre estos dos objetivos.

La descomposición bias-varianza formaliza este trade-off. El error esperado de un modelo se puede descomponer como la suma de tres términos: el sesgo (qué tan lejos está la predicción promedio del valor real), la varianza (qué tan sensible es la predicción a los datos específicos usados para entrenar) y el ruido irreducible. Los modelos simples tienen alto sesgo pero baja varianza. Los modelos complejos tienen bajo sesgo pero alta varianza. La regularización es, en esencia, una manera de ajustar este equilibrio hacia un punto más favorable.

## Las Tres Estrategias Fundamentales

La regularización L2, también llamada weight decay, añade una penalización proporcional a la magnitud de los pesos a la función de pérdida. El efecto es directo: el optimizador, al intentar minimizar la pérdida total, tiene un incentivo para mantener los pesos pequeños. Pesos pequeños producen funciones más suaves, menos sensibles a perturbaciones pequeñas en la entrada.

Dropout opera con una lógica completamente diferente. En lugar de penalizar los pesos directamente, apaga aleatoriamente un porcentaje de las neuronas en cada paso de entrenamiento. La red nunca puede confiar en que una neurona específica estará disponible, por lo que aprende representaciones más distribuidas y redundantes. En inferencia, todas las neuronas están activas pero sus contribuciones se escalan para mantener la misma expectativa.

Early stopping es conceptualmente la más simple de las tres: detén el entrenamiento cuando el rendimiento en el conjunto de validación deja de mejorar. La curva de validación típicamente sube, alcanza un máximo, y luego baja lentamente a medida que el modelo empieza a memorizar el entrenamiento. El punto máximo es la "zona óptima" donde el modelo ha aprendido los patrones genuinos sin comenzar a memorizar el ruido.

Estas tres estrategias no son mutuamente excluyentes. En la práctica, los mejores modelos combinan las tres: L2 para controlar la magnitud global de los pesos, Dropout para forzar representaciones robustas, y early stopping para no entrenar más allá del punto de generalización óptima.

## Batch Normalization: Un Caso Especial

Batch Normalization, introducida por Ioffe y Szegedy en 2015, no fue diseñada originalmente como técnica de regularización. Su objetivo declarado era estabilizar el entrenamiento de redes profundas normalizando las activaciones de cada capa durante el entrenamiento.

Sin embargo, en la práctica tiene un efecto regularizador significativo. La razón es que usar las estadísticas del mini-batch actual en lugar de las estadísticas de todo el dataset introduce ruido en el proceso de normalización —un ruido que actúa como una forma implícita de perturbación, similar al Dropout.

Este efecto fue debatido inicialmente. Santurkar et al. (2019) demostraron que el mecanismo principal de BatchNorm no es reducir el covariate shift interno (como argumentó el paper original) sino suavizar el paisaje de la función de pérdida, haciendo los gradientes más consistentes y permitiendo tasas de aprendizaje más altas. La regularización implícita es un efecto secundario beneficioso, no el mecanismo central.

## Diagnóstico: Leer las Curvas de Aprendizaje

Las curvas de aprendizaje —accuracy (o pérdida) en función del tamaño del conjunto de entrenamiento— son la herramienta de diagnóstico fundamental para identificar si un modelo sufre de sesgo, varianza, o ninguno.

Un modelo con **sobreajuste** (alta varianza) muestra una brecha grande y estable entre la curva de entrenamiento y la de validación: el modelo aprende bien el train pero no generaliza. La solución típica es más regularización, más datos, o una arquitectura más simple.

Un modelo con **subajuste** (alto sesgo) muestra ambas curvas bajas y convergentes: no importa cuántos datos tengas, el modelo simplemente no tiene capacidad para capturar los patrones. La solución es aumentar la capacidad del modelo (más neuronas, más capas) o reducir la regularización.

Un modelo bien ajustado muestra una brecha pequeña y curvas que convergen hacia un valor razonablemente alto. La validación nunca alcanza el 100% —hay ruido irreducible— pero tampoco se aleja demasiado del entrenamiento.

Saber leer estas curvas es una habilidad práctica de alto valor. La pregunta "¿más datos ayudarían?" tiene respuesta diferente dependiendo de si el problema es sesgo o varianza: más datos reducen la varianza pero no el sesgo.

## Para reflexionar

1. En el diagnóstico de curvas de aprendizaje, ¿cuándo podría ser engañoso usar accuracy como métrica en lugar de pérdida? ¿Hay casos donde accuracy alta en validación coexiste con sobreajuste no detectado?

2. Dropout apaga neuronas aleatoriamente en entrenamiento. ¿Podría aplicarse la misma lógica a las entradas (features) en lugar de las neuronas? ¿Crees que tendría el mismo efecto regularizador? ¿Sabes si existe una técnica que haga esto?

3. Early stopping usa un conjunto de validación separado para decidir cuándo detener el entrenamiento. ¿Puede el conjunto de validación "sobreajustarse" también? ¿Qué pasaría si usas early stopping y luego seleccionas el modelo basándote en muchas corridas con diferentes seeds?

## Para ir más lejos

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958. [El paper original de Dropout]

- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating deep network training by reducing internal covariate shift. *ICML 2015*. arXiv:1502.03167

- Santurkar, S., Tsipras, D., Ilyas, A., & Mądry, A. (2019). How does batch normalization help optimization? *NeurIPS 2019*. arXiv:1805.11604 [Revisión crítica del mecanismo de BatchNorm]

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Cap. 7: Regularization for Deep Learning. https://www.deeplearningbook.org

- Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural networks and the bias/variance dilemma. *Neural Computation*, 4(1), 1–58. [Paper clásico sobre bias-varianza]

---
*Lectura relacionada con la Clase ML_U3_C02 · Sección 2 (Regularización)*
*· Lab ML_U3_Lab02 · Assignment ML_A8*
