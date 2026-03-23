# Bibliografía Recomendada: Machine Learning y Optimización

Este documento recopila las referencias fundamentales para profundizar en los conceptos estudiados en los laboratorios y clases teóricas (Regresión Lineal Simple/Múltiple, Funciones de Costo Cuadráticas y Optimización por Gradiente Descendente). 

**💡 Nota sobre Accesibilidad (Prioridad):** La inmensa mayoría de la literatura de élite en Machine Learning ha sido liberada de forma gratuita por sus propios autores para fomentar la educación abierta. Por favor prefieran los enlaces directos proveídos abajo.

---

## 🟢 Nivel Fundamental y Aplicado (Principalmente Pregrado)

Textos ideales para comenzar. El enfoque es altísimamente práctico y la curva matemática es amigable e intuitiva.

1. **[RECURSO GRATUITO] An Introduction to Statistical Learning (con aplicaciones en Python o R)**
   - **Autores:** Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani.
   - **Enlace Oficial:** [https://www.statlearning.com/](https://www.statlearning.com/) (El PDF completo está disponible legalmente para descarga gratuita).
   - **Enfoque:** Es la introducción estándar de la industria. Cubre maravillosamente de forma intuitiva la regresión lineal simple y múltiple sin saturar al alumno con cálculo avanzado.
   - **Lectura Clave para esta clase:** Capítulo 3 (Linear Regression).

2. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**
   - **Autor:** Aurélien Géron.
   - **Enlace Híbrido:** [Repositorio Oficial de GitHub](https://github.com/ageron/handson-ml3) (Aunque el libro teórico es de pago, el autor libera todos los Jupyter Notebooks con el código 100% gratuito que puedes correr en Google Colab o de forma local).
   - **Enfoque:** Orientado a la ingeniería computacional y programación en Python. Demuestra cómo programar paso a paso los descendentes de gradiente y las clases algorítmicas de `sklearn`.
   - **Lectura Clave para esta clase:** Capítulo 4 (Entrenamiento de Modelos).

3. **[RECURSO GRATUITO] The Hundred-Page Machine Learning Book**
   - **Autor:** Andriy Burkov.
   - **Enlace Oficial:** [http://themlbook.com/](http://themlbook.com/) (El autor distribuye el libro en PDF bajo la política de confianza de "leer primero, pagar después" apoyando la educación libre).
   - **Enfoque:** Un destilado fenomenal de toda la disciplina analítica de datos encapsulada en 100 páginas. Es extremadamente útil generalizando la fundamentación de la Regresión Lineal, sirviendo como el mejor manual rápido de consulta.

4. **Deep Learning with Python**
   - **Autor:** François Chollet (Ingeniero Clave de IA en Google, creador de Keras).
   - **Enlace Híbrido:** [Repositorio Oficial de GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks) (El autor tiene liberado de manera pública todos los Notebooks de práctica de los capítulos).
   - **Enfoque:** A pesar de ser un texto enfocado más adelante hacia las Redes Neuronales, el Capítulo 2 sobre cómo estructurar la Intuición Matemática de Tensores y el Descenso de Gradiente es posiblemente el que mejor explica estos conceptos de manera simplificada en la literatura moderna.

---

## 🔴 Nivel Analítico Avanzado (Posgrado / Profundización Teorico-Matemática)

Textos de altísimo rigor. Se utilizan cuando las guías solicitan demostraciones de teoremas (ej. equivalencia probabilística), optimización convexa, estocástica o de derivadas parciales complejas.

3. **[RECURSO GRATUITO] The Elements of Statistical Learning: Data Mining, Inference, and Prediction**
   - **Autores:** Trevor Hastie, Robert Tibshirani, Jerome Friedman (Universidad de Stanford).
   - **Enlace Oficial:** [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/) (Descarga de PDF gratuito desde la academia).
   - **Enfoque:** El "hermano mayor" y avanzado de *Introduction to Statistical Learning*. Explora las demostraciones de por qué funcionan el estimador de mínimos cuadrados (OLS) y Maximum Likelihood Estimation (MLE).

4. **[RECURSO GRATUITO] Pattern Recognition and Machine Learning**
   - **Autor:** Christopher M. Bishop.
   - **Enlace Oficial:** [https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) (Microsoft Research patrocina el PDF libre).
   - **Enfoque:** La obra suprema del mundo bayesiano y probabilístico. Vital para leer las pruebas analíticas e integrales que afirman que la Distribución Gaussiana genera matemáticamente la función MSE.

5. **[RECURSO GRATUITO] Deep Learning (El "Libro de Ian Goodfellow")**
   - **Autores:** Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press).
   - **Enlace Oficial:** [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) (Lectura completa liberada en formato web HTML).
   - **Enfoque:** Si bien el libro aborda redes profundas, su Parte I establece los fundamentos de optimización numérica pura y la Parte II los cimientos de ML empíricos. Perfecto para posgrado al lidiar con colapso de tasas de aprendizaje (Desvanecimiento o Explosión del Gradiente).
   - **Lectura Clave para esta clase:** Capítulo 4 (Numerical Computation) y Capítulo 5 (Machine Learning Basics).
