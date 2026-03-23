# Teorema del Límite Central (TLC)

El **Teorema del Límite Central (Central Limit Theorem, CLT)** es uno de los pilares fundacionales de la teoría de la probabilidad, la estadística inferencial y, por extensión, las demostraciones núcleo de la matemática que rige el Machine Learning.

## 1. Definición Conceptual

El teorema establece un comportamiento asombroso de la naturaleza probabilística:
Dada una población de datos con **cualquier forma de distribución original** (sin importar si es uniforme, asimétrica, bimodal o completamente caótica), siempre y cuando tenga una media límite ($\mu$) y una varianza explícita finita ($\sigma^2$), el acto de promediarla iterativamente crea simetría.

A medida que extraemos muestras aleatorias de tamaño $n$, si $n$ es suficientemente grande ($n \to \infty$), la distribución estandarizada de sus promedios muestrales ($\bar{X}$) colapsará de forma garantizada convergiendo hacia una **Distribución Normal** o Campana de Gauss ($\mathcal{N}$).

$$ \bar{X}_n \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) $$

## 2. Intuición Práctica 

Imagina hipotéticamente un dado honesto de 6 caras. La probabilidad de caída de cada cara es idéntica (un $16.66\%$). Si graficamos esta función límite original, observaremos una caja rectangular plana (**Distribución Uniforme**). No hay nada que indique un "centro" o alguna campana en ese rectángulo.

No obstante, en lugar de lanzar un solo dado, empuñamos **100 dados simultáneamente** y calculamos el promedio aritmético simple originado en esa jugada.
Si registramos ese promedio y repetimos el super-lanzamiento de los cien dados miles de veces graficando los resultados, la distribución plana original colapsa ante la ley de los grandes números. Promediar un "1" perfecto requiere que estadísticamente los cien dados aterricen sincrónicamente en 1 (algo en el borde de lo imposible), mientras que promediar "3.5" agrupará la inmensa mayoría de combinaciones caóticas posibles. 

La Campana de Gauss emerge de forma autónoma. El ruido agregado siempre tiende al consenso central.

## 3. Implicaciones Críticas en Regresión Lineal (Machine Learning)

En clase demostramos analíticamente que la función de costo **Error Cuadrático Medio (MSE)** no es un invento arbitrario. Usarlo garantiza descubrir el estimador óptimo poblacional fundamentado en la Estimación de Máxima Verosimilitud (MLE), **pero únicamente bajo la asunción severa de que el ruido (error) de los datos está distribuido normalmente.**

¿De dónde sale la audacia e impunidad estadística para suponer libremente que nuestros errores residuales en el mundo real serán "Normales" o "Gaussianos"?

La respuesta es la extrapolación directa del Teorema del Límite Central.
El error ($\epsilon^{(i)}$) de nuestra predicción lineal nunca es una variable simple; suele ser en realidad el producto acústico o estocástico sumado de docenas, cientos o miles de características sutiles, aleatorias e independientes que nuestro modelo base ignoró o no midió (por ejemplo sensores defectuosos, cambios ambientales sutiles en la medición, factores genéticos no tipificados, etc).

Al ser la acumulación agregada de cientos de fuerzas minúsculas e independientes en el momento de la inferencia, **la estructura del ruido sumado converge innegablemente hacia una distribución Gaussiana con media cero ($\mu = 0$)**.
Es exactamente este mecanismo oculto de la naturaleza originado en el TLC lo que legaliza y afianza matemáticamente la validez fundacional de usar Regresión Lineal Clásica.
