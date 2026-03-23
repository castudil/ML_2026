regresion lineal.


que es regresion

Concepto generales
Variable dependiente

Variables independientes

Variables que vamos a recolectar y que tienen que ver con un problema en particular que queremos estudiar.
Una característica importante de estas variables es que se **asume** independencia entre ellas.

En un problema de regresión utilizamos estas variables independientes para predecir una variable dependiente.

Lo que hace diferente al problema de regresión del problema de clasificación es la naturaleza de la variable dependiente.

En la regresión la variable dependiente es de naturaleza continua. En otras palabras podría ser por ejemplo un número real.

Un ejemplo concreto es estimar el valor de una propiedad basada en los metros cuadrados de la propiedad, el número de habitaciones, etc. En este caso la variable dependiente es el valor de la propiedad y las variables independientes son los metros cuadrados y el número de habitaciones. Intuitivamente, una propiedad con más metros cuadrados tiende a costar más que una de menor dimensión si es que están ubicadas más o menos en el mismo sector. Puesto que estoy haciendo un ejemplo simple no estoy considerando la ubicación geográfica porque ya tendríamos que considerar variables más complejas para poder hacer la estimación.

Los supuestos en matemáticas nos ofrecen algunos elementos que podrían ser un poco confusos. Por ejemplo asumir independencia entre variables. Consideramos el caso que queremos predecir la edad de una persona en base a la estatura y a su peso. Es claro que la estatura y el peso no son independientes, de hecho una persona más alta tiende a pesar más. Sin embargo, en el contexto de un modelo de regresión lineal, **asumimos** independencia entre las variables independientes. Esto no significa que no podamos usar variables que no son independientes, sino que el modelo asume que son independientes. 

Se viene en la cabeza una frase muy famosa de un estadístico del siglo XX llamado George Fox que dijo "todos los modelos están equivocados, pero algunos son útiles". a mi juicio esta frase encierra profundamente el espíritu ingenieril. Esto porque en la ingeniería no buscamos la verdad absoluta, sino que buscamos modelos que sean útiles para resolver problemas concretos. 

Matemáticamente un problema de revisión lo podemos plantear de la siguiente manera: 

$$ y = f(\mathbf{x}) + \epsilon $$

donde

y Es la variable dependiente que también llamamos variable respuesta o variable objetivo. no es negrita dado que es un escalar.

\mathbf{x} Es el vector de variables independientes que también llamamos variables predictoras o características. se esccirbe en negrita para indicar que es un vector.

\epsilon Es el término de error o ruido. representa la diferencia entre la variable dependiente y la variable independiente.

Lo que queremos hacer en regresión es construir un f. La idea es que esa función se equivoque lo menos posible. En otras palabras, queremos minimizar el error entre la variable dependiente y la variable independiente. 

Cuándo construyamos el F, vamos a tener la posibilidad de alimentar la función con valores que nosotros queramos. La salida será una predicción de la variable dependiente. a esa salida generalmente la denotamos como h theta de x. En otros textos también la podemos ver como $\hat{y}$





