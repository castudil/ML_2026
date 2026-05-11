# Paisajes de Pérdida: Geometría de la Optimización en Redes Profundas

**Unidad 3 · Lectura complementaria 05 · Audiencia: Doctorado**
versión: 2025-1 | modificado: 2026-05-09 | lectura estimada: 7 min

---

Cuando entrenamos una red neuronal, navegamos un paisaje. La función de pérdida define una superficie en un espacio de dimensión igual al número de parámetros —millones o miles de millones en redes modernas. El gradiente descendente es nuestro método de navegación: en cada paso, evaluamos la pendiente local y nos movemos cuesta abajo.

El problema es que este paisaje tiene una geometría extraordinariamente compleja, y nuestra intuición sobre espacios tridimensionales nos traiciona al extrapolarlo a dimensiones altas. Entender esa geometría no es un ejercicio académico: explica por qué ciertas redes entrenan mejor que otras, por qué Adam converge rápido pero SGD a veces generaliza mejor, y por qué la inicialización importa tanto.

## Mínimos Locales: Un Problema que No Es el Problema

La intuición inicial sobre la dificultad de optimizar redes neuronales era esta: hay muchos mínimos locales, y el gradiente descendente quedaría atrapado en los peores. Esta intuición resultó ser casi completamente equivocada.

Dauphin et al. (2014) argumentaron, y Choromanska et al. (2015) demostraron bajo supuestos gaussianos, que en espacios de alta dimensión los mínimos locales son raros. Los puntos donde el gradiente es cero pero no son mínimos globales son, predominantemente, puntos de ensilladura —puntos donde la función decrece en algunas direcciones y aumenta en otras. Estos puntos sí son problemáticos porque el gradiente es pequeño cerca de ellos, pero no son "trampas" en el mismo sentido que los mínimos locales.

La razón intuitiva: para que un punto sea un mínimo local en un espacio de $n$ dimensiones, la curvatura debe ser positiva en todas las $n$ direcciones. Si las curvaturas son variables aleatorias independientes, la probabilidad de que todas sean positivas decrece exponencialmente con $n$. En dimensiones altas, casi todos los puntos críticos son sillas.

## La Geometría de los Mínimos Globales

Hay otro hecho sorprendente: las redes neuronales sobrepaametrizadas (con más parámetros que datos) tienen no solo un mínimo global sino toda una variedad de mínimos globales conectada —el "manifold" de cero pérdida. El gradiente descendente puede llegar a cualquier punto de este manifold dependiendo de la inicialización y el camino recorrido.

Aquí emerge la pregunta crucial para la generalización: no todos los puntos del manifold de cero pérdida son igualmente buenos para datos no vistos. Algunos mínimos son "planos" —una pequeña perturbación de los pesos cambia poco la pérdida— y otros son "agudos" —una pequeña perturbación aumenta la pérdida drásticamente.

Keskar et al. (2017) demostraron empíricamente que los mínimos planos generalizan mejor. La intuición: si el mínimo es plano, el modelo no es sensible a pequeñas perturbaciones, incluyendo las perturbaciones introducidas por datos nuevos que difieren ligeramente del entrenamiento. Si el mínimo es agudo, el modelo es frágil.

Este hallazgo tiene implicaciones directas para la elección de optimizador y batch size. Los mini-batches pequeños introducen más ruido en el gradiente, que actúa como una forma de perturbación y empuja al optimizador hacia mínimos más planos. Los mini-batches grandes convergen hacia mínimos más agudos, explicando por qué entrenar con batch sizes muy grandes puede deteriorar la generalización.

## Adam vs. SGD: Una Tensión Productiva

La tensión entre Adam y SGD con momentum es uno de los debates más activos en optimización para deep learning.

Adam adapta la tasa de aprendizaje por parámetro, dividiendo el gradiente por la raíz del segundo momento estimado. Esto acelera la convergencia —especialmente en las primeras épocas— porque los parámetros con gradientes históricamente grandes reciben pasos más pequeños, y los parámetros con gradientes pequeños reciben pasos más grandes.

Pero esta adaptabilidad tiene un costo. Wilson et al. (2017) demostraron en varios benchmarks de visión por computadora que SGD con momentum, con un learning rate ajustado cuidadosamente, alcanza menor error de test que Adam, aunque converge más lento. La hipótesis es que la adaptabilidad de Adam lo lleva hacia mínimos más agudos: al reducir efectivamente el lr en dimensiones de alta curvatura, Adam escapa de esas dimensiones, pero esas dimensiones pueden ser importantes para la generalización.

AdamW (Loshchilov & Hutter, 2019) propone una corrección. En Adam estándar, weight decay interactúa con el paso adaptativo: para parámetros con alta varianza de gradiente (segundo momento grande), la penalización L2 efectiva se reduce. AdamW desacopla la regularización del paso adaptativo, aplicando weight decay directamente sobre los pesos sin escalar. El resultado es que AdamW combina la velocidad de convergencia de Adam con una regularización más efectiva, y se ha convertido en el optimizador de referencia para transformers y modelos de lenguaje grandes.

```python
# [PhD] Comparación empírica: Adam vs SGD+momentum en términos de sharpness
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset sintético pequeño para forzar diferencias de generalización
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr).astype('float32')
X_te_s  = sc.transform(X_te).astype('float32')

def build_model():
    return nn.Sequential(nn.Linear(20,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,2))

def train_and_eval(opt_name, lr=0.01, n_epochs=200, batch_size=32):
    model = build_model()
    if opt_name == 'adam':
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X_tr_s); yt = torch.from_numpy(y_tr.astype('int64'))
    for ep in range(n_epochs):
        idx = torch.randperm(len(Xt))[:batch_size]
        opt.zero_grad(); loss = crit(model(Xt[idx]), yt[idx])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        te_acc = (model(torch.from_numpy(X_te_s)).argmax(1).numpy() == y_te).mean()
    return te_acc

results = {name: [train_and_eval(name) for _ in range(5)]
           for name in ['adam', 'sgd']}
for name, accs in results.items():
    print(f"{name:>6}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
```

La varianza entre corridas (std) es tan informativa como la media: un optimizador más estable es preferible en producción aunque tenga media ligeramente inferior.

## Learning Rate Scheduling: Navegar el Paisaje en Fases

Una práctica estándar en entrenamiento moderno es variar la tasa de aprendizaje durante el entrenamiento. La intuición geométrica es clara: al inicio, pasos grandes permiten explorar el paisaje y llegar a la región del mínimo; al final, pasos pequeños permiten afinar la posición dentro de esa región.

Los schedules más usados son el coseno annealing (reduce lr suavemente siguiendo una curva coseno), warmup lineal (aumenta lr desde cero las primeras K iteraciones, luego decae) y ReduceLROnPlateau (reduce lr cuando la métrica de validación se estanca). Los transformers modernos usan invariablemente warmup + coseno annealing, combinación que empíricamente produce los mejores resultados en modelos de lenguaje grandes.

El warmup merece atención especial. Iniciar con lr alto en las primeras iteraciones puede ser catastrófico porque el modelo está lejos de cualquier mínimo y el gradiente puede ser muy ruidoso. El warmup permite que el optimizador primero estime buenas estadísticas de los momentos (en el caso de Adam) antes de dar pasos grandes.

## Para reflexionar

1. La evidencia sugiere que mínimos planos generalizan mejor. ¿Podría diseñarse un optimizador que explícitamente busque mínimos planos en lugar de solo seguir el gradiente? ¿Cómo medirías la "planura" de un mínimo de manera eficiente durante el entrenamiento?

2. El batch size afecta la calidad del mínimo alcanzado, no solo la velocidad de convergencia. ¿Cómo cambiaría tu estrategia de entrenamiento si tienes una GPU con 80GB de VRAM versus 8GB? ¿Solo aumentarías el batch size o cambiarías algo más?

3. AdamW desacopla weight decay del paso adaptativo. Pero todavía hay debate sobre si AdamW es óptimo. ¿Qué propiedades teóricas debería tener un optimizador ideal para deep learning? ¿Qué información usaría además del gradiente de primer y segundo orden?

## Para ir más lejos

- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*. arXiv:1412.6980 [El paper original de Adam]

- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization (AdamW). *ICLR 2019*. arXiv:1711.05101 [Corrección crítica a Adam]

- Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). On large-batch training for deep learning: Generalization gap and sharp minima. *ICLR 2017*. arXiv:1609.04836 [Mínimos planos vs. agudos]

- Wilson, A. C., Roelofs, R., Stern, M., Srebro, N., & Recht, B. (2017). The marginal value of momentum for small learning rate SGD. *ICLR 2018*. arXiv:1705.08292 [Adam vs SGD en generalización]

- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *NeurIPS 2018*. arXiv:1712.09913 [Visualización de paisajes de pérdida — altamente recomendado]

---
*Lectura relacionada con la Clase ML_U3_C02 · Sección 3 (Optimización)*
*· Lab ML_U3_Lab02 · Assignment ML_A8*
