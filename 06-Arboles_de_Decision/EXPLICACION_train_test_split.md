# Explicación: train_test_split | Celda 19, Línea 4

**Fecha:** 22 de abril de 2026  
**Archivo:** ML_U2_C04_arboles_decision.ipynb  
**Sección:** Entrenamiento del Árbol de Decisión

---

## 📝 Código Analizado

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
```

---

## 🎯 Propósito General

Esta línea **divide el dataset en dos partes independientes: entrenamiento y prueba** usando la función `train_test_split` de scikit-learn. Es un paso fundamental en machine learning para evaluar honestamente el rendimiento de un modelo.

---

## 🔍 Desglose por Componentes

### Parámetros de Entrada

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `X` | DataFrame/Array | Features (características) del dataset completo |
| `y` | Series/Array | Target (etiquetas) del dataset completo |
| `test_size=0.3` | float | Porcentaje de datos para test (30%), el resto (70%) para entrenamiento |
| `random_state=RANDOM_STATE` | int | Semilla para reproducibilidad (valor: 42) |

### Salidas (4 Variables)

```
X_train  → 70% de los features para entrenamiento
X_test   → 30% de los features para prueba
y_train  → 70% de las etiquetas para entrenamiento  
y_test   → 30% de las etiquetas para prueba
```

---

## 📊 Visualización del Proceso

```
Dataset Original (569 muestras × 30 features)
         │
         ├─ 70% (398 muestras) ──→ X_train, y_train
         │
         └─ 30% (171 muestras) ──→ X_test, y_test
```

---

## 🤔 ¿Por Qué es Importante?

### 1. **Evaluación Honesta del Modelo**
```
❌ INCORRECTO: Entrenar y evaluar con los mismos datos
   └─ Accuracy: 100% (pero es falso, memoriza)

✅ CORRECTO: Entrenar con datos nuevos, evaluar con datos no vistos
   └─ Accuracy real: 95% (más confiable)
```

### 2. **Detectar Overfitting**
```
Si:
  - Train Accuracy = 99%
  - Test Accuracy = 70%
  
Entonces: EL MODELO ESTÁ OVERFITTEANDO (memorizando)
```

### 3. **Reproducibilidad**
```python
# Con random_state=42, siempre obtienes la misma división
random_state=42  → División #1: muestras [0, 45, 123, ...]
random_state=42  → División #2: muestras [0, 45, 123, ...] (IDÉNTICA)
```

---

## 📐 Parámetro `test_size=0.3` Explicado

| Razón | Detalle |
|-------|---------|
| **Por qué 30%?** | Regla empírica: 70-30, 80-20 o 90-10 (depende del dataset) |
| **Datasets grandes** | Puedes usar 30% o menos |
| **Datasets pequeños** | Usa 20% para no "perder" datos de entrenamiento |
| **Validación cruzada** | Si tienes pocos datos, evita este simple split |

---

## 🎓 Analogía: El Examen

Imagina que eres estudiante:

```
📚 ESTUDIO (70% del tiempo)
   │
   ├─ Resuelvo problemas 1-70 del libro
   ├─ Entiendo conceptos
   └─ Memorizo fórmulas

📝 EXAMEN (30% del tiempo)
   │
   ├─ Me preguntan problemas 71-100
   ├─ Si entendí → Respondo bien ✅
   └─ Si memoricé → Repruebo ❌
```

Con machine learning es igual:
- **Training set (70%)**: el libro de estudio
- **Test set (30%)**: el examen final

---

## 💻 Ejemplo Práctico Completo

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

# 1. Cargar datos
data = load_breast_cancer()
X = pd.DataFrame(data.data)
y = pd.Series(data.target)

print(f"Dataset original: {X.shape[0]} muestras")
# Output: Dataset original: 569 muestras

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42
)

# 3. Verificar tamaños
print(f"X_train: {X_train.shape[0]} muestras ({X_train.shape[0]/569*100:.1f}%)")
print(f"X_test:  {X_test.shape[0]} muestras ({X_test.shape[0]/569*100:.1f}%)")

# Output:
# X_train: 398 muestras (69.9%)
# X_test:  171 muestras (30.1%)
```

---

## ⚙️ Parámetro `random_state=RANDOM_STATE`

### ¿Qué es RANDOM_STATE?

```python
RANDOM_STATE = 42  # Constante definida al inicio del notebook
```

### ¿Para qué sirve?

**Reproducibilidad**: Asegura que cada ejecución produzca la misma división

### Ejemplo:

```python
# Sin random_state (cada vez diferente):
split_1 = train_test_split(X, y, test_size=0.3)  # Muestras [0, 100, 250, ...]
split_2 = train_test_split(X, y, test_size=0.3)  # Muestras [5, 45, 340, ...]

# Con random_state=42 (siempre igual):
split_1 = train_test_split(X, y, test_size=0.3, random_state=42)  # Muestras [0, 100, 250, ...]
split_2 = train_test_split(X, y, test_size=0.3, random_state=42)  # Muestras [0, 100, 250, ...]
```

---

## 🚀 Flujo Típico Completo

```python
# 1️⃣ PREPARACIÓN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

# 2️⃣ ENTRENAMIENTO (solo con X_train, y_train)
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# 3️⃣ PREDICCIÓN EN TEST (datos nunca vistos)
y_pred = model.predict(X_test)

# 4️⃣ EVALUACIÓN
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en test: {accuracy:.4f}")
```

---

## ⚠️ Errores Comunes

### ❌ Error 1: Evaluar en Training Set
```python
# INCORRECTO:
y_pred_train = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)  # Siempre ≈100%
```

### ❌ Error 2: Olvida random_state
```python
# Cada ejecución da resultados diferentes:
split_1 = train_test_split(X, y, test_size=0.3)  # 95% accuracy
split_2 = train_test_split(X, y, test_size=0.3)  # 92% accuracy
```

### ✅ Correcto:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
y_pred = model.predict(X_test)  # Siempre predice con datos no vistos
accuracy = accuracy_score(y_test, y_pred)  # Evaluación honesta
```

---

## 📚 Conceptos Relacionados

- **Validation Set**: A veces necesitas 3 divisiones (train-val-test) para tuning de hiperparámetros
- **Cross-Validation**: Divide en k folds para mejor estimación, especialmente con datasets pequeños
- **Stratified Split**: Mantiene proporciones de clases en ambas divisiones (importante en clasificación desbalanceada)

---

## 🎯 Resumen Clave

| Concepto | Explicación |
|----------|-------------|
| **Función** | `train_test_split` divide datos en entrenamiento y prueba |
| **Proporción** | 70% entrenamiento, 30% prueba (parámetro `test_size`) |
| **Reproducibilidad** | `random_state=42` asegura divisiones consistentes |
| **Propósito** | Evaluar honestamente sin overfitting |
| **Uso** | X_train/y_train para entrenar; X_test/y_test para evaluar |

---

**Última actualización:** 22 de abril de 2026  
**Autor:** Documentación de Clase - Machine Learning 2026  
**Referencia:** Unidad 2, Clase 4 - Árboles de Decisión
