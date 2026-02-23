# XCurveFitLab

**XCurveFitLab** es una aplicación de escritorio hecha en **Python + PyQt5** para ajustar curvas a partir de datos reales usando un **algoritmo genético**.  
La idea del proyecto es simple: cargar un CSV, elegir las columnas de trabajo y ver cómo el modelo va mejorando generación tras generación.

Este repositorio acompaña el trabajo de grado de **Licenciatura en Matemáticas de la Universidad del Quindío**, dentro del proyecto **“Ajuste de curvas usando técnicas evolutivas”**.

---

## ¿Para qué sirve?

Con XCurveFitLab puedes:

- Cargar datos desde un archivo **CSV**.
- Elegir qué columna usar como **X** y cuál como **Y**.
- Probar distintos tipos de modelos de ajuste.
- Ver la evolución del error (**MSE**) en tiempo real.
- Visualizar la curva ajustada sobre los datos originales.
- Revisar métricas como **R²** para tener una idea rápida de la calidad del ajuste.

---

## Modelos disponibles

La app permite trabajar con estos modelos:

- **Lineal**
- **Polinómico (grado 1 a 5)**
- **Exponencial**
- **Logarítmico**
- **Trigonométrico**
- **Logístico**

> Nota: en modelos sensibles (por ejemplo, logarítmico, exponencial y logístico), el código aplica penalizaciones para manejar desbordamientos y valores inválidos.

---

## ¿Cómo está planteado el algoritmo genético?

En términos prácticos, el flujo es:

1. Cada individuo representa una posible solución (un conjunto de parámetros del modelo).
2. Se evalúa su calidad con el **MSE**.
3. Se conservan los mejores, se cruzan y mutan.
4. Se repite el proceso por generaciones hasta encontrar un buen ajuste.

Durante la ejecución se actualizan dos gráficas:

- **Aptitud (MSE) por generación**.
- **Curva ajustada vs. datos originales**.

---

## Estructura del repositorio

Archivos y carpetas principales:

- `XCurveFitLab.py`: aplicación principal con interfaz gráfica.
- `benchmark_modelos_en_la_literatura.py.py`: script de comparación/benchmark.
- `Data_Sets/`: conjuntos de datos de ejemplo.
- `Icons/`: recursos gráficos usados por la interfaz.
- `README.md`: este documento.

---

## Requisitos

- **Python 3.10 o superior** (recomendado: **3.12**)
- Librerías:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `PyQt5`

---

## Instalación rápida

### En Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib PyQt5
```

### En Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib PyQt5
```

---

## Uso de la aplicación

```bash
python XCurveFitLab.py
```

Flujo recomendado dentro de la interfaz:

1. Cargar archivo CSV.
2. Elegir columna para X e Y.
3. Graficar puntos para validar datos.
4. Seleccionar modelo y configurar parámetros del algoritmo genético.
5. Iniciar el ajuste y observar la convergencia.

---

## Benchmark por consola

Si quieres ejecutar pruebas sin GUI:

```bash
python3.12 benchmark_modelos_en_la_literatura.py.py
```

También puedes adaptar ese script para tus propios datasets y comparar modelos con distintos parámetros.
