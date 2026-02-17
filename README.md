# XCurveFitLab

**XCurveFitLab** es una herramienta desarrollada en **Python** (con interfaz gráfica) para **ajuste de curvas** lineales y no lineales mediante un **Algoritmo Genético (GA)**, orientada a la exploración didáctica y a la validación visual del proceso de optimización.

Este repositorio acompaña el trabajo de grado de **Licenciatura en Matemáticas – Universidad del Quindío**, en el marco del proyecto **“Ajuste de curvas usando técnicas evolutivas”**.

---

## Objetivo del proyecto

Construir una herramienta que permita:

- Ajustar curvas a partir de datos en **CSV** (selección de columnas **X** e **Y**).
- Minimizar el error mediante **MSE** (Mean Squared Error) como función objetivo.
- Visualizar de forma clara:
  - La **convergencia** del algoritmo (MSE por generaciones).
  - La **curva ajustada** sobre la nube de puntos.
  - Métricas complementarias (p. ej., **R²**) y lectura de residuales.

---

## Modelos soportados

XCurveFitLab permite ajustar distintas familias de modelos (según la configuración seleccionada):

- **Lineal:**  \(\hat{y}=ax+b\)
- **Polinómico (grado 1 a 5):**  \(\hat{y}=\sum_{i=0}^{n} a_i x^i\)
- **Exponencial:**  \(\hat{y}=a e^{bx} + c\)
- **Logarítmico:**  \(\hat{y}=a\ln(bx+c)+d\)  *(con restricción \(bx+c>0\))*
- **Trigonométrico:**  \(\hat{y}=a\sin(bx)+c\cos(dx)+e\)
- **Logístico (Verhulst):**  \(\hat{y}=\dfrac{L}{1+e^{-k(x-x_0)}}\)

> Nota: Para modelos sensibles (exponencial/logístico/logarítmico), se aplican **restricciones de dominio** y **penalizaciones** para reducir fallos por overflow/valores inválidos.

---

## ¿Cómo funciona el Algoritmo Genético?

- Cada **individuo** representa un vector de parámetros \(\theta\).
- Se evalúa el individuo con **MSE**.
- Se usa un esquema de:
  - **Selección elitista por truncamiento** (p. ej., mejores individuos).
  - **Cruce** (punto simple).
  - **Mutación** (uniforme).
- Se reporta el mejor individuo y se actualiza la gráfica durante la ejecución.

---

## Estructura del repositorio

Ejemplo típico (puede variar según tus commits):

├─ XCurveFitLab.py # Aplicación GUI (PyQt5)
├─ benchmark_classic_fits_full.py # Script CLI para benchmarks y comparaciones
├─ data.csv # Dataset de prueba (ejemplo)
├─ Data_Sets/ # Conjunto de datasets
├─ Icons/ # Recursos gráficos
└─ README.md



---

## Requisitos

- **Python 3.10+** (recomendado: **Python 3.12**)
- Paquetes:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `PyQt5`

---

## Instalación (recomendada con entorno virtual)

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib PyQt5

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib PyQt5

Ejecución de la aplicación (GUI)

La GUI requiere entorno con pantalla (Windows / Linux con escritorio / macOS).

python XCurveFitLab.py
Flujo en la interfaz:

Cargar CSV.

Elegir columna para X y para Y.

Graficar puntos (validación visual).

Seleccionar modelo y configurar el GA.

Ejecutar y observar MSE por generaciones + curva ajustada.



Benchmarks (modo consola / sin GUI)

Este repositorio incluye un script CLI para comparar ajustes y generar gráficas de salida, ideal para ejecución en entornos sin interfaz.

Ejemplo (dataset propio)
python3.12 benchmark_classic_fits_full.py \
  --csv data.csv \
  --x var5 --y var6 \
  --model exponencial \
  --seed 123 --restarts 40 \
  --out res_var5_var6_exp.csv \
  --plot --plot_file fig_var5_var6_exp.png
