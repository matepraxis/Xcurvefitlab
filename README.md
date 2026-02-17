# XCurveFitLab

XCurveFitLab es una herramienta de **ajuste de curvas no lineales** con enfoque **evolutivo**, diseñada para explorar y comparar el desempeño de modelos y algoritmos de optimización sobre datos experimentales (CSV). El proyecto integra una **interfaz gráfica (GUI)** para carga de datos, visualización y ejecución del ajuste, y un **script de benchmarking** para comparar resultados contra métodos clásicos.

---

## Objetivo del proyecto

El objetivo principal es **materializar una herramienta práctica** que permita:

- Ajustar modelos no lineales a partir de datos reales cargados desde CSV.
- Visualizar el proceso de optimización (aptitud / MSE) y el ajuste final sobre los datos.
- Incorporar condiciones de **estabilidad numérica** (control de dominios, penalizaciones por overflow, etc.) para modelos sensibles.
- Realizar **pruebas internas por etapas** y comparaciones con métodos clásicos de mínimos cuadrados.
- Proveer una base reproducible para evaluación académica (métricas, gráficas y resultados exportables).

---

## ¿Qué hace XCurveFitLab?

### 1) Interfaz gráfica (GUI)
La aplicación permite:

- Cargar un archivo **CSV**.
- Seleccionar columnas para **Eje X** y **Eje Y**.
- Graficar la nube de puntos.
- Seleccionar el tipo de modelo/algoritmo de ajuste (ej.: trigonométrico, exponencial, logístico, polinómico, etc.).
- Configurar parámetros del proceso evolutivo (población, mutación, cruce, generaciones).
- Ejecutar el algoritmo y observar:
  - Curva de aptitud (MSE) por generaciones
  - Curva ajustada vs datos originales
  - Métricas (por ejemplo, \(R^2\), MSE)

### 2) Benchmarking (métodos clásicos)
Incluye un script para comparar el ajuste de un mismo dataset con enfoques clásicos, como:

- Gauss–Newton (GN)
- Levenberg–Marquardt (LM)
- Newton–Raphson sobre SSE
- Descenso por gradiente sobre SSE
- (y OLS para casos lineales/polinómicos, cuando aplique)

El benchmark exporta resultados a CSV y genera una gráfica comparativa (curvas ajustadas por método).

---

## Estructura del repositorio

> (La estructura puede evolucionar, pero la idea es mantener separación clara entre GUI, scripts, datos y resultados.)



---

## Requisitos

- Python **3.12+** (recomendado)
- Dependencias principales:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `PyQt5` (solo para GUI)

Instalación rápida:
```bash
pip install numpy pandas matplotlib pyqt5


python3.12 benchmark_classic_fits_full.py \
  --csv data.csv \
  --x var5 --y var6 \
  --model trigonometrico \
  --seed 123 --restarts 40 \
  --out resultados.csv \
  --plot --plot_file comparacion.png



