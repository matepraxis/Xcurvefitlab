import sys
import random
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QSizePolicy, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QLineEdit, QWidget, QApplication, QMainWindow, QFileDialog, QComboBox, QGroupBox, QMessageBox)
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import copy
import time
from datetime import datetime


# Clase Individuo
class Individuo:
    def __init__(self, longitud_gen):
        # Inicializar los genes en un rango más amplio y permitir valores negativos
        self.genes = [random.uniform(-10, 10) for _ in range(longitud_gen)]
        self.aptitud = 0

    # Funciones para calcular la aptitud según el modelo
    def calcular_aptitud_logaritmico(self, x_data, y_data):
        a, b, c, d = self.genes
        errores = []
        for x, y in zip(x_data, y_data):
            try:
                argumento_log = b * x + c
                if argumento_log <= 0:
                    raise ValueError("Argumento de logaritmo no positivo.")
                prediccion = a * np.log(argumento_log) + d
                errores.append((y - prediccion) ** 2)
            except (ValueError, OverflowError):
                errores.append(1e6)  # Asignar un error grande pero finito
        self.aptitud = sum(errores) / len(errores)  # MSE

    def calcular_aptitud_polinomico(self, x_data, y_data, grado):
        errores = [(y - self.calcular_y(x, grado))**2 for x, y in zip(x_data, y_data)]
        self.aptitud = sum(errores) / len(errores)  # MSE

    def calcular_y(self, x, grado):
        y = 0
        for i in range(grado + 1):
            y += self.genes[i] * (x ** (grado - i))
        return y

    def calcular_aptitud_exponencial(self, x_data, y_data):
        a, b, c = self.genes
        errores = []
        for x, y in zip(x_data, y_data):
            try:
                prediccion = a * np.exp(b * x) + c
                errores.append((y - prediccion) ** 2)
            except OverflowError:
                errores.append(1e6)  # Asignar un error grande pero finito
        self.aptitud = sum(errores) / len(errores)  # MSE

    def calcular_aptitud_trigonometrico(self, x_data, y_data):
        a, b, c, d, e = self.genes
        errores = [(y - (a * np.sin(b * x) + c * np.cos(d * x) + e))**2 for x, y in zip(x_data, y_data)]
        self.aptitud = sum(errores) / len(errores)  # MSE

    def calcular_aptitud_logistico(self, x_data, y_data):
        L, k, x0 = self.genes
        errores = []
        for x, y in zip(x_data, y_data):
            try:
                prediccion = L / (1 + np.exp(-k * (x - x0)))
                errores.append((y - prediccion) ** 2)
            except OverflowError:
                errores.append(1e6)  # Asignar un error grande pero finito
        self.aptitud = sum(errores) / len(errores)  # MSE

    # Función para mutar un gen
    def mutar(self, tasa_mutacion):
        if random.random() < tasa_mutacion:
            gen_a_mutar = random.randint(0, len(self.genes) - 1)
            self.genes[gen_a_mutar] += random.uniform(-1, 1)

    # Función de cruce entre individuos
    def cruzar(self, otro, tasa_cruce):
        if random.random() < tasa_cruce:
            gen_a_cruzar = random.randint(0, len(self.genes) - 1)
            self.genes[gen_a_cruzar] = otro.genes[gen_a_cruzar]

    def __repr__(self):
        return f"Individuo con genes: {self.genes}"


# Clase para la gráfica Matplotlib
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use('seaborn-v0_8-darkgrid')
        # Estilo de los gráficos
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot(self, x_data, y_data, title="Gráfico"):
        self.ax.clear()
        self.ax.plot(x_data, y_data, 'r-')
        self.ax.set_title(title, fontsize=14)
        self.ax.set_xlabel("X-axis", fontsize=12)
        self.ax.set_ylabel("Y-axis", fontsize=12)
        self.draw()

    def plot_points(self, x_data, y_data, title="Gráfico de Puntos"):
        self.ax.clear()
        self.ax.scatter(x_data, y_data, color='blue')
        self.ax.set_title(title, fontsize=14)
        self.ax.set_xlabel("X-axis", fontsize=12)
        self.ax.set_ylabel("Y-axis", fontsize=12)
        self.draw()


# Clases de los Algoritmos Genéticos
class GeneticAlgorithmLogarithmicThread(QThread):
    update_plot = pyqtSignal(list, list, list, list, Individuo)

    def __init__(self, x_data, y_data, num_generaciones, tamano_poblacion, tasa_mutacion, tasa_cruce):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.num_generaciones = num_generaciones
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce = tasa_cruce
        self.tamano_poblacion = tamano_poblacion
        self.is_running = True

    def run(self):
        longitud_gen = 4  # Logarítmico tiene 4 genes (a, b, c, d)
        poblacion = [Individuo(longitud_gen) for _ in range(self.tamano_poblacion)]
        mejor_individuo_global = Individuo(longitud_gen)
        mejor_individuo_global.calcular_aptitud_logaritmico(self.x_data, self.y_data)
        historial_aptitud = []

        for gen in range(self.num_generaciones):
            if not self.is_running:
                break

            for individuo in poblacion:
                individuo.calcular_aptitud_logaritmico(self.x_data, self.y_data)

            mejores_poblacion = sorted(poblacion, key=lambda x: x.aptitud)[:10]

            for individuo in poblacion:
                padre = random.choice(mejores_poblacion)
                individuo.cruzar(padre, self.tasa_cruce)

            for individuo in poblacion:
                individuo.mutar(self.tasa_mutacion)

            mejor_individuo = min(poblacion, key=lambda x: x.aptitud)
            if mejor_individuo.aptitud < mejor_individuo_global.aptitud:
                mejor_individuo_global = copy.deepcopy(mejor_individuo)

            historial_aptitud.append(mejor_individuo_global.aptitud)

            y_pred = []
            for x in self.x_data:
                try:
                    argumento_log = mejor_individuo_global.genes[1] * x + mejor_individuo_global.genes[2]
                    if argumento_log <= 0:
                        raise ValueError("Argumento de logaritmo no positivo.")
                    y_p = mejor_individuo_global.genes[0] * np.log(argumento_log) + mejor_individuo_global.genes[3]
                except (ValueError, OverflowError):
                    y_p = 1e6  # Asignar un error grande pero finito
                y_pred.append(y_p)

            y_restantes = []
            for individuo in mejores_poblacion:
                y_rest = []
                for x in self.x_data:
                    try:
                        argumento_log = individuo.genes[1] * x + individuo.genes[2]
                        if argumento_log <= 0:
                            raise ValueError("Argumento de logaritmo no positivo.")
                        y_r = individuo.genes[0] * np.log(argumento_log) + individuo.genes[3]
                    except (ValueError, OverflowError):
                        y_r = 1e6  # Asignar un error grande pero finito
                    y_rest.append(y_r)
                y_restantes.append(y_rest)

            self.update_plot.emit(self.x_data.tolist(), y_pred, historial_aptitud, y_restantes, mejor_individuo_global)

            time.sleep(0.1)

    def stop(self):
        self.is_running = False


class GeneticAlgorithmPolynomialThread(QThread):
    update_plot = pyqtSignal(list, list, list, list, Individuo)

    def __init__(self, x_data, y_data, num_generaciones, grado, tamano_poblacion, tasa_mutacion, tasa_cruce):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.num_generaciones = num_generaciones
        self.grado = grado
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce = tasa_cruce
        self.tamano_poblacion = tamano_poblacion
        self.is_running = True

    def run(self):
        longitud_gen = self.grado + 1
        poblacion = [Individuo(longitud_gen) for _ in range(self.tamano_poblacion)]
        mejor_individuo_global = Individuo(longitud_gen)
        mejor_individuo_global.calcular_aptitud_polinomico(self.x_data, self.y_data, self.grado)
        historial_aptitud = []

        for gen in range(self.num_generaciones):
            if not self.is_running:
                break

            for individuo in poblacion:
                individuo.calcular_aptitud_polinomico(self.x_data, self.y_data, self.grado)

            mejores_poblacion = sorted(poblacion, key=lambda x: x.aptitud)[:10]

            for individuo in poblacion:
                padre = random.choice(mejores_poblacion)
                individuo.cruzar(padre, self.tasa_cruce)

            for individuo in poblacion:
                individuo.mutar(self.tasa_mutacion)

            mejor_individuo = min(poblacion, key=lambda x: x.aptitud)
            if mejor_individuo.aptitud < mejor_individuo_global.aptitud:
                mejor_individuo_global = copy.deepcopy(mejor_individuo)

            historial_aptitud.append(mejor_individuo_global.aptitud)

            y_pred = [mejor_individuo_global.calcular_y(x, self.grado) for x in self.x_data]
            y_restantes = []
            for individuo in mejores_poblacion:
                y_rest = [individuo.calcular_y(x, self.grado) for x in self.x_data]
                y_restantes.append(y_rest)

            self.update_plot.emit(self.x_data.tolist(), y_pred, historial_aptitud, y_restantes, mejor_individuo_global)

            time.sleep(0.1)

    def stop(self):
        self.is_running = False


class GeneticAlgorithmExponentialThread(QThread):
    update_plot = pyqtSignal(list, list, list, list, Individuo)

    def __init__(self, x_data, y_data, num_generaciones, tamano_poblacion, tasa_mutacion, tasa_cruce):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.num_generaciones = num_generaciones
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce = tasa_cruce
        self.tamano_poblacion = tamano_poblacion
        self.is_running = True

    def run(self):
        longitud_gen = 3  # Exponencial tiene 3 genes (a, b, c)
        poblacion = [Individuo(longitud_gen) for _ in range(self.tamano_poblacion)]
        mejor_individuo_global = Individuo(longitud_gen)
        mejor_individuo_global.calcular_aptitud_exponencial(self.x_data, self.y_data)
        historial_aptitud = []

        for gen in range(self.num_generaciones):
            if not self.is_running:
                break

            for individuo in poblacion:
                individuo.calcular_aptitud_exponencial(self.x_data, self.y_data)

            mejores_poblacion = sorted(poblacion, key=lambda x: x.aptitud)[:10]

            for individuo in poblacion:
                padre = random.choice(mejores_poblacion)
                individuo.cruzar(padre, self.tasa_cruce)

            for individuo in poblacion:
                individuo.mutar(self.tasa_mutacion)

            mejor_individuo = min(poblacion, key=lambda x: x.aptitud)
            if mejor_individuo.aptitud < mejor_individuo_global.aptitud:
                mejor_individuo_global = copy.deepcopy(mejor_individuo)

            historial_aptitud.append(mejor_individuo_global.aptitud)

            y_pred = []
            for x in self.x_data:
                try:
                    y_p = mejor_individuo_global.genes[0] * np.exp(mejor_individuo_global.genes[1] * x) + mejor_individuo_global.genes[2]
                except OverflowError:
                    y_p = 1e6  # Asignar un error grande pero finito
                y_pred.append(y_p)

            y_restantes = []
            for individuo in mejores_poblacion:
                y_rest = []
                for x in self.x_data:
                    try:
                        y_r = individuo.genes[0] * np.exp(individuo.genes[1] * x) + individuo.genes[2]
                    except OverflowError:
                        y_r = 1e6  # Asignar un error grande pero finito
                    y_rest.append(y_r)
                y_restantes.append(y_rest)

            self.update_plot.emit(self.x_data.tolist(), y_pred, historial_aptitud, y_restantes, mejor_individuo_global)

            time.sleep(0.1)

    def stop(self):
        self.is_running = False


class GeneticAlgorithmTrigonometricThread(QThread):
    update_plot = pyqtSignal(list, list, list, list, Individuo)

    def __init__(self, x_data, y_data, num_generaciones, tamano_poblacion, tasa_mutacion, tasa_cruce):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.num_generaciones = num_generaciones
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce = tasa_cruce
        self.tamano_poblacion = tamano_poblacion
        self.is_running = True

    def run(self):
        longitud_gen = 5  # Trigonométrico tiene 5 genes (a, b, c, d, e)
        poblacion = [Individuo(longitud_gen) for _ in range(self.tamano_poblacion)]
        mejor_individuo_global = Individuo(longitud_gen)
        mejor_individuo_global.calcular_aptitud_trigonometrico(self.x_data, self.y_data)
        historial_aptitud = []

        for gen in range(self.num_generaciones):
            if not self.is_running:
                break

            for individuo in poblacion:
                individuo.calcular_aptitud_trigonometrico(self.x_data, self.y_data)

            mejores_poblacion = sorted(poblacion, key=lambda x: x.aptitud)[:10]

            for individuo in poblacion:
                padre = random.choice(mejores_poblacion)
                individuo.cruzar(padre, self.tasa_cruce)

            for individuo in poblacion:
                individuo.mutar(self.tasa_mutacion)

            mejor_individuo = min(poblacion, key=lambda x: x.aptitud)
            if mejor_individuo.aptitud < mejor_individuo_global.aptitud:
                mejor_individuo_global = copy.deepcopy(mejor_individuo)

            historial_aptitud.append(mejor_individuo_global.aptitud)

            a, b, c, d, e = mejor_individuo_global.genes
            y_pred = [a * np.sin(b * x) + c * np.cos(d * x) + e for x in self.x_data]

            y_restantes = []
            for individuo in mejores_poblacion:
                a_i, b_i, c_i, d_i, e_i = individuo.genes
                y_rest = [a_i * np.sin(b_i * x) + c_i * np.cos(d_i * x) + e_i for x in self.x_data]
                y_restantes.append(y_rest)

            self.update_plot.emit(self.x_data.tolist(), y_pred, historial_aptitud, y_restantes, mejor_individuo_global)

            time.sleep(0.1)

    def stop(self):
        self.is_running = False


class GeneticAlgorithmLogisticThread(QThread):
    update_plot = pyqtSignal(list, list, list, list, Individuo)

    def __init__(self, x_data, y_data, num_generaciones, tamano_poblacion, tasa_mutacion, tasa_cruce):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.num_generaciones = num_generaciones
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce = tasa_cruce
        self.tamano_poblacion = tamano_poblacion
        self.is_running = True

    def run(self):
        longitud_gen = 3  # Logístico tiene 3 genes (L, k, x0)
        poblacion = [Individuo(longitud_gen) for _ in range(self.tamano_poblacion)]
        mejor_individuo_global = Individuo(longitud_gen)
        mejor_individuo_global.calcular_aptitud_logistico(self.x_data, self.y_data)
        historial_aptitud = []

        for gen in range(self.num_generaciones):
            if not self.is_running:
                break

            for individuo in poblacion:
                individuo.calcular_aptitud_logistico(self.x_data, self.y_data)

            mejores_poblacion = sorted(poblacion, key=lambda x: x.aptitud)[:10]

            for individuo in poblacion:
                padre = random.choice(mejores_poblacion)
                individuo.cruzar(padre, self.tasa_cruce)

            for individuo in poblacion:
                individuo.mutar(self.tasa_mutacion)

            mejor_individuo = min(poblacion, key=lambda x: x.aptitud)
            if mejor_individuo.aptitud < mejor_individuo_global.aptitud:
                mejor_individuo_global = copy.deepcopy(mejor_individuo)

            historial_aptitud.append(mejor_individuo_global.aptitud)

            y_pred = []
            for x in self.x_data:
                try:
                    y_p = mejor_individuo_global.genes[0] / (1 + np.exp(-mejor_individuo_global.genes[1] * (x - mejor_individuo_global.genes[2])))
                except OverflowError:
                    y_p = 1e6  # Asignar un error grande pero finito
                y_pred.append(y_p)

            y_restantes = []
            for individuo in mejores_poblacion:
                y_rest = []
                for x in self.x_data:
                    try:
                        y_r = individuo.genes[0] / (1 + np.exp(-individuo.genes[1] * (x - individuo.genes[2])))
                    except OverflowError:
                        y_r = 1e6  # Asignar un error grande pero finito
                    y_rest.append(y_r)
                y_restantes.append(y_rest)

            self.update_plot.emit(self.x_data.tolist(), y_pred, historial_aptitud, y_restantes, mejor_individuo_global)

            time.sleep(0.1)

    def stop(self):
        self.is_running = False


# Clase principal para la ventana y la interfaz gráfica
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("XCurveFitLab")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('Icons/Logo.png'))

        # Inicialización de variables
        self.csv_data = None
        self.algorithm_type = "Logarítmico"
        self.algorithm_thread = None
        self.x_column = None
        self.y_column = None

        # Estado y últimos resultados (para exportar reporte)
        self.algorithm_running = False
        self.last_run = None

        # Parámetros por defecto
        self.tamano_poblacion = 100
        self.tasa_mutacion = 0.001
        self.tasa_cruce = 0.1

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(20, 20, 20, 20)

        # Agrupar elementos relacionados en QGroupBox
        data_group = QGroupBox("Datos de Entrada")
        data_layout = QVBoxLayout()
        data_layout.setSpacing(10)

        # Botón para cargar CSV
        data_layout.addWidget(self.create_load_csv_button())

        # Selectores de columnas
        x_label, x_selector = self.create_x_column_selector()
        y_label, y_selector = self.create_y_column_selector()
        data_layout.addWidget(x_label)
        data_layout.addWidget(x_selector)
        data_layout.addWidget(y_label)
        data_layout.addWidget(y_selector)

        # Botón para graficar puntos
        data_layout.addWidget(self.create_plot_button())

        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)

        # Grupo de configuración del algoritmo
        algo_group = QGroupBox("Configuración del Algoritmo")
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(10)

        # Selector de algoritmo
        algo_layout.addWidget(QLabel("Seleccionar Tipo de Algoritmo:"))
        algo_layout.addWidget(self.create_algorithm_selector())

        # Grado del Polinomio
        grade_label, grade_input = self.create_grade_input()
        algo_layout.addWidget(grade_label)
        algo_layout.addWidget(grade_input)

        # Parámetros del algoritmo genético
        label, population_input = self.create_population_size_input()
        algo_layout.addWidget(label)
        algo_layout.addWidget(population_input)

        label, mutation_input = self.create_mutation_rate_input()
        algo_layout.addWidget(label)
        algo_layout.addWidget(mutation_input)

        label, crossover_input = self.create_crossover_rate_input()
        algo_layout.addWidget(label)
        algo_layout.addWidget(crossover_input)

        algo_layout.addWidget(self.create_generation_label())
        algo_layout.addWidget(self.create_generation_input())

        # Botones para iniciar y detener el algoritmo
        algo_layout.addWidget(self.create_start_algorithm_button())
        algo_layout.addWidget(self.create_stop_algorithm_button())
        algo_layout.addWidget(self.create_residual_analysis_button())

        algo_group.setLayout(algo_layout)
        left_layout.addWidget(algo_group)

        left_layout.addStretch()

        # Agregar el texto del desarrollador
        developer_label = QLabel("XCurveFitLab fue desarrollado por Diego Alejandro Marulanda Patiño")
        developer_label.setAlignment(Qt.AlignCenter)
        developer_label.setStyleSheet("font-size: 12px; color: #888;")
        left_layout.addWidget(developer_label)

        main_layout.addLayout(left_layout, 1)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(20, 20, 20, 20)

        # Gráficos
        self.canvas1 = MatplotlibCanvas(self, width=5, height=4)
        right_layout.addWidget(self.canvas1)
        self.canvas2 = MatplotlibCanvas(self, width=5, height=4)
        right_layout.addWidget(self.canvas2)

        main_layout.addLayout(right_layout, 4)

        # Crear el widget central
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Activar modo interactivo en Matplotlib
        plt.ion()

    # Métodos para crear los widgets
    def create_load_csv_button(self):
        self.load_csv_button = QPushButton("Cargar archivo CSV")
        self.load_csv_button.setStyleSheet(self.button_style())
        self.load_csv_button.clicked.connect(self.load_csv_file)
        self.load_csv_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.load_csv_button.setToolTip("Selecciona un archivo CSV con los datos a analizar.")
        self.load_csv_button.setIcon(QIcon('Icons/CSV.png'))
        self.load_csv_button.setIconSize(self.load_csv_button.sizeHint() * 0.5)
        return self.load_csv_button

    def create_x_column_selector(self):
        self.x_column_label = QLabel("Seleccionar Columna para Eje X:")
        self.x_column_label.setStyleSheet("font-weight: bold;")
        self.x_column_selector = QComboBox()
        self.x_column_selector.setEnabled(False)
        self.x_column_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.x_column_selector.currentIndexChanged.connect(self.update_x_column)
        self.x_column_selector.setToolTip("Selecciona la columna que representa los valores de X.")
        return self.x_column_label, self.x_column_selector

    def create_y_column_selector(self):
        self.y_column_label = QLabel("Seleccionar Columna para Eje Y:")
        self.y_column_label.setStyleSheet("font-weight: bold;")
        self.y_column_selector = QComboBox()
        self.y_column_selector.setEnabled(False)
        self.y_column_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.y_column_selector.currentIndexChanged.connect(self.update_y_column)
        self.y_column_selector.setToolTip("Selecciona la columna que representa los valores de Y.")
        return self.y_column_label, self.y_column_selector

    def create_plot_button(self):
        self.plot_button = QPushButton("Graficar Puntos")
        self.plot_button.setStyleSheet(self.button_style())
        self.plot_button.setEnabled(False)
        self.plot_button.clicked.connect(self.plot_csv_points)
        self.plot_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.plot_button.setToolTip("Grafica los puntos seleccionados del archivo CSV.")
        self.plot_button.setIcon(QIcon('Icons/Graficar.png'))
        self.plot_button.setIconSize(self.plot_button.sizeHint() * 0.8)
        return self.plot_button

    def create_algorithm_selector(self):
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems(["Logarítmico", "Polinómico", "Exponencial", "Trigonométrico", "Logístico"])
        self.algorithm_selector.currentIndexChanged.connect(self.update_algorithm_type)
        self.algorithm_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.algorithm_selector.setEnabled(False)
        self.algorithm_selector.setToolTip("Selecciona el tipo de algoritmo para ajustar la curva.")
        return self.algorithm_selector

    def create_grade_input(self):
        self.grade_label = QLabel("Grado del Polinomio (1-5):")
        self.grade_label.setStyleSheet("font-weight: bold;")
        self.grade_input = QLineEdit()
        self.grade_input.setValidator(QIntValidator(1, 5))
        self.grade_input.setEnabled(False)
        self.grade_input.setToolTip("Especifica el grado del polinomio para el ajuste polinómico.")
        return self.grade_label, self.grade_input

    def create_population_size_input(self):
        label = QLabel("Tamaño de Población (100-10000):")
        label.setStyleSheet("font-weight: bold;")
        self.population_input = QLineEdit()
        self.population_input.setValidator(QIntValidator(100, 10000))
        self.population_input.setText(str(self.tamano_poblacion))
        self.population_input.setToolTip("Establece el tamaño de la población para el algoritmo genético.")
        return label, self.population_input

    def create_mutation_rate_input(self):
        label = QLabel("Tasa de Mutación (0.001 - 1.0):")
        label.setStyleSheet("font-weight: bold;")
        self.mutation_input = QLineEdit()
        self.mutation_input.setValidator(QDoubleValidator(0.001, 1.0, 3))
        self.mutation_input.setText(str(self.tasa_mutacion))
        self.mutation_input.setToolTip("Define la tasa de mutación para el algoritmo genético.")
        return label, self.mutation_input

    def create_crossover_rate_input(self):
        label = QLabel("Tasa de Cruce (0.1 - 1.0):")
        label.setStyleSheet("font-weight: bold;")
        self.crossover_input = QLineEdit()
        self.crossover_input.setValidator(QDoubleValidator(0.1, 1.0, 3))
        self.crossover_input.setText(str(self.tasa_cruce))
        self.crossover_input.setToolTip("Define la tasa de cruce para el algoritmo genético.")
        return label, self.crossover_input

    def create_generation_label(self):
        self.generation_label = QLabel("Generaciones (1-2000):")
        self.generation_label.setStyleSheet("font-weight: bold;")
        return self.generation_label

    def create_generation_input(self):
        self.generation_input = QLineEdit()
        self.generation_input.setValidator(QIntValidator(1, 2000))
        self.generation_input.setToolTip("Especifica el número de generaciones para el algoritmo genético.")
        return self.generation_input

    def create_start_algorithm_button(self):
        self.start_algorithm_button = QPushButton("Iniciar Algoritmo")
        self.start_algorithm_button.setStyleSheet(self.button_style())
        self.start_algorithm_button.setEnabled(False)
        self.start_algorithm_button.clicked.connect(self.start_algorithm)
        self.start_algorithm_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_algorithm_button.setToolTip("Inicia el algoritmo genético para ajustar la curva.")
        self.start_algorithm_button.setIcon(QIcon('Icons/START.png'))
        self.start_algorithm_button.setIconSize(self.start_algorithm_button.sizeHint() * 0.6)
        return self.start_algorithm_button

    def create_stop_algorithm_button(self):
        self.stop_algorithm_button = QPushButton("Detener Algoritmo")
        self.stop_algorithm_button.setStyleSheet(self.stop_button_style())
        self.stop_algorithm_button.setEnabled(False)
        self.stop_algorithm_button.clicked.connect(self.stop_algorithm)
        self.stop_algorithm_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_algorithm_button.setToolTip("Detiene la ejecución del algoritmo genético.")
        self.stop_algorithm_button.setIcon(QIcon('Icons/STOP.png'))
        self.stop_algorithm_button.setIconSize(self.start_algorithm_button.sizeHint() * 0.8)
        return self.stop_algorithm_button

    def create_residual_analysis_button(self):
        # Botón oculto para el usuario hasta que el algoritmo termine
        self.residuals_button = QPushButton("Análisis de Residuales")
        self.residuals_button.setStyleSheet(self.button_style())
        self.residuals_button.setEnabled(False)
        self.residuals_button.clicked.connect(self.export_residuals_report)
        self.residuals_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.residuals_button.setToolTip(
            "Genera y guarda un reporte con MSE, R², ecuación del modelo y Residuales vs Fitted."
        )
        return self.residuals_button


    # Otros métodos
    def update_x_column(self):
        self.x_column = self.x_column_selector.currentText()
        self.check_if_ready_to_plot()

    def update_y_column(self):
        self.y_column = self.y_column_selector.currentText()
        self.check_if_ready_to_plot()

    def check_if_ready_to_plot(self):
        if self.x_column and self.y_column:
            self.plot_button.setEnabled(True)
        else:
            self.plot_button.setEnabled(False)
            self.start_algorithm_button.setEnabled(False)

    def load_csv_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo CSV", "", "CSV Files (*.csv)", options=options)
        if file_name:
            self.csv_data = pd.read_csv(file_name)

            # Obtener nombres de columnas
            columnas = self.csv_data.columns.tolist()

            # Poblar los selectores de columnas
            self.x_column_selector.clear()
            self.x_column_selector.addItems(columnas)
            self.x_column_selector.setEnabled(True)

            self.y_column_selector.clear()
            self.y_column_selector.addItems(columnas)
            self.y_column_selector.setEnabled(True)

            self.algorithm_selector.setEnabled(True)

            print(f"Archivo CSV cargado: {file_name}")

    def plot_csv_points(self):
        if self.csv_data is not None and self.x_column and self.y_column:
            x_data = self.csv_data[self.x_column]
            y_data = self.csv_data[self.y_column]
            self.canvas2.plot_points(x_data, y_data, title="Gráfico de Puntos CSV")
            self.canvas2.ax.set_xlabel(self.x_column)
            self.canvas2.ax.set_ylabel(self.y_column)
            self.canvas2.draw()
            # Habilitar el botón "Iniciar Algoritmo" después de graficar los puntos
            self.start_algorithm_button.setEnabled(True)

    def update_algorithm_type(self):
        self.algorithm_type = self.algorithm_selector.currentText()
        if self.algorithm_type == "Polinómico":
            self.grade_input.setEnabled(True)
        else:
            self.grade_input.setEnabled(False)

    def start_algorithm(self):
        if self.csv_data is not None and self.x_column and self.y_column:
            # Captura los valores de entrada
            num_generaciones_text = self.generation_input.text()
            if not num_generaciones_text:
                num_generaciones = 1000  # Valor por defecto
                self.generation_input.setText(str(num_generaciones))
            else:
                num_generaciones = int(num_generaciones_text)

            population_text = self.population_input.text()
            if not population_text:
                self.tamano_poblacion = 100
                self.population_input.setText(str(self.tamano_poblacion))
            else:
                self.tamano_poblacion = int(population_text)

            mutation_text = self.mutation_input.text()
            if not mutation_text:
                self.tasa_mutacion = 0.001
                self.mutation_input.setText(str(self.tasa_mutacion))
            else:
                self.tasa_mutacion = float(mutation_text)

            crossover_text = self.crossover_input.text()
            if not crossover_text:
                self.tasa_cruce = 0.1
                self.crossover_input.setText(str(self.tasa_cruce))
            else:
                self.tasa_cruce = float(crossover_text)

            # Obtener los datos seleccionados
            x_data = self.csv_data[self.x_column].values
            y_data = self.csv_data[self.y_column].values

            # Verifica y muestra los valores capturados
            print(f"Generaciones: {num_generaciones}, Población: {self.tamano_poblacion}, Mutación: {self.tasa_mutacion}, Cruce: {self.tasa_cruce}")

            # Inicializa el hilo del algoritmo según el tipo seleccionado
            if self.algorithm_type == "Logarítmico":
                print("Iniciando ajuste logarítmico...")
                self.algorithm_thread = GeneticAlgorithmLogarithmicThread(
                    x_data, y_data, num_generaciones, self.tamano_poblacion, self.tasa_mutacion, self.tasa_cruce
                )
            elif self.algorithm_type == "Polinómico":
                grado_text = self.grade_input.text()
                if not grado_text:
                    grado = 2  # Valor por defecto
                    self.grade_input.setText(str(grado))
                else:
                    grado = int(grado_text)
                print(f"Iniciando ajuste polinómico de grado {grado}...")
                self.algorithm_thread = GeneticAlgorithmPolynomialThread(
                    x_data, y_data, num_generaciones, grado, self.tamano_poblacion, self.tasa_mutacion, self.tasa_cruce
                )
            elif self.algorithm_type == "Exponencial":
                print("Iniciando ajuste exponencial...")
                self.algorithm_thread = GeneticAlgorithmExponentialThread(
                    x_data, y_data, num_generaciones, self.tamano_poblacion, self.tasa_mutacion, self.tasa_cruce
                )
            elif self.algorithm_type == "Trigonométrico":
                print("Iniciando ajuste trigonométrico...")
                self.algorithm_thread = GeneticAlgorithmTrigonometricThread(
                    x_data, y_data, num_generaciones, self.tamano_poblacion, self.tasa_mutacion, self.tasa_cruce
                )
            elif self.algorithm_type == "Logístico":
                print("Iniciando ajuste logístico...")
                self.algorithm_thread = GeneticAlgorithmLogisticThread(
                    x_data, y_data, num_generaciones, self.tamano_poblacion, self.tasa_mutacion, self.tasa_cruce
                )
            else:
                print("Por favor, selecciona un algoritmo válido.")
                return

            # Asegúrate de que el hilo esté listo antes de establecer is_running
            self.algorithm_thread.is_running = True
            self.algorithm_thread.update_plot.connect(self.update_plots)
            self.algorithm_thread.start()

            # Estado UI / flags
            self.algorithm_running = True
            self.residuals_button.setEnabled(False)

            # Cuando el hilo termine (por generaciones o por stop), habilitar análisis
            self.algorithm_thread.finished.connect(self.on_algorithm_finished)

            # Actualiza botones
            self.stop_algorithm_button.setEnabled(True)
            self.start_algorithm_button.setEnabled(False)

    def on_algorithm_finished(self):
        # Se ejecuta cuando el QThread termina (por stop o por fin de generaciones)
        self.algorithm_running = False
        self.stop_algorithm_button.setEnabled(False)
        self.start_algorithm_button.setEnabled(True)

        # Solo habilitar si hay resultados guardados
        self.residuals_button.setEnabled(self.last_run is not None)
        print("Algoritmo finalizado.")

    def _build_model_plain(self, genes):
        # Versión en texto simple (mejor para exportar a imagen)
        try:
            if self.algorithm_type == "Logarítmico":
                a, b, c, d = genes
                return f"y = {a:.6g} * log({b:.6g} * x + {c:.6g}) + {d:.6g}"
            if self.algorithm_type == "Polinómico":
                grado = len(genes) - 1
                parts = []
                for i, coef in enumerate(genes[:-1]):
                    power = grado - i
                    parts.append(f"{coef:.6g}*x^{power}")
                parts.append(f"{genes[-1]:.6g}")
                return "y = " + " + ".join(parts)
            if self.algorithm_type == "Exponencial":
                a, b, c = genes
                return f"y = {a:.6g} * exp({b:.6g} * x) + {c:.6g}"
            if self.algorithm_type == "Trigonométrico":
                a, b, c, d, e = genes
                return f"y = {a:.6g}*sin({b:.6g}*x) + {c:.6g}*cos({d:.6g}*x) + {e:.6g}"
            if self.algorithm_type == "Logístico":
                L, k, x0 = genes
                return f"y = {L:.6g} / (1 + exp(-{k:.6g}*(x - {x0:.6g})))"
        except Exception:
            pass
        return "Modelo no disponible"

    def _build_model_latex(self, genes):
        """Construye una representación tipo LaTeX (MathText de Matplotlib) del modelo."""
        try:
            if self.algorithm_type == "Logarítmico":
                a, b, c, d = genes
                return rf"$y = {a:.6g}\,\log\left({b:.6g}\,x + {c:.6g}\right) + {d:.6g}$"
            if self.algorithm_type == "Polinómico":
                grado = len(genes) - 1
                terms = []
                for i, coef in enumerate(genes):
                    power = grado - i
                    if power > 1:
                        base = rf"{abs(coef):.6g}\,x^{{{power}}}"
                    elif power == 1:
                        base = rf"{abs(coef):.6g}\,x"
                    else:
                        base = rf"{abs(coef):.6g}"

                    if i == 0:
                        sign = "-" if coef < 0 else ""
                        terms.append(sign + base)
                    else:
                        terms.append(("- " if coef < 0 else "+ ") + base)
                return r"$y = " + " ".join(terms) + r"$"
            if self.algorithm_type == "Exponencial":
                a, b, c = genes
                return rf"$y = {a:.6g}\,e^{{{b:.6g}x}} + {c:.6g}$"
            if self.algorithm_type == "Trigonométrico":
                a, b, c, d, e = genes
                return rf"$y = {a:.6g}\,\sin\left({b:.6g}x\right) + {c:.6g}\,\cos\left({d:.6g}x\right) + {e:.6g}$"
            if self.algorithm_type == "Logístico":
                L, k, x0 = genes
                return rf"$y = \frac{{{L:.6g}}}{{1 + e^{{-{k:.6g}(x - {x0:.6g})}}}}$"
        except Exception:
            pass
        return r"$\mathrm{Modelo\ no\ disponible}$"

    def export_residuals_report(self):
        # Genera un reporte (una sola imagen) con:
        # - Error (MSE) vs generaciones
        # - R²
        # - Función final
        # - Residuales vs Fitted
        if self.last_run is None:
            QMessageBox.warning(self, "Sin resultados", "Primero ejecuta el algoritmo para generar resultados.")
            return

        suggested = f"reporte_residuales_{self.algorithm_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar reporte de residuales",
            suggested,
            "PNG (*.png);;PDF (*.pdf)"
        )
        if not file_path:
            return

        x = self.last_run["x"]
        y_real = self.last_run["y_real"]
        y_pred = self.last_run["y_pred"]
        historial = self.last_run["historial"]
        r2 = float(self.last_run["r2"])
        mse = float(self.last_run["mse"])
        model_latex = self.last_run.get("model_latex")
        if not model_latex:
            # Fallback simple si por alguna razón no hay LaTeX construido
            mp = self.last_run.get("model_plain", "Modelo no disponible")
            model_latex = "$" + mp.replace("*", "\\cdot ") + "$"

        resid = y_real - y_pred

        fig = Figure(figsize=(12, 8), dpi=150)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(historial)
        ax1.set_title("Error (MSE) vs Generación")
        ax1.set_xlabel("Generación")
        ax1.set_ylabel("MSE")
        ax1.text(
            0.98, 0.98,
            f"R² = {r2:.4f}\nMSE = {mse:.6g}",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(x, y_real, label="Datos", color="blue")
        order = np.argsort(x)
        ax2.plot(
            x[order],
            y_pred[order],
            label="Fitted",
            color="lime",
            linewidth=3,
        )
        ax2.set_title("Datos vs Ajuste")
        ax2.set_xlabel(self.x_column)
        ax2.set_ylabel(self.y_column)
        ax2.legend()

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(y_pred, resid)
        ax3.axhline(0)
        ax3.set_title("Residuales vs Fitted")
        ax3.set_xlabel("Fitted")
        ax3.set_ylabel("Residual")

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        stats_block = "\n".join([
            r"$\mathbf{Funci\'on\ conseguida:}$",
            model_latex,
            "",
            rf"$R^2 = {r2:.4f}$",
            rf"$\mathrm{{MSE}} = {mse:.6g}$",
            rf"$N = {len(x)}$",
        ])
        ax4.text(0.0, 1.0, stats_block, va="top", fontsize=12)

        fig.tight_layout()
        fig.savefig(file_path, bbox_inches="tight")
        QMessageBox.information(self, "Reporte guardado", f"Reporte guardado en:\n{file_path}")

    def stop_algorithm(self):
        if self.algorithm_thread is not None:
            # Solicita detener el hilo (se cerrará cuando termine la iteración actual)
            self.algorithm_thread.stop()
            self.stop_algorithm_button.setEnabled(False)
            self.start_algorithm_button.setEnabled(False)
            self.residuals_button.setEnabled(False)
            print("Deteniendo algoritmo...")

    def update_plots(self, x_data, y_pred, historial_aptitud, y_restantes, mejor_individuo):
        # Limpiar y actualizar la gráfica de aptitud
        self.canvas1.ax.clear()
        self.canvas1.ax.set_title("Aptitud (MSE)", fontsize=14)
        self.canvas1.ax.set_xlabel("Generaciones", fontsize=12)
        self.canvas1.ax.plot(historial_aptitud)

        # Obtener los valores reales de y (datos originales)
        y_real = self.csv_data[self.y_column].values

        # Convertir y_pred a un arreglo numpy para facilitar cálculos
        y_pred_np = np.array(y_pred)

        # Reemplazar valores extremadamente altos con la media de y_real
        y_pred_np = np.where(y_pred_np > 1e5, np.mean(y_real), y_pred_np)

        # Calcular el R^2
        ss_res = np.sum((y_real - y_pred_np) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Ajustar R^2 si es negativo
        if r2 < 0:
            r2 = 0

        # Texto con el R^2 en la gráfica de aptitud
        texto_r2 = r"$R^2 = {:.2f}\%$".format(r2 * 100)
        self.canvas1.ax.text(0.95, 0.95, texto_r2, transform=self.canvas1.ax.transAxes, fontsize=14,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Dibujar la gráfica de aptitud con el R^2
        self.canvas1.draw()

        # Limpiar y actualizar la gráfica de predicciones
        self.canvas2.ax.clear()
        self.canvas2.ax.scatter(x_data, y_real, label='Datos originales', color='blue')

        # Graficar el resto de los individuos sin etiqueta
        for y_rest in y_restantes:
            y_rest_np = np.array(y_rest)
            y_rest_np = np.where(y_rest_np > 1e5, np.mean(y_real), y_rest_np)
            self.canvas2.ax.plot(x_data, y_rest_np, color='gray', alpha=0.25, label='_nolegend_')

        # Graficar el mejor individuo
        y_pred_plot = y_pred_np.tolist()
        self.canvas2.ax.plot(x_data, y_pred_plot, color='lime', linewidth=2.5, label='Fitted')

        # Mostrar la ecuación del modelo según el tipo de algoritmo
        if self.algorithm_type == "Logarítmico":
            texto_modelo = r"$y = {:.2f} \cdot \log({:.2f}x + {:.2f}) + {:.2f}$".format(
                mejor_individuo.genes[0], mejor_individuo.genes[1], mejor_individuo.genes[2], mejor_individuo.genes[3])
        elif self.algorithm_type == "Polinómico":
            grado = len(mejor_individuo.genes) - 1
            coeficientes = mejor_individuo.genes
            ecuacion = "y = " + " + ".join([f"{coeficientes[i]:.2f}x^{grado - i}" for i in range(grado)]) + f" + {coeficientes[-1]:.2f}"
            texto_modelo = ecuacion
        elif self.algorithm_type == "Exponencial":
            texto_modelo = r"$y = {:.2f} \cdot e^{{{:.2f}x}} + {:.2f}$".format(
                mejor_individuo.genes[0], mejor_individuo.genes[1], mejor_individuo.genes[2])
        elif self.algorithm_type == "Trigonométrico":
            texto_modelo = r"$y = {:.2f} \cdot \sin({:.2f}x) + {:.2f} \cdot \cos({:.2f}x) + {:.2f}$".format(
                mejor_individuo.genes[0], mejor_individuo.genes[1], mejor_individuo.genes[2], mejor_individuo.genes[3], mejor_individuo.genes[4])
        elif self.algorithm_type == "Logístico":
            texto_modelo = r"$y = \frac{{{:.2f}}}{{1 + e^{{-{:.2f}(x - {:.2f})}}}}$".format(
                mejor_individuo.genes[0], mejor_individuo.genes[1], mejor_individuo.genes[2])
        else:
            texto_modelo = ""

        # Mostrar la ecuación del modelo en la gráfica de predicciones
        self.canvas2.ax.text(0.05, 0.95, texto_modelo, transform=self.canvas2.ax.transAxes, fontsize=12,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Aplicar los límites calculados para el eje Y
        self.canvas2.ax.set_ylim([min(y_real) - 0.2 * (max(y_real) - min(y_real)),
                                   max(y_real) + 0.2 * (max(y_real) - min(y_real))])

        # Etiquetas de los ejes
        self.canvas2.ax.set_xlabel(self.x_column, fontsize=12)
        self.canvas2.ax.set_ylabel(self.y_column, fontsize=12)

        # Añadir la leyenda y dibujar la gráfica de predicciones
        self.canvas2.ax.legend()
        self.canvas2.draw()

        # Guardar últimos resultados (para exportar reporte sin mostrar más gráficos en la UI)
        try:
            self.last_run = {
                "x": np.array(x_data, dtype=float),
                "y_real": np.array(y_real, dtype=float),
                "y_pred": np.array(y_pred_np, dtype=float),
                "historial": np.array(historial_aptitud, dtype=float),
                "r2": float(r2),
                "mse": float(historial_aptitud[-1]) if len(historial_aptitud) else float("nan"),
                "genes": list(mejor_individuo.genes),
                "model_plain": self._build_model_plain(mejor_individuo.genes),
                "model_latex": self._build_model_latex(mejor_individuo.genes),
            }
        except Exception:
            self.last_run = None

    def button_style(self):
        return """
        QPushButton {
            background-color: #007ACC;
            color: white;
            font-size: 14px;
            padding: 8px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #005999;
        }
        """

    def stop_button_style(self):
        return """
        QPushButton {
            background-color: #ff4d4d;
            color: white;
            font-size: 14px;
            padding: 8px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #ff6666;
        }
        """


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Aplicar estilos globales
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QLabel {
            font-size: 14px;
            color: #333;
        }
        QLineEdit {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 14px;
        }
        QComboBox {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 14px;
        }
        QGroupBox {
            font-size: 14px;
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
