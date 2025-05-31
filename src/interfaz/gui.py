"""
Modulo para la interfaz grafica del sistema de mineria de datos para trading
"""

import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Importar configuracion
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Importar modulos del proyecto
from src.extraccion_datos.extractor import Extractor
from src.preprocesamiento.preprocesador import Preprocesador
from src.modelos.clustering import ModeloClustering
from src.modelos.anomalias import ModeloAnomalias
from src.modelos.reglas_asociacion import MineroReglas
from src.evaluacion.evaluador import Evaluador
from src.utils.utilidades import Utilidades

class InterfazTrading:
    """
    Clase principal para la interfaz grafica del sistema de trading
    """
    
    def __init__(self, master, datos=None):
        """
        Inicializa la interfaz grafica
        
        Args:
            master: Ventana principal de Tkinter
            datos (dict): Diccionario con datos precargados (opcional)
        """
        self.logger = logging.getLogger(__name__)
        self.master = master
        self.datos = datos or {}
        self.configurar_ventana()
        self.crear_widgets()
        self.inicializar_datos()
        
    def configurar_ventana(self):
        """Configura la ventana principal"""
        # Establecer titulo y tamaño
        self.master.title(config.GUI_CONFIG["ventana_titulo"])
        self.master.geometry(f"{config.GUI_CONFIG['ventana_ancho']}x{config.GUI_CONFIG['ventana_alto']}")
        
        # Aplicar tema
        style = ttk.Style()
        if config.GUI_CONFIG["tema"] == "oscuro":
            # Tema oscuro
            self.master.configure(bg=config.GUI_CONFIG["colores"]["fondo"])
            style.theme_use("clam")
            style.configure(".", 
                background=config.GUI_CONFIG["colores"]["fondo"],
                foreground=config.GUI_CONFIG["colores"]["texto"]
            )
        else:
            # Tema claro (predeterminado)
            style.theme_use("clam")
        
        # Configurar estilo de botones y widgets
        style.configure(
            "TButton", 
            padding=6, 
            relief="flat",
            background=config.GUI_CONFIG["colores"]["primario"],
            foreground="white"
        )
        
        style.map(
            "TButton",
            background=[("active", config.GUI_CONFIG["colores"]["secundario"])]
        )
        
        style.configure(
            "Accent.TButton", 
            padding=6, 
            relief="flat",
            background=config.GUI_CONFIG["colores"]["secundario"],
            foreground="white"
        )
        
        style.map(
            "Accent.TButton",
            background=[("active", config.GUI_CONFIG["colores"]["primario"])]
        )
        
        # Permitir que la ventana se redimensione
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
    
    def crear_widgets(self):
        """Crea los widgets de la interfaz"""
        # Contenedor principal
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)  # El area de graficos es expansible
        
        # Panel superior - Controles
        self.controles_frame = ttk.Frame(self.main_frame)
        self.controles_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        # Botones principales
        self.btn_cargar_datos = ttk.Button(
            self.controles_frame, 
            text="Cargar Datos", 
            command=self.cargar_datos
        )
        self.btn_cargar_datos.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_procesar_datos = ttk.Button(
            self.controles_frame, 
            text="Procesar Datos", 
            command=self.procesar_datos
        )
        self.btn_procesar_datos.grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_ejecutar_modelos = ttk.Button(
            self.controles_frame, 
            text="Ejecutar Modelos", 
            command=self.ejecutar_modelos,
            style="Accent.TButton"
        )
        self.btn_ejecutar_modelos.grid(row=0, column=2, padx=5, pady=5)
        
        self.btn_generar_reporte = ttk.Button(
            self.controles_frame, 
            text="Generar Reporte", 
            command=self.generar_reporte
        )
        self.btn_generar_reporte.grid(row=0, column=3, padx=5, pady=5)
        
        # Panel de opciones de visualizacion
        self.opciones_frame = ttk.LabelFrame(self.controles_frame, text="Opciones de Visualización")
        self.opciones_frame.grid(row=0, column=4, padx=10, pady=5, sticky="e")
        
        self.modo_visualizacion = tk.StringVar(value="precios")
        
        self.rb_precios = ttk.Radiobutton(
            self.opciones_frame, 
            text="Precios", 
            variable=self.modo_visualizacion, 
            value="precios",
            command=self.actualizar_grafico
        )
        self.rb_precios.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        
        self.rb_clusters = ttk.Radiobutton(
            self.opciones_frame, 
            text="Clusters", 
            variable=self.modo_visualizacion, 
            value="clusters",
            command=self.actualizar_grafico
        )
        self.rb_clusters.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        self.rb_anomalias = ttk.Radiobutton(
            self.opciones_frame, 
            text="Anomalías", 
            variable=self.modo_visualizacion, 
            value="anomalias",
            command=self.actualizar_grafico
        )
        self.rb_anomalias.grid(row=0, column=2, padx=5, pady=2, sticky="w")
        
        # Panel central - Area de graficos
        self.grafico_frame = ttk.Frame(self.main_frame)
        self.grafico_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        self.grafico_frame.columnconfigure(0, weight=1)
        self.grafico_frame.rowconfigure(0, weight=1)
        
        # Figura inicial
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.grafico_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")
        
        # Panel inferior - Pestañas para diferentes vistas
        self.tabs = ttk.Notebook(self.main_frame)
        self.tabs.grid(row=2, column=0, sticky="ew", pady=5)
        
        # Pestaña de Reglas
        self.tab_reglas = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_reglas, text="Reglas de Asociación")
        
        # Área de texto para reglas
        self.reglas_text = ScrolledText(self.tab_reglas, height=8)
        self.reglas_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Pestaña de Anomalías
        self.tab_anomalias = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_anomalias, text="Anomalías Detectadas")
        
        # Tabla de anomalías
        self.anomalias_frame = ttk.Frame(self.tab_anomalias)
        self.anomalias_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.anomalias_tree = ttk.Treeview(self.anomalias_frame, columns=("fecha", "precio", "retorno", "prob"))
        self.anomalias_tree.heading("fecha", text="Fecha")
        self.anomalias_tree.heading("precio", text="Precio")
        self.anomalias_tree.heading("retorno", text="Retorno %")
        self.anomalias_tree.heading("prob", text="Probabilidad")
        self.anomalias_tree.column("#0", width=0, stretch=tk.NO)
        self.anomalias_tree.column("fecha", width=150, anchor=tk.CENTER)
        self.anomalias_tree.column("precio", width=150, anchor=tk.CENTER)
        self.anomalias_tree.column("retorno", width=150, anchor=tk.CENTER)
        self.anomalias_tree.column("prob", width=150, anchor=tk.CENTER)
        self.anomalias_tree.pack(fill="both", expand=True)
        
        # Pestaña de Evaluación
        self.tab_evaluacion = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_evaluacion, text="Evaluación de Modelos")
        
        # Área de texto para evaluación
        self.evaluacion_text = ScrolledText(self.tab_evaluacion, height=8)
        self.evaluacion_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Barra de estado
        self.status_bar = ttk.Label(
            self.main_frame, 
            text="Sistema inicializado. Listo para cargar datos.", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.grid(row=3, column=0, sticky="ew")
    
    def inicializar_datos(self):
        """Inicializa los datos si se proporcionaron en el constructor"""
        if self.datos:
            self.mostrar_datos_cargados()
    
    def actualizar_estado(self, mensaje):
        """
        Actualiza la barra de estado
        
        Args:
            mensaje (str): Mensaje a mostrar
        """
        self.status_bar.config(text=mensaje)
        self.master.update_idletasks()
    
    def cargar_datos(self):
        """Carga datos desde archivos o extrae nuevos datos"""
        self.actualizar_estado("Cargando datos...")
        
        try:
            # Si ya están en memoria, usarlos
            if 'df_raw' in self.datos and self.datos['df_raw'] is not None:
                self.mostrar_datos_cargados()
                return
            
            # Mostrar diálogo para elegir: cargar existentes o extraer nuevos
            opciones = ["Cargar datos existentes", "Extraer nuevos datos", "Cancelar"]
            respuesta = messagebox.askquestion(
                "Cargar Datos", 
                "¿Desea cargar datos existentes o extraer nuevos datos?",
                type="yesnocancel"
            )
            
            if respuesta == "yes":  # Cargar existentes
                # Verificar si existe el archivo de datos crudos
                if os.path.exists(config.DATOS_CRUDOS):
                    extractor = Extractor(config.API_KEY, config.API_URL, config.API_RATE_LIMIT)
                    self.datos['df_raw'] = extractor.cargar_datos(config.DATOS_CRUDOS)
                    self.mostrar_datos_cargados()
                else:
                    messagebox.showerror(
                        "Error", 
                        f"No se encontró el archivo de datos crudos en {config.DATOS_CRUDOS}"
                    )
                    self.extraer_nuevos_datos()
                    
            elif respuesta == "no":  # Extraer nuevos
                self.extraer_nuevos_datos()
            
            # Si cancela, no hacer nada
                
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {str(e)}")
            messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            self.actualizar_estado("Error al cargar datos")
    
    def extraer_nuevos_datos(self):
        """Extrae nuevos datos de la API"""
        try:
            extractor = Extractor(config.API_KEY, config.API_URL, config.API_RATE_LIMIT)
            
            # Diálogo para confirmar extracción
            mensaje = f"Se extraerán datos de Bitcoin desde {config.FECHA_INICIO.strftime('%d/%m/%Y')} hasta {config.FECHA_FIN.strftime('%d/%m/%Y')}. Este proceso puede tardar varios minutos. ¿Desea continuar?"
            confirmar = messagebox.askyesno("Extraer datos", mensaje)
            
            if not confirmar:
                self.actualizar_estado("Extracción cancelada")
                return
            
            # Extraer datos
            self.actualizar_estado("Extrayendo datos de la API... (puede tardar varios minutos)")
            self.datos['df_raw'] = extractor.extraer_datos_historicos(
                config.FECHA_INICIO, 
                config.FECHA_FIN
            )
            
            # Guardar datos
            extractor.guardar_datos(self.datos['df_raw'], config.DATOS_CRUDOS)
            
            self.mostrar_datos_cargados()
            messagebox.showinfo("Éxito", f"Datos extraídos correctamente y guardados en {config.DATOS_CRUDOS}")
            
        except Exception as e:
            self.logger.error(f"Error al extraer datos: {str(e)}")
            messagebox.showerror("Error", f"Error al extraer datos: {str(e)}")
            self.actualizar_estado("Error al extraer datos")
    
    def mostrar_datos_cargados(self):
        """Muestra los datos cargados en la interfaz"""
        if 'df_raw' not in self.datos or self.datos['df_raw'] is None:
            return
            
        df = self.datos['df_raw']
        
        # Actualizar el gráfico
        self.actualizar_grafico()
        
        # Actualizar estado
        self.actualizar_estado(f"Datos cargados: {len(df)} registros desde {df.index.min().strftime('%d/%m/%Y')} hasta {df.index.max().strftime('%d/%m/%Y')}")
    
    def procesar_datos(self):
        """Procesa los datos cargados"""
        self.actualizar_estado("Procesando datos...")
        
        try:
            # Verificar que hay datos cargados
            if 'df_raw' not in self.datos or self.datos['df_raw'] is None:
                messagebox.showerror("Error", "No hay datos cargados para procesar")
                return
            
            # Procesar datos
            preprocesador = Preprocesador(config.PARAMETROS)
            self.datos['df_procesado'] = preprocesador.procesar(self.datos['df_raw'])
            self.datos['df_discretizado'] = preprocesador.discretizar(self.datos['df_procesado'])
            
            # Guardar datos procesados
            preprocesador.guardar_datos(self.datos['df_procesado'], config.DATOS_PROCESADOS)
            preprocesador.guardar_datos(self.datos['df_discretizado'], config.DATOS_DISCRETIZADOS)
            
            # Actualizar estado
            self.actualizar_estado("Datos procesados correctamente")
            
            # Actualizar gráfico
            self.actualizar_grafico()
            
            messagebox.showinfo("Éxito", "Datos procesados correctamente")
            
        except Exception as e:
            self.logger.error(f"Error al procesar datos: {str(e)}")
            messagebox.showerror("Error", f"Error al procesar datos: {str(e)}")
            self.actualizar_estado("Error al procesar datos")
    
    def ejecutar_modelos(self):
        """Ejecuta todos los modelos de minería de datos"""
        self.actualizar_estado("Ejecutando modelos...")
        
        try:
            # Verificar que hay datos procesados
            if 'df_procesado' not in self.datos or self.datos['df_procesado'] is None:
                # Intentar cargar datos procesados si existen
                if os.path.exists(config.DATOS_PROCESADOS):
                    preprocesador = Preprocesador(config.PARAMETROS)
                    self.datos['df_procesado'] = preprocesador.cargar_datos(config.DATOS_PROCESADOS)
                    self.datos['df_discretizado'] = preprocesador.cargar_datos(config.DATOS_DISCRETIZADOS) if os.path.exists(config.DATOS_DISCRETIZADOS) else None
                else:
                    messagebox.showerror("Error", "No hay datos procesados. Procese los datos primero.")
                    return
            
            # 1. Clustering
            self.actualizar_estado("Aplicando clustering...")
            modelo_clustering = ModeloClustering(
                n_clusters=config.PARAMETROS['clustering_num_clusters'],
                random_state=config.PARAMETROS['clustering_random_state']
            )
            self.datos['df_clusters'] = modelo_clustering.entrenar_y_predecir(self.datos['df_procesado'])
            modelo_clustering.guardar_modelo(config.MODELO_CLUSTERING)
            
            # 2. Detección de anomalías
            self.actualizar_estado("Detectando anomalías...")
            modelo_anomalias = ModeloAnomalias(
                n_estimators=config.PARAMETROS['rf_num_arboles'],
                max_depth=config.PARAMETROS['rf_max_depth'],
                random_state=config.PARAMETROS['rf_random_state']
            )
            self.datos['df_anomalias'] = modelo_anomalias.entrenar_y_predecir(self.datos['df_procesado'])
            modelo_anomalias.guardar_modelo(config.MODELO_ANOMALIAS)
            
            # 3. Reglas de asociación
            self.actualizar_estado("Extrayendo reglas de asociación...")
            minero_reglas = MineroReglas(
                soporte_min=config.PARAMETROS['reglas_soporte_min'],
                confianza_min=config.PARAMETROS['reglas_confianza_min'],
                lift_min=config.PARAMETROS['reglas_lift_min']
            )
            self.datos['reglas'] = minero_reglas.extraer_reglas(self.datos['df_discretizado'])
            minero_reglas.guardar_reglas(self.datos['reglas'], config.REGLAS_ASOCIACION)
            
            # 4. Evaluación de modelos
            self.actualizar_estado("Evaluando modelos...")
            evaluador = Evaluador(config.PARAMETROS)
            self.datos['resultados'] = evaluador.evaluar_modelos(
                self.datos['df_procesado'], 
                self.datos['df_clusters'], 
                self.datos['df_anomalias'], 
                self.datos['reglas']
            )
            
            # Actualizar interfaz con resultados
            self.actualizar_interfaz_con_resultados()
            
            # Actualizar estado
            self.actualizar_estado("Modelos ejecutados correctamente")
            
            messagebox.showinfo("Éxito", "Todos los modelos se ejecutaron correctamente")
            
        except Exception as e:
            self.logger.error(f"Error al ejecutar modelos: {str(e)}")
            messagebox.showerror("Error", f"Error al ejecutar modelos: {str(e)}")
            self.actualizar_estado("Error al ejecutar modelos")
    
    def actualizar_interfaz_con_resultados(self):
        """Actualiza la interfaz con los resultados de los modelos"""
        # Actualizar gráfico según el modo seleccionado
        self.actualizar_grafico()
        
        # Actualizar lista de anomalías
        self.mostrar_anomalias()
        
        # Actualizar reglas de asociación
        self.mostrar_reglas()
        
        # Actualizar evaluación
        self.mostrar_evaluacion()
    
    def mostrar_anomalias(self):
        """Muestra las anomalías detectadas en la tabla"""
        if 'df_anomalias' not in self.datos or self.datos['df_anomalias'] is None:
            return
        
        # Limpiar tabla
        for row in self.anomalias_tree.get_children():
            self.anomalias_tree.delete(row)
        
        # Filtrar solo anomalías
        if 'anomalia_pred' in self.datos['df_anomalias'].columns:
            anomalias = self.datos['df_anomalias'][self.datos['df_anomalias']['anomalia_pred'] == 1]
            
            if len(anomalias) > 0:
                # Ordenar por probabilidad descendente
                if 'prob_anomalia' in anomalias.columns:
                    anomalias = anomalias.sort_values('prob_anomalia', ascending=False)
                
                # Agregar a la tabla
                for idx, row in anomalias.iterrows():
                    fecha = idx.strftime('%Y-%m-%d')
                    precio = f"{row['close']:.2f}" if 'close' in row else "N/A"
                    retorno = f"{row['retorno']:.2f}" if 'retorno' in row else "N/A"
                    prob = f"{row['prob_anomalia']*100:.2f}%" if 'prob_anomalia' in row else "N/A"
                    
                    self.anomalias_tree.insert('', 'end', values=(fecha, precio, retorno, prob))
    
    def mostrar_reglas(self):
        """Muestra las reglas de asociación en el área de texto"""
        if 'reglas' not in self.datos or self.datos['reglas'] is None or self.datos['reglas'].empty:
            self.reglas_text.delete(1.0, tk.END)
            self.reglas_text.insert(tk.END, "No se han encontrado reglas de asociación.")
            return
        
        # Mostrar primeras 20 reglas ordenadas por lift
        reglas = self.datos['reglas'].sort_values('lift', ascending=False).head(20)
        
        texto = "TOP REGLAS DE ASOCIACIÓN (ordenadas por lift)\n"
        texto += "="*50 + "\n\n"
        
        for i, (_, regla) in enumerate(reglas.iterrows(), 1):
            texto += f"Regla #{i}:\n"
            texto += f"  SI {regla['antecedentes']}\n"
            texto += f"  ENTONCES {regla['consecuentes']}\n"
            texto += f"  (confianza: {regla['confianza']:.3f}, lift: {regla['lift']:.3f}, soporte: {regla['soporte']:.3f})\n\n"
        
        self.reglas_text.delete(1.0, tk.END)
        self.reglas_text.insert(tk.END, texto)
    
    def mostrar_evaluacion(self):
        """Muestra los resultados de evaluación en el área de texto"""
        if 'resultados' not in self.datos or self.datos['resultados'] is None:
            self.evaluacion_text.delete(1.0, tk.END)
            self.evaluacion_text.insert(tk.END, "No hay resultados de evaluación disponibles.")
            return
        
        # Crear evaluador para generar reporte
        evaluador = Evaluador(config.PARAMETROS)
        reporte = evaluador.generar_reporte(self.datos['resultados'])
        
        self.evaluacion_text.delete(1.0, tk.END)
        self.evaluacion_text.insert(tk.END, reporte)
    
    def actualizar_grafico(self):
        """Actualiza el gráfico según el modo seleccionado"""
        # Limpiar gráfico anterior
        self.ax.clear()
        
        # Obtener modo de visualización
        modo = self.modo_visualizacion.get()
        
        if modo == "precios":
            self.graficar_precios()
        elif modo == "clusters":
            self.graficar_clusters()
        elif modo == "anomalias":
            self.graficar_anomalias()
        
        # Actualizar canvas
        self.canvas.draw()
    
    def graficar_precios(self):
        """Grafica los precios de Bitcoin"""
        if 'df_raw' not in self.datos or self.datos['df_raw'] is None:
            self.ax.text(0.5, 0.5, "No hay datos disponibles", ha='center', va='center')
            return
        
        df = self.datos['df_raw']
        
        # Graficar precio de cierre
        self.ax.plot(df.index, df['close'], label='Precio de cierre', color='blue')
        
        # Graficar volumen como barras en eje secundario si está disponible
        if 'volume' in df.columns:
            ax2 = self.ax.twinx()
            ax2.bar(df.index, df['volume'], alpha=0.3, color='gray', label='Volumen')
            ax2.set_ylabel('Volumen', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
        
        # Añadir medias móviles si están disponibles en datos procesados
        if 'df_procesado' in self.datos and self.datos['df_procesado'] is not None:
            df_proc = self.datos['df_procesado']
            if 'sma_20' in df_proc.columns:
                self.ax.plot(df_proc.index, df_proc['sma_20'], label='SMA 20', color='orange', linestyle='--')
            if 'sma_50' in df_proc.columns:
                self.ax.plot(df_proc.index, df_proc['sma_50'], label='SMA 50', color='green', linestyle='--')
        
        # Configurar gráfico
        self.ax.set_title('Precio de Bitcoin')
        self.ax.set_xlabel('Fecha')
        self.ax.set_ylabel('Precio (USD)')
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
    
    def graficar_clusters(self):
        """Grafica los clusters identificados"""
        if 'df_clusters' not in self.datos or self.datos['df_clusters'] is None:
            self.ax.text(0.5, 0.5, "No hay datos de clusters disponibles", ha='center', va='center')
            return
        
        df = self.datos['df_clusters']
        
        # Verificar que tenemos la columna cluster
        if 'cluster' not in df.columns:
            self.ax.text(0.5, 0.5, "No se encontró la columna 'cluster' en los datos", ha='center', va='center')
            return
        
        # Colores para clusters
        colores = config.GUI_CONFIG["colores"]["cluster_colores"]
        
        # Graficar cada cluster por separado
        for cluster in sorted(df['cluster'].unique()):
            if cluster < 0:  # Saltar puntos no asignados
                continue
                
            mask = df['cluster'] == cluster
            self.ax.scatter(
                df.index[mask], 
                df['close'][mask] if 'close' in df.columns else df['retorno'][mask], 
                s=30, 
                c=colores[int(cluster) % len(colores)], 
                alpha=0.7, 
                label=f'Cluster {int(cluster)}'
            )
        
        # Línea de precio en segundo plano
        if 'close' in df.columns:
            self.ax.plot(df.index, df['close'], color='gray', alpha=0.3, zorder=0)
            self.ax.set_ylabel('Precio (USD)')
        else:
            self.ax.plot(df.index, df['retorno'], color='gray', alpha=0.3, zorder=0)
            self.ax.set_ylabel('Retorno (%)')
        
        # Configurar gráfico
        self.ax.set_title('Clustering de Bitcoin')
        self.ax.set_xlabel('Fecha')
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
    
    def graficar_anomalias(self):
        """Grafica las anomalías detectadas"""
        if 'df_anomalias' not in self.datos or self.datos['df_anomalias'] is None:
            self.ax.text(0.5, 0.5, "No hay datos de anomalías disponibles", ha='center', va='center')
            return
        
        df = self.datos['df_anomalias']
        
        # Verificar que tenemos las columnas necesarias
        if 'anomalia_pred' not in df.columns or 'close' not in df.columns:
            self.ax.text(0.5, 0.5, "No se encontraron columnas necesarias para visualizar anomalías", ha='center', va='center')
            return
        
        # Graficar precio
        self.ax.plot(df.index, df['close'], color='blue', alpha=0.6, zorder=1)
        
        # Destacar anomalías
        mask_anomalias = df['anomalia_pred'] == 1
        self.ax.scatter(
            df.index[mask_anomalias], 
            df['close'][mask_anomalias], 
            color='red', 
            s=80, 
            marker='o', 
            label='Anomalía', 
            zorder=2
        )
        
        # Configurar gráfico
        self.ax.set_title('Anomalías Detectadas en Bitcoin')
        self.ax.set_xlabel('Fecha')
        self.ax.set_ylabel('Precio (USD)')
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
    
    def generar_reporte(self):
        """Genera un reporte completo y lo guarda en archivo"""
        self.actualizar_estado("Generando reporte...")
        
        try:
            # Verificar que tenemos resultados para generar reporte
            if 'resultados' not in self.datos or self.datos['resultados'] is None:
                messagebox.showerror("Error", "No hay resultados para generar reporte")
                return
            
            # Solicitar ubicación para guardar reporte
            fecha_actual = datetime.now().strftime("%Y%m%d_%H%M")
            nombre_reporte = f"reporte_bitcoin_{fecha_actual}.txt"
            ruta_reporte = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
                initialfile=nombre_reporte,
                title="Guardar reporte"
            )
            
            if not ruta_reporte:  # Si el usuario cancela
                self.actualizar_estado("Generación de reporte cancelada")
                return
            
            # Crear evaluador para generar reporte
            evaluador = Evaluador(config.PARAMETROS)
            evaluador.generar_reporte(self.datos['resultados'], ruta_reporte)
            
            # Guardar gráficos
            ruta_base = os.path.dirname(ruta_reporte)
            
            # Guardar gráfico de precios
            if 'df_raw' in self.datos and self.datos['df_raw'] is not None:
                util = Utilidades()
                fig_precios = util.crear_grafico_precios(
                    self.datos['df_raw'], 
                    titulo='Precio Bitcoin',
                    guardar_como=os.path.join(ruta_base, f"grafico_precios_{fecha_actual}.png")
                )
            
            # Guardar gráfico de clusters
            if 'df_clusters' in self.datos and self.datos['df_clusters'] is not None:
                fig_clusters = util.crear_grafico_clusters(
                    self.datos['df_clusters'],
                    titulo='Clustering de Bitcoin',
                    guardar_como=os.path.join(ruta_base, f"grafico_clusters_{fecha_actual}.png")
                )
            
            # Guardar gráfico de anomalías
            if 'df_anomalias' in self.datos and self.datos['df_anomalias'] is not None:
                fig_anomalias = util.crear_grafico_anomalias(
                    self.datos['df_anomalias'],
                    titulo='Anomalías Detectadas',
                    guardar_como=os.path.join(ruta_base, f"grafico_anomalias_{fecha_actual}.png")
                )
            
            messagebox.showinfo("Éxito", f"Reporte generado y guardado en:\n{ruta_reporte}")
            self.actualizar_estado("Reporte generado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error al generar reporte: {str(e)}")
            messagebox.showerror("Error", f"Error al generar reporte: {str(e)}")
            self.actualizar_estado("Error al generar reporte")


def iniciar_interfaz(datos=None):
    """
    Inicia la interfaz grafica
    
    Args:
        datos (dict): Diccionario con datos precargados
    """
    # Crear ventana principal
    root = tk.Tk()
    
    # Crear interfaz
    app = InterfazTrading(root, datos)
    
    # Iniciar bucle principal
    root.mainloop()


if __name__ == "__main__":
    iniciar_interfaz()
