import matplotlib.pyplot as plt
from cycler import cycler

def style(figsize=(8, 5), show_minor_ticks=True, dpi= 300):

    """
    Configura el estilo de las gráficas según las preferencias proporcionadas.
    
    Parámetros:
    - figsize: Tamaño de la figura (ancho, alto).
    - show_minor_ticks: Si es True, muestra los ticks menores.
    - show_grid: Si es True, muestra la cuadrícula.
    """
    
    # Configuración de la figura
    plt.figure(figsize=figsize , dpi = dpi)

    # Configuración de texto y fuentes
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \boldmath'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    plt.rcParams['font.size'] = 15

    # Configuración de los ejes x e y
    for axis in ['xtick', 'ytick']:
        plt.rcParams[f'{axis}.direction'] = 'in'
        plt.rcParams[f'{axis}.major.size'] = 4
        plt.rcParams[f'{axis}.major.width'] = 1
        plt.rcParams[f'{axis}.minor.size'] = 2
        plt.rcParams[f'{axis}.minor.width'] = 1
        plt.rcParams[f'{axis}.labelsize'] = 12
        plt.rcParams[f'{axis}.major.pad'] = 9

    plt.rcParams['ytick.right'] = True

    # Configuración de títulos y etiquetas
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 15

    # Configuración de líneas y márgenes
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.solid_capstyle'] = 'round'
    plt.rcParams['axes.xmargin'] = 0.02
    plt.rcParams['axes.ymargin'] = 0.02

    # Configuración de la leyenda
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['legend.title_fontsize'] = 15
    plt.rcParams['legend.frameon'] = False

    # Configuración del ciclo de colores
    plt.rcParams['axes.prop_cycle'] = cycler('color', 
        ["#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#7F7F7F", "#9467BD"])

    # Configuración de los ticks
    if show_minor_ticks:
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['ytick.minor.visible'] = True