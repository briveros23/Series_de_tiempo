import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest


def plot_time_series(data, start_year, freq, detect_outliers=False):
    """
    Grafica una serie de tiempo con opción de detección de atípicos mediante Isolation Forest.
    
    Parámetros:
    - data: pd.Series -> Serie de datos.
    - start_year: int -> Año de inicio.
    - freq: str -> Frecuencia de la serie ('M', 'Q', 'A', 'W', etc.).
    - detect_outliers: bool -> Si es True, resalta los atípicos detectados.
    """
    
    # Determinar el primer día en función de la frecuencia
    start_date = f'{start_year}-01-01'
    
    # Ajustar la fecha inicial para frecuencia semanal
    if freq == 'W':
        start_date = pd.to_datetime(start_date) + pd.DateOffset(weeks=0)
    
    # Crear el índice de tiempo
    date_range = pd.date_range(start=start_date, periods=len(data), freq=freq)
    time_series = pd.Series(data.values, index=date_range)
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, linestyle='-', label='Serie de tiempo')
    
    if detect_outliers:
        model = IsolationForest(contamination=0.05, random_state=42)
        outliers = model.fit_predict(data.values.reshape(-1, 1))
        outlier_mask = outliers == -1

        
        plt.scatter(date_range[outlier_mask], data[outlier_mask], color='red', label='Atípicos', marker='o')
    
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.title('Gráfico de Serie de Tiempo')
    plt.legend()
    plt.grid(True)
    plt.show()
    if detect_outliers:
        return data[outlier_mask]


##  funcion para devolver una columna con los atipicos  para insertar en la base de datos 
def detect_outliers(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    outliers = model.fit_predict(data.values.reshape(-1, 1))
    outlier_mask = outliers == -1
    return outlier_mask



def plot_cross_correlation(x, y, max_lags=20, alpha=0.05):
    """
    Grafica la correlación cruzada entre dos series de tiempo con líneas de confianza.
    
    Parámetros:
    - x: pd.Series -> Primera serie de datos.
    - y: pd.Series -> Segunda serie de datos.
    - max_lags: int -> Número máximo de rezagos a considerar.
    - alpha: float -> Nivel de significancia para las bandas de confianza.
    """
    
    x = x.to_numpy() if isinstance(x, pd.Series) else np.asarray(x)
    y = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    
    lags = np.arange(-max_lags, max_lags + 1)
    ccf = [np.corrcoef(np.roll(x, lag)[max_lags:], y[max_lags:])[0, 1] for lag in lags]
    
    # Calcular intervalo de confianza
    conf_level = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(x))
    
    plt.figure(figsize=(10, 5))
    plt.stem(lags, ccf)
    plt.axhline(conf_level, color='red', linestyle='dashed', label=f'Confianza {100 * (1 - alpha):.1f}%')
    plt.axhline(-conf_level, color='red', linestyle='dashed')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel('Rezagos')
    plt.ylabel('Correlación')
    plt.title('Correlación Cruzada')
    plt.legend()
    plt.grid(True)
    plt.show()