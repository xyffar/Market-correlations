import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess_series_for_correlation(series: pd.Series):
    """
    Preprocessa una serie di dati per l'analisi di correlazione.
    Tenta di calcolare il cambiamento percentuale; se non significativo, usa la differenza.
    Questo aiuta a rendere le serie stazionarie.

    Args:
        series (pd.Series): La serie di dati da pre-processare.

    Returns:
        pd.Series: La serie pre-processata (rendimenti o differenze).
    """
    if series.empty:
        return pd.Series(dtype=float)

    # Tenta di calcolare il cambiamento percentuale (rendimenti)
    # È robusto per i prezzi azionari e i rendimenti obbligazionari
    processed_series = series.pct_change()
    if processed_series.abs().sum() == 0 or processed_series.isnull().all():
        # Se pct_change è tutto zero o NaN (es. per serie con valori molto piccoli o costanti),
        # prova a usare la differenza assoluta.
        processed_series = series.diff()

    return processed_series.dropna()

def calculate_cross_correlation(series1: pd.Series, series2: pd.Series, max_lags: int, selected_interval_label: str):
    """
    Calcola la cross-correlazione tra due serie di dati pre-processate.

    Args:
        series1 (pd.Series): La prima serie di dati.
        series2 (pd.Series): La seconda serie di dati.
        max_lags (int): Il numero massimo di lag da considerare.
        selected_interval_label (str): Etichetta per la granularità dei dati (per i grafici).

    Returns:
        tuple: (lags_range, cross_correlations, (best_lag, max_corr_value), error_message)
               error_message sarà None se non ci sono errori.
    """
    # Preprocessa le serie per la correlazione (es. trasformazione in rendimenti)
    processed_series1 = preprocess_series_for_correlation(series1)
    processed_series2 = preprocess_series_for_correlation(series2)

    if processed_series1.empty or processed_series2.empty:
        return None, None, None, "Errore: Dati insufficienti dopo la pre-elaborazione per la correlazione."

    # Allinea le serie pre-processate per avere un indice temporale comune
    aligned_data = pd.DataFrame({
        'series1': processed_series1,
        'series2': processed_series2
    }).dropna()

    if aligned_data.empty:
        return None, None, None, "Errore: Nessun dato comune trovato tra le serie pre-processate per la correlazione."

    series1_aligned = aligned_data['series1']
    series2_aligned = aligned_data['series2']

    lags_range = np.arange(-max_lags, max_lags + 1)
    cross_correlations = []

    # Calcolo della cross-correlazione usando lo shift di pandas
    for lag in lags_range:
        if lag >= 0:
            # Correlazione tra series1[t] e series2[t-lag]
            # Se lag > 0, series2 si verifica prima (anticipa) series1
            corr = series1_aligned.corr(series2_aligned.shift(lag))
        else:
            # Correlazione tra series1[t] e series2[t-|lag|]
            # Se lag < 0, series1 si verifica prima (anticipa) series2
            corr = series1_aligned.shift(-lag).corr(series2_aligned)
        cross_correlations.append(corr)

    cross_correlations = np.array(cross_correlations)

    # Trova il lag con la massima correlazione in valore assoluto
    max_corr_abs_idx = np.argmax(np.abs(cross_correlations))
    best_lag = lags_range[max_corr_abs_idx]
    max_corr_value = cross_correlations[max_corr_abs_idx]

    return lags_range, cross_correlations, (best_lag, max_corr_value), None


def plot_cross_correlation(lags: np.ndarray, correlations: np.ndarray, ticker1_name: str, ticker2_name: str, selected_interval_label: str):
    """
    Genera un grafico a stelo della cross-correlazione.

    Args:
        lags (np.ndarray): Array dei lag.
        correlations (np.ndarray): Array dei coefficienti di cross-correlazione.
        ticker1_name (str): Nome del primo ticker/serie.
        ticker2_name (str): Nome del secondo ticker/serie.
        selected_interval_label (str): Etichetta per la granularità (per l'asse X).

    Returns:
        matplotlib.figure.Figure: L'oggetto figura del grafico.
    """
    fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
    ax_corr.stem(lags, correlations, use_line_collection=True)
    ax_corr.set_title(f'Cross-Correlazione Rendimenti', fontsize=12)
    ax_corr.set_xlabel(f'Lag (Periodi {selected_interval_label}) - Positivo: {ticker2_name} anticipa {ticker1_name}; Negativo: {ticker1_name} anticipa {ticker2_name}', fontsize=10)
    ax_corr.set_ylabel('Coefficiente di Correlazione', fontsize=10)
    ax_corr.grid(True, linestyle='--', alpha=0.6)
    ax_corr.axvline(0, color='red', linestyle=':', linewidth=1.5, label='Lag 0')
    ax_corr.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_corr.legend()
    plt.tight_layout()
    return fig_corr

def plot_data_trends(data_df: pd.DataFrame, identifier1_name: str, identifier2_name: str):
    """
    Genera un grafico a linee dell'andamento dei dati grezzi.

    Args:
        data_df (pd.DataFrame): DataFrame contenente i dati grezzi dei due identificatori.
        identifier1_name (str): Nome del primo identificatore.
        identifier2_name (str): Nome del secondo identificatore.

    Returns:
        matplotlib.figure.Figure: L'oggetto figura del grafico.
    """
    fig_prices, ax_prices = plt.subplots(figsize=(8, 4))
    
    # I nomi delle colonne nel DataFrame 'data_df' sono già gli identificatori stessi
    # (es. 'SPY', 'UNRATE'), grazie a come le serie vengono nominate in data_loader.py.
    # Non è più necessario verificare la presenza di '_Adj Close'.
    col1_name_for_plot = identifier1_name
    col2_name_for_plot = identifier2_name

    ax_prices.plot(data_df.index, data_df[col1_name_for_plot], label=identifier1_name, color='blue')
    ax_prices.plot(data_df.index, data_df[col2_name_for_plot], label=identifier2_name, color='green')
    ax_prices.set_title(f'Andamento Dati {identifier1_name} vs {identifier2_name}', fontsize=12)
    ax_prices.set_xlabel('Data', fontsize=10)
    ax_prices.set_ylabel('Valore', fontsize=10) # Etichetta più generica per adattarsi a prezzi/tassi
    ax_prices.grid(True, linestyle='--', alpha=0.6)
    ax_prices.legend()
    plt.tight_layout()
    return fig_prices

