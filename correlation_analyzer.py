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
    ax_corr.stem(lags, correlations) # Rimosso use_line_collection=True
    ax_corr.set_title(f'Cross-Correlazione Rendimenti', fontsize=12)
    ax_corr.set_xlabel(f'Lag (Periodi {selected_interval_label}) - Positivo: {ticker2_name} anticipa {ticker1_name}; Negativo: {ticker1_name} anticipa {ticker2_name}', fontsize=10)
    ax_corr.set_ylabel('Coefficiente di Correlazione', fontsize=10)
    ax_corr.grid(True, linestyle='--', alpha=0.6)
    ax_corr.axvline(0, color='red', linestyle=':', linewidth=1.5, label='Lag 0')
    ax_corr.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_corr.legend()
    plt.tight_layout()
    return fig_corr

def plot_data_trends(data_df: pd.DataFrame, identifier1_name: str, identifier2_name: str, fred_series_ids: list):
    """
    Genera un grafico a linee dell'andamento dei dati grezzi, adattandosi al tipo di dati.

    Args:
        data_df (pd.DataFrame): DataFrame contenente i dati grezzi dei due identificatori.
        identifier1_name (str): Nome del primo identificatore.
        identifier2_name (str): Nome del secondo identificatore.
        fred_series_ids (list): Lista degli ID delle serie FRED per identificare gli indicatori economici.

    Returns:
        matplotlib.figure.Figure: L'oggetto figura del grafico.
    """
    is_id1_fred = identifier1_name in fred_series_ids
    is_id2_fred = identifier2_name in fred_series_ids

    # Caso 1: Entrambi sono prezzi (non FRED)
    if not is_id1_fred and not is_id2_fred:
        fig_prices, ax_prices = plt.subplots(figsize=(8, 4))
        
        # Calcola l'aumento percentuale rispetto al valore iniziale
        initial_value_1 = data_df[identifier1_name].iloc[0]
        initial_value_2 = data_df[identifier2_name].iloc[0]

        if initial_value_1 == 0 or initial_value_2 == 0:
            # Gestisci il caso in cui il valore iniziale sia zero per evitare divisione per zero
            ax_prices.set_title(f'Andamento Dati {identifier1_name} vs {identifier2_name}\n(Impossibile calcolare aumento % per valore iniziale zero)', fontsize=12)
            ax_prices.plot(data_df.index, data_df[identifier1_name], label=identifier1_name, color='blue')
            ax_prices.plot(data_df.index, data_df[identifier2_name], label=identifier2_name, color='green')
            ax_prices.set_ylabel('Valore', fontsize=10)
        else:
            percentage_increase_1 = (data_df[identifier1_name] / initial_value_1 - 1) * 100
            percentage_increase_2 = (data_df[identifier2_name] / initial_value_2 - 1) * 100
            
            ax_prices.plot(data_df.index, percentage_increase_1, label=identifier1_name, color='blue')
            ax_prices.plot(data_df.index, percentage_increase_2, label=identifier2_name, color='green')
            ax_prices.set_title(f'Andamento Dati (Aumento Percentuale) {identifier1_name} vs {identifier2_name}', fontsize=12)
            ax_prices.set_ylabel('Aumento Percentuale (%)', fontsize=10)

        ax_prices.set_xlabel('Data', fontsize=10)
        ax_prices.grid(True, linestyle='--', alpha=0.6)
        ax_prices.legend()
        plt.tight_layout()
        return fig_prices

    # Caso 2: Un prezzo e un indicatore economico
    elif (not is_id1_fred and is_id2_fred) or (is_id1_fred and not is_id2_fred):
        fig_prices, ax_prices = plt.subplots(figsize=(8, 4))
        ax_prices_twin = ax_prices.twinx()

        # Determina quale è il prezzo e quale l'indicatore
        price_id = identifier1_name if not is_id1_fred else identifier2_name
        econ_id = identifier2_name if not is_id1_fred else identifier1_name

        ax_prices.plot(data_df.index, data_df[price_id], label=price_id, color='blue')
        ax_prices_twin.plot(data_df.index, data_df[econ_id], label=econ_id, color='green', linestyle='--')

        ax_prices.set_title(f'Andamento Dati {price_id} vs {econ_id}', fontsize=12)
        ax_prices.set_xlabel('Data', fontsize=10)
        ax_prices.set_ylabel(f'Valore {price_id}', color='blue', fontsize=10)
        ax_prices_twin.set_ylabel(f'Valore {econ_id}', color='green', fontsize=10)

        # Colora i tick labels degli assi per chiarezza
        ax_prices.tick_params(axis='y', labelcolor='blue')
        ax_prices_twin.tick_params(axis='y', labelcolor='green')

        ax_prices.grid(True, linestyle='--', alpha=0.6)
        # Combinare le leggende dai due assi
        lines, labels = ax_prices.get_legend_handles_labels()
        lines2, labels2 = ax_prices_twin.get_legend_handles_labels()
        ax_prices_twin.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        return fig_prices

    # Caso 3: Entrambi sono indicatori economici FRED
    else: # is_id1_fred and is_id2_fred
        fig_prices, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True) # Due subplots verticali, condividono asse X

        # Primo indicatore
        ax1.plot(data_df.index, data_df[identifier1_name], label=identifier1_name, color='blue')
        ax1.set_title(f'Andamento Dati {identifier1_name}', fontsize=12)
        ax1.set_ylabel(f'Valore {identifier1_name}', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # Secondo indicatore
        ax2.plot(data_df.index, data_df[identifier2_name], label=identifier2_name, color='green')
        ax2.set_title(f'Andamento Dati {identifier2_name}', fontsize=12)
        ax2.set_xlabel('Data', fontsize=10)
        ax2.set_ylabel(f'Valore {identifier2_name}', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        return fig_prices

