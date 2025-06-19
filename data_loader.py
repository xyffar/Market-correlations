import yfinance as yf
import pandas as pd
import requests
import pandas_datareader.data as web # Importa pandas_datareader

def get_yfinance_data(ticker: str, start_date: str, end_date: str, interval: str, price_type: str = "Close"):
    """
    Scarica i dati storici da Yahoo Finance per un dato ticker.
    Gestisce la potenziale presenza di colonne multi-livello restituite da yfinance.

    Args:
        ticker (str): Il simbolo del ticker.
        start_date (str): Data di inizio (formato 'YYYY-MM-DD').
        end_date (str): Data di fine (formato 'YYYY-MM-DD').
        interval (str): Intervallo dei dati (es. '1d', '1wk').
        price_type (str): Tipo di prezzo da usare (default 'Adj Close').

    Returns:
        tuple: (pd.Series or None, str or None, str or None)
                Serie dei prezzi, frequenza inferita, messaggio di errore.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty:
            return None, None, f"Errore: Impossibile scaricare i dati da Yahoo Finance per '{ticker}'. Controlla il simbolo o il periodo."

        # Controllo e appiattimento delle colonne multi-livello
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Assicurati che la colonna richiesta esista dopo l'appiattimento
        if price_type not in data.columns:
            return None, None, f"Errore: La colonna '{price_type}' non è disponibile per '{ticker}' dopo l'elaborazione dei dati. Prova a scegliere un altro tipo di prezzo."

        series_to_return = data[price_type]
        series_to_return.name = ticker # Imposta il nome della serie al ticker stesso per il DataFrame combinato

        # Inferisci la frequenza dai dati scaricati.
        # Yahoo Finance ha intervalli specifici, quindi possiamo mappare.
        inferred_freq = pd.infer_freq(series_to_return.index)
        if inferred_freq is None: # A volte infer_freq non è perfetto, forziamo in base all'intervallo yf
            if interval == '1d': inferred_freq = 'D'
            elif interval == '1wk': inferred_freq = 'W'
            elif interval == '1mo': inferred_freq = 'M'

        return series_to_return, inferred_freq, None
    except Exception as e:
        return None, None, f"Errore durante il download o l'elaborazione da Yahoo Finance per '{ticker}': {e}"

def get_fred_data(series_id: str, start_date: str, end_date: str):
    """
    Scarica i dati storici da FRED (Federal Reserve Economic Data) per un dato series_id usando pandas_datareader.

    Args:
        series_id (str): L'ID della serie FRED (es. 'UNRATE', 'DGS10').
        start_date (str): Data di inizio (formato 'YYYY-MM-DD').
        end_date (str): Data di fine (formato 'YYYY-MM-DD').
        api_key (str): La tua chiave API FRED personale.

    Returns:
        tuple: (pd.Series or None, str or None, str or None)
                Serie dei dati FRED, frequenza inferita, messaggio di errore.
    """
    try:
        # pandas_datareader gestisce la data di inizio e fine direttamente
        series_data = web.DataReader(series_id, 'fred', start=start_date, end=end_date)

        if series_data is None or series_data.empty:
            return None, None, f"Errore: Nessun dato FRED disponibile per '{series_id}' nel periodo specificato o chiave API non valida."

        # DataReader restituisce un DataFrame con una colonna, la convertiamo in Series
        if isinstance(series_data, pd.DataFrame):
            series_data = series_data.iloc[:, 0]
        
        series_data.name = series_id # Imposta il nome della serie

        # Assicurati che l'indice sia di tipo datetime
        series_data.index = pd.to_datetime(series_data.index)
        
        inferred_freq = pd.infer_freq(series_data.index)
        return series_data, inferred_freq, None
    except Exception as e:
        return None, None, f"Errore durante il download o l'elaborazione da FRED per '{series_id}' (con API): {e}. Controlla la chiave API e l'ID serie."

def get_data_for_identifier(identifier: str, start_date: str, end_date: str, yf_interval: str, yf_price_type: str = "Close"):
    """
    Funzione wrapper per scaricare dati, distinguendo tra Yahoo Finance e FRED.

    Args:
        identifier (str): Il simbolo del ticker di Yahoo Finance o l'ID della serie FRED.
        start_date (str): Data di inizio (formato 'YYYY-MM-DD').
        end_date (str): Data di fine (formato 'YYYY-MM-DD').
        yf_interval (str): Intervallo per i dati di Yahoo Finance (es. '1d').
        yf_price_type (str): Tipo di prezzo da usare per i dati di Yahoo Finance.
        fred_api_key (str): La chiave API FRED, necessaria per i dati FRED.

    Returns:
        tuple: (pd.Series or None, str or None, str or None)
                Serie dei dati, frequenza inferita, messaggio di errore.
    """
    fred_series_ids = ['UNRATE', 'DGS10', 'DGS02', 'DGS30',
                        'GDPC1', 'CPIAUCSL', 'FEDFUNDS', 'RSAFS', 'INDPRO',
                        'UMICHCSII', 'PERMIT', 'ICSA', 'AWHMAN',
                        'NEWORDER', 'IPMAN', 'ISRATIO', 'HOUST', 'TCU', 'DSPIC96', 'CPILFESL']

    if identifier in fred_series_ids:
        return get_fred_data(identifier, start_date, end_date) # Passa la chiave API
    else:
        return get_yfinance_data(identifier, start_date, end_date, yf_interval, yf_price_type)
