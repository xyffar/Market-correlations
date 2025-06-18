import yfinance as yf
import pandas as pd
import requests

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
        pd.Series or None: La serie dei prezzi se il download ha successo, altrimenti None.
        str or None: Messaggio di errore se si verifica un problema, altrimenti None.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty:
            return None, f"Errore: Impossibile scaricare i dati da Yahoo Finance per '{ticker}'. Controlla il simbolo o il periodo."

        # Controllo e appiattimento delle colonne multi-livello
        if isinstance(data.columns, pd.MultiIndex):
            # Prende il primo livello del MultiIndex come nome delle colonne
            # Esempio: ('Adj Close', 'TICKER') diventa 'Adj Close'
            data.columns = data.columns.get_level_values(0) # Corretto a 0 per appiattimento standard

        # Ritorna la serie dei prezzi
        # Assicurati che la colonna richiesta esista dopo l'appiattimento
        if price_type not in data.columns:
            return None, f"Errore: La colonna '{price_type}' non è disponibile per '{ticker}' dopo l'elaborazione dei dati. Prova a scegliere un altro tipo di prezzo."

        # Imposta il nome della serie con il tipo di prezzo scelto, per chiarezza
        series_to_return = data[price_type]
        series_to_return.name = ticker # Imposta il nome della serie al ticker stesso per il DataFrame combinato

        return series_to_return, None
    except Exception as e:
        return None, f"Errore durante il download o l'elaborazione da Yahoo Finance per '{ticker}': {e}"

def get_fred_data(series_id: str, start_date: str, end_date: str):
    """
    Scarica i dati storici da FRED (Federal Reserve Economic Data) per un dato series_id.
    I dati FRED hanno la loro frequenza nativa (es. mensile per UNRATE, giornaliero per DGS10).
    Non è necessario specificare un 'interval' qui, poiché FRED fornisce la frequenza predefinita.

    Args:
        series_id (str): L'ID della serie FRED (es. 'UNRATE', 'DGS10').
        start_date (str): Data di inizio (formato 'YYYY-MM-DD').
        end_date (str): Data di fine (formato 'YYYY-MM-DD').

    Returns:
        pd.Series or None: La serie dei dati FRED se il download ha successo, altrimenti None.
        str or None: Messaggio di errore se si verifica un problema, altrimenti None.
    """
    base_url = "https://fred.stlouisfed.org/series/"
    csv_url = f"{base_url}{series_id}/downloaddata/{series_id}.csv"

    try:
        # FRED restituisce un CSV che può essere letto direttamente da pandas
        df = pd.read_csv(csv_url,
                         index_col='DATE',
                         parse_dates=True)

        # Standardizza il nome della colonna dei valori. FRED a volte usa 'VALUE', a volte il series_id.
        if 'VALUE' in df.columns:
            series_data = df['VALUE']
        elif len(df.columns) == 1:
            series_data = df.iloc[:, 0] # Prende la prima (e probabilmente unica) colonna di dati
        else:
            return None, f"Errore: Formato dati FRED inatteso per '{series_id}'. Impossibile identificare la colonna dei valori."

        # Imposta il nome della serie per chiarezza nel plot
        series_data.name = series_id

        # FRED usa a volte '.' per i dati mancanti; converti in NaN e pulisci
        series_data = series_data.replace('.', pd.NA).dropna().astype(float)

        # Filtra per l'intervallo di date richiesto
        series_data = series_data.loc[start_date:end_date]

        if series_data.empty:
            return None, f"Errore: Nessun dato FRED disponibile per '{series_id}' nel periodo specificato."

        return series_data, None
    except requests.exceptions.RequestException as req_e:
        return None, f"Errore di rete durante il download da FRED per '{series_id}': {req_e}. Controlla la tua connessione internet."
    except Exception as e:
        return None, f"Errore durante l'elaborazione dei dati FRED per '{series_id}': {e}"

def get_data_for_identifier(identifier: str, start_date: str, end_date: str, yf_interval: str, yf_price_type: str = "Close"):
    """
    Funzione wrapper per scaricare dati, distinguendo tra Yahoo Finance e FRED.
    Per ora, gli ID serie FRED riconosciuti sono 'UNRATE', 'DGS10', 'DGS02', 'DGS30'.
    Qualsiasi altro identificatore sarà trattato come un ticker di Yahoo Finance.

    Args:
        identifier (str): Il simbolo del ticker di Yahoo Finance o l'ID della serie FRED.
        start_date (str): Data di inizio (formato 'YYYY-MM-DD').
        end_date (str): Data di fine (formato 'YYYY-MM-DD').
        yf_interval (str): Intervallo per i dati di Yahoo Finance (es. '1d').
        yf_price_type (str): Tipo di prezzo da usare per i dati di Yahoo Finance.

    Returns:
        pd.Series or None: La serie dei dati.
        str or None: Messaggio di errore.
    """
    fred_series_ids = ['UNRATE', 'DGS10', 'DGS02', 'DGS30',
                       'GDPC1', 'CPIAUCSL', 'FEDFUNDS', 'RSAFS', 'INDPRO',
                       'UMICHCSII', 'PERMIT', 'ICSA', 'AWHMAN', # Existing
                       'NEWORDER', 'IPMAN', 'ISRATIO', 'HOUST', 'TCU', 'DSPIC96'] # Newly added FRED IDs

    if identifier in fred_series_ids:
        # Se l'identificatore è un ID serie FRED
        return get_fred_data(identifier, start_date, end_date)
    else:
        # Altrimenti, assumi che sia un ticker di Yahoo Finance
        return get_yfinance_data(identifier, start_date, end_date, yf_interval, yf_price_type)

