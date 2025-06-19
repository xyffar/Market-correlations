import streamlit as st
import pandas as pd
import itertools # Per generare combinazioni di identificatori
import matplotlib.pyplot as plt # Assicurati che sia importato qui per plt.close()

# Importa le funzioni dai file di utilità
from data_loader import get_data_for_identifier
from correlation_analyzer import calculate_cross_correlation, plot_cross_correlation, plot_data_trends

# Imposta il titolo dell'applicazione Streamlit
st.set_page_config(
    page_title="Cross-Correlazione Finanziaria",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Analisi di Cross-Correlazione Finanziaria")
st.markdown("Valuta se un asset anticipa i movimenti di un altro tramite l'analisi dei rendimenti. Supporta ticker Yahoo Finance e serie dati FRED.")

# --- Inizializzazione dello stato per il popup ---
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = False

# Funzione per aprire il popup
def open_popup():
    st.session_state.show_popup = True

# Funzione per chiudere il popup
def close_popup():
    st.session_state.show_popup = False

# --- Campi di Input nella Sidebar ---
st.sidebar.header("Parametri di Input")

# Aggiungi l'input per la chiave API FRED
fred_api_key = st.sidebar.text_input(
    "Chiave API FRED (obbligatoria per dati FRED):",
    type="password", # Nasconde la chiave mentre viene digitata
    help="Ottieni la tua chiave API gratuita su https://fred.stlouisfed.org/docs/api/api_key.html. Puoi anche impostarla come variabile d'ambiente 'FRED_API_KEY' sul tuo sistema."
)

# Definizione della lista di FRED series IDs anche in app.py per passarla al plotting
fred_series_ids = ['UNRATE', 'DGS10', 'DGS02', 'DGS30',
                   'GDPC1', 'CPIAUCSL', 'FEDFUNDS', 'RSAFS', 'INDPRO',
                   'UMICHCSII', 'PERMIT', 'ICSA', 'AWHMAN',
                   'NEWORDER', 'IPMAN', 'ISRATIO', 'HOUST', 'TCU', 'DSPIC96', 'CPILFESL']

# 1. Text box per i ticker/serie dati
identifier_input = st.sidebar.text_area(
    "Inserisci i simboli (es. SPY, GLD) o ID serie FRED (es. UNRATE, DGS10) separati da virgole o spazi:",
    value="SPY, QQQ, UNRATE, DGS10", # Esempio di default
    help="Ad esempio: SPY, QQQ, UNRATE. Gli ID serie FRED supportati includono: UNRATE (tasso disoccupazione), DGS10 (rendimento 10Y), DGS02 (rendimento 2Y), DGS30 (rendimento 30Y), GDPC1 (PIL Reale), CPIAUCSL (Inflazione CPI), FEDFUNDS (Tasso Federal Funds), RSAFS (Vendite al Dettaglio), INDPRO (Produzione Industriale), UMICHCSII (Sentimento Consumatori), PERMIT (Permessi Costruzione), ICSA (Sussidi Disoccupazione Iniziali), AWHMAN (Ore Lavoro Manifatturiero), NEWORDER (Nuovi Ordini Beni Durevoli), IPMAN (Produzione Manifatturiera), ISRATIO (Rapporto Scorte/Vendite), HOUST (Inizi Costruzione Case), TCU (Tasso Utilizzo Capacità), DSPIC96 (Reddito Personale Disponibile Reale), CPILFESL (Inflazione Core)."
)

# 2. Date di inizio e fine
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input(
    "Data di Inizio:",
    value=pd.to_datetime("2022-01-01"),
    help="Seleziona la data di inizio per il recupero dei dati."
)
end_date = col2.date_input(
    "Data di Fine:",
    value=pd.to_datetime("2024-12-31"),
    help="Seleziona la data di fine per il recupero dei dati."
)

# 3. Granularità dei dati (intervallo) per Yahoo Finance
interval_options = {
    "1 giorno": "1d",
    "1 settimana": "1wk",
    "1 mese": "1mo"
}
selected_interval_label = st.sidebar.selectbox(
    "Granularità Desiderata (per Yahoo Finance):",
    options=list(interval_options.keys()),
    index=0, # Default to '1 giorno'
    help="Seleziona l'intervallo di tempo per i dati Yahoo Finance. **Nota:** Se vengono inclusi indicatori FRED con frequenza mensile o superiore, l'analisi verrà eseguita su base mensile per tutti i dati per garantire l'allineamento."
)
yf_interval = interval_options[selected_interval_label]

# 4. Selezione del tipo di prezzo per Yahoo Finance
price_type_options = ["Adj Close", "Close", "Open", "High", "Low"] # Re-aggiunto "Adj Close"
selected_price_type = st.sidebar.selectbox(
    "Tipo di Prezzo (per Yahoo Finance):",
    options=price_type_options,
    index=0, # Default to 'Adj Close'
    help="Scegli quale tipo di prezzo utilizzare per i ticker di Yahoo Finance."
)


# 5. Max Lag
max_lags = st.sidebar.slider(
    "Massimo Lag (periodi):",
    min_value=1,
    max_value=60,
    value=20,
    step=1,
    help="Numero massimo di periodi da considerare per l'anticipo/ritardo. Questo sarà relativo alla frequenza di correlazione finale (es. 20 mesi, 20 giorni)."
)

# Prepara gli identificatori (ticker o FRED ID)
identifiers = [id.strip().upper() for id in identifier_input.replace(',', ' ').split() if id.strip()]

# Mapping per l'ordinamento delle frequenze (dal più fine al più grossolano)
freq_ranking = {'D': 0, 'B': 0, 'W': 1, 'M': 2, 'MS': 2, 'Q': 3, 'QS': 3, 'A': 4, 'AS': 4}
freq_label_map = {'D': 'giornaliera', 'W': 'settimanale', 'M': 'mensile', 'Q': 'trimestrale', 'A': 'annuale'}

def get_base_freq_key(freq_str):
    """Estrae la chiave base della frequenza da una stringa di frequenza inferita da Pandas."""
    if freq_str is None:
        return None
    if freq_str.startswith('D') or freq_str.startswith('B'): return 'D'
    if freq_str.startswith('W'): return 'W'
    if freq_str.startswith('M'): return 'M'
    if freq_str.startswith('Q'): return 'Q'
    if freq_str.startswith('A'): return 'A'
    return None

# Bottone per eseguire l'analisi
if st.sidebar.button("Esegui Analisi"):
    if len(identifiers) < 2:
        st.warning("Per eseguire l'analisi di cross-correlazione, devi inserire almeno due simboli/ID serie.")
    else:
        st.header("Risultati dell'Analisi di Cross-Correlazione")
        st.markdown("---")

        # --- FASE 1: Download Iniziale e Inferenza Frequenza per tutti gli Identificatori ---
        all_series_raw_and_freq = {}
        coarsest_freq_rank = -1 # Inizia con il rank più basso (frequenza più fine)
        final_correlation_frequency_code = yf_interval # Inizializza con la preferenza utente YF

        st.info("Scaricando e verificando le frequenze di tutti i dati. Questo passaggio potrebbe richiedere qualche istante.")
        with st.spinner("Preparazione dati..."):
            for identifier in identifiers:
                # Passa la chiave API FRED alla funzione di caricamento dati
                series_raw, inferred_freq, error = get_data_for_identifier(
                    identifier,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    yf_interval,
                    selected_price_type,
                    fred_api_key # Passa qui la chiave API
                )
                if error:
                    st.error(f"Errore durante il download o la determinazione della frequenza per '{identifier}': {error}")
                    st.stop() # Ferma l'esecuzione se un dato critico non può essere scaricato

                if series_raw is None: # Se non ci sono dati validi per l'identificatore
                    st.error(f"Nessun dato valido scaricato per '{identifier}'. Controlla il simbolo o il periodo.")
                    st.stop()

                base_freq_key = get_base_freq_key(inferred_freq)
                if base_freq_key is None:
                    st.warning(f"Impossibile inferire una frequenza base per '{identifier}'. Si userà la frequenza di default.")
                    base_freq_key = get_base_freq_key(yf_interval) # Fallback

                current_freq_rank = freq_ranking.get(base_freq_key, -1) # -1 per frequenze sconosciute (dovrebbe essere gestito)
                if current_freq_rank > coarsest_freq_rank:
                    coarsest_freq_rank = current_freq_rank
                    final_correlation_frequency_code = base_freq_key # Aggiorna la frequenza di resampling

                all_series_raw_and_freq[identifier] = series_raw # Conserva solo la serie raw

        # Traduci il codice della frequenza finale in un'etichetta leggibile
        correlation_frequency_label = freq_label_map.get(final_correlation_frequency_code, "sconosciuta")
        if coarsest_freq_rank > freq_ranking.get(get_base_freq_key(yf_interval), 0): # Se la frequenza finale è più grossolana di quella desiderata dall'utente
            st.warning(f"È stata rilevata una frequenza di dati più grossolana tra gli indicatori selezionati. L'analisi di cross-correlazione verrà eseguita su base **{correlation_frequency_label}** per tutti i dati.")
        else:
            st.info(f"L'analisi di cross-correlazione verrà eseguita su base **{correlation_frequency_label}** per tutti i dati.")


        # Liste per i risultati finali
        summary_data = []
        graphical_results_list = []

        # --- FASE 2: Calcolo delle Correlazioni su Dati Resampled ---
        id_combinations = list(itertools.combinations(identifiers, 2))

        if not id_combinations:
            st.info("Nessuna combinazione di simboli/ID serie valida da analizzare. Controlla i tuoi input.")
        else:
            with st.spinner(f"Calcolando le cross-correlazioni su base {correlation_frequency_label}..."):
                for id1, id2 in id_combinations:
                    series1_raw = all_series_raw_and_freq.get(id1)
                    series2_raw = all_series_raw_and_freq.get(id2)

                    # Questo dovrebbe essere già gestito dalla Fase 1, ma per sicurezza
                    if series1_raw is None or series2_raw is None:
                        error_msg = f"Errore interno: dati non disponibili per '{id1}' o '{id2}'. Riprova l'analisi."
                        st.error(error_msg)
                        graphical_results_list.append((id1, id2, error_msg, None, None))
                        continue

                    # Resampling di entrambe le serie alla frequenza comune determinata
                    series1_resampled = series1_raw.resample(final_correlation_frequency_code).last().ffill().dropna()
                    series2_resampled = series2_raw.resample(final_correlation_frequency_code).last().ffill().dropna()

                    if series1_resampled.empty or series2_resampled.empty:
                        error_msg = f"Nessun dato comune disponibile tra '{id1}' e '{id2}' nel periodo selezionato dopo il resampling a frequenza {correlation_frequency_label}. Prova un periodo o una granularità diversa."
                        st.error(error_msg)
                        graphical_results_list.append((id1, id2, error_msg, None, None))
                        continue

                    # Unisci le serie di dati resampled per il plot dei trend.
                    combined_resampled_data_df = pd.DataFrame({series1_resampled.name: series1_resampled, series2_resampled.name: series2_resampled})
                    combined_resampled_data_df = combined_resampled_data_df.dropna() # Dropna finale dopo l'unione per allineamento perfetto

                    if combined_resampled_data_df.empty:
                        error_msg = f"Nessun dato comune disponibile tra '{id1}' e '{id2}' dopo il resampling e l'allineamento degli indici. Periodo troppo breve o dati mancanti?"
                        st.error(error_msg)
                        graphical_results_list.append((id1, id2, error_msg, None, None))
                        continue


                    # Calcola la cross-correlazione
                    lags, correlations, best_info, corr_error = calculate_cross_correlation(
                        series1_resampled,
                        series2_resampled,
                        max_lags,
                        correlation_frequency_label
                    )

                    if lags is not None:
                        best_lag, max_corr_value = best_info

                        summary_data.append({
                            "Indice 1": id1,
                            "Indice 2": id2,
                            "Valore Max Correlazione": max_corr_value,
                            "Tau (Lag)": best_lag
                        })

                        text_summary_content = f"Correlazione Massima: `{max_corr_value:.4f}` al Lag: `{best_lag}` (frequenza {correlation_frequency_label})\n\n"
                        if best_lag > 0:
                            text_summary_content += f"Questo suggerisce che **{id2}** tende ad anticipare **{id1}** di **{best_lag}** periodi ({correlation_frequency_label}).\n"
                            text_summary_content += f"I movimenti di {id2} si riflettono in {id1} dopo {best_lag} periodi."
                        elif best_lag < 0:
                            text_summary_content += f"Questo suggerisce che **{id1}** tende ad anticipare **{id2}** di **{-best_lag}** periodi ({correlation_frequency_label}).\n"
                            text_summary_content += f"I movimenti di {id1} si riflettono in {id2} dopo {-best_lag} periodi."
                        else:
                            text_summary_content += "Non è stata trovata una relazione di anticipo/ritardo significativa o la correlazione è massima a lag zero (movimenti contemporanei)."

                        graphical_results_list.append((
                            id1, id2,
                            text_summary_content,
                            plot_cross_correlation(lags, correlations, id1, id2, correlation_frequency_label),
                            plot_data_trends(combined_resampled_data_df, id1, id2, fred_series_ids) # Passa fred_series_ids qui
                        ))
                    else:
                        st.error(f"Errore calcolo correlazione per {id1} vs {id2}: {corr_error}")
                        graphical_results_list.append((id1, id2, f"Errore calcolo correlazione: {corr_error}", None, None))

            st.success("Analisi completata!")

            # Crea le due tab
            tab1, tab2 = st.tabs(["📊 Tabella Riepilogo", "📈 Dettagli Grafici"])

            with tab1:
                st.subheader("Tabella Riepilogo Correlazioni")
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df['Valore Max Correlazione Assoluto'] = summary_df['Valore Max Correlazione'].abs()
                    summary_df = summary_df.sort_values(by="Valore Max Correlazione Assoluto", ascending=False).drop(columns=['Valore Max Correlazione Assoluto'])
                    st.dataframe(summary_df, hide_index=True)
                else:
                    st.info("Nessun risultato di correlazione da mostrare nella tabella.")

            with tab2:
                st.subheader("Dettagli Grafici per Combinazione")
                if graphical_results_list:
                    for id1, id2, text_summary, fig_corr, fig_prices in graphical_results_list:
                        st.markdown(f"### {id1} vs {id2}")
                        if fig_corr is not None and fig_prices is not None:
                            col1_res, col2_res, col3_res = st.columns(3)
                            with col1_res:
                                st.subheader("Riepilogo")
                                st.markdown(text_summary)
                            with col2_res:
                                st.subheader("Cross-Correlazione")
                                st.pyplot(fig_corr)
                                plt.close(fig_corr)
                            with col3_res:
                                st.subheader("Andamento Dati Grezzi")
                                st.pyplot(fig_prices)
                                plt.close(fig_prices)
                        else:
                            st.error(text_summary)
                        st.markdown("---")
                else:
                    st.info("Nessun risultato grafico da mostrare.")


# --- Aggiungi il bottone per il popup nella sidebar ---
st.sidebar.markdown("---")
st.sidebar.button("Apri Informazioni Extra", on_click=open_popup)

# --- Contenitore per il popup ---
if st.session_state.show_popup:
    with st.container(border=True):
        st.subheader("Indici FRED Disponibili")
        st.markdown("""
        Di seguito trovi un elenco degli ID serie FRED che puoi utilizzare nell'input, con il loro nome esteso e una breve descrizione.

        * **UNRATE**: **Tasso di Disoccupazione Civile (Civilian Unemployment Rate)**
            * Misura la percentuale della forza lavoro civile che è disoccupata ma attivamente in cerca di lavoro. È un indicatore chiave della salute del mercato del lavoro.
        * **DGS10**: **Rendimento del Tesoro a 10 Anni (10-Year Treasury Constant Maturity Rate)**
            * Rappresenta il rendimento annuale che un investitore si aspetterebbe da un'obbligazione del Tesoro statunitense con scadenza a 10 anni. È un punto di riferimento importante per i tassi di interesse a lungo termine e le aspettative di inflazione e crescita.
        * **DGS02**: **Rendimento del Tesoro a 2 Anni (2-Year Treasury Constant Maturity Rate)**
            * Rappresenta il rendimento annuale di un'obbligazione del Tesoro statunitense con scadenza a 2 anni. Spesso utilizzato come indicatore delle aspettative della Federal Reserve sui tassi di interesse a breve termine.
        * **DGS30**: **Rendimento del Tesoro a 30 Anni (30-Year Treasury Constant Maturity Rate)**
            * Rappresenta il rendimento annuale di un'obbligazione del Tesoro statunitense con scadenza a 30 anni, riflettendo le aspettative a lunghissimo termine su inflazione e crescita.
        * **GDPC1**: **Prodotto Interno Lordo Reale (Real Gross Domestic Product)**
            * Misura il valore totale di beni e servizi finali prodotti negli Stati Uniti, aggiustato per l'inflazione. È l'indicatore più ampio dell'attività economica e della crescita.
        * **CPIAUCSL**: **Indice dei Prezzi al Consumo - Tutti gli Articoli (Consumer Price Index for All Urban Consumers: All Items)**
            * Misura la variazione media nel tempo dei prezzi pagati dai consumatori urbani per un paniere di beni e servizi di consumo. È l'indicatore più comune dell'inflazione.
        * **FEDFUNDS**: **Tasso sui Federal Funds (Federal Funds Effective Rate)**
            * Il tasso di interesse overnight a cui le banche si prestano riserve non impegnate nel saldo dei rispettivi conti presso la Federal Reserve. È il principale strumento della politica monetaria della Fed.
        * **RSAFS**: **Vendite al Dettaglio: Totale (Escluse le Vendite di Servizi Alimentari) (Retail Sales: Total (Excluding Food Services))**
            * Misura il valore totale delle vendite di beni al dettaglio. È un indicatore chiave della spesa dei consumatori e della domanda interna.
        * **INDPRO**: **Indice di Produzione Industriale (Industrial Production Index)**
            * Misura la produzione del settore manifatturiero, minerario e delle utility. Offre una panoramica dell'attività produttiva e della salute dell'industria.
        * **UMICHCSII**: **Indice di Sentimento dei Consumatori (University of Michigan: Consumer Sentiment Index)**
            * Un indicatore che riflette la fiducia dei consumatori nell'economia. Un sentiment elevato può portare a una maggiore spesa e crescita.
        * **PERMIT**: **Permessi di Costruzione (New Private Housing Units Authorized by Building Permits)**
            * Il numero di permessi rilasciati per la costruzione di nuove case private. È un indicatore anticipatore dell'attività nel settore immobiliare e dell'economia in generale.
        * **ICSA**: **Richieste Iniziali di Sussidi di Disoccupazione (Initial Claims for Unemployment Insurance, Seasonally Adjusted)**
            * Il numero di persone che per la prima volta richiedono i sussidi di disoccupazione. Un aumento può segnalare un indebolimento del mercato del lavoro.
        * **AWHMAN**: **Ore Settimanali Medie, Manifatturiero (Average Weekly Hours of Production and Nonsupervisory Employees: Manufacturing)**
            * Le ore medie lavorate dal personale di produzione e non supervisore nel settore manifatturiero. Variazioni in questo indice possono anticipare cambiamenti nell'occupazione totale.
        * **NEWORDER**: **Nuovi Ordini di Beni Durevoli (Manufacturers' New Orders: Durable Goods)**
            * Misura il valore monetario dei nuovi ordini ricevuti dai produttori di beni durevoli. È un indicatore chiave della domanda futura e della fiducia delle imprese.
        * **IPMAN**: **Produzione Manifatturiera (Manufacturing Output)**
            * Un sotto-componente dell'Indice di Produzione Industriale che si concentra specificamente sulla produzione del settore manifatturiero, sensibile ai cicli economici.
        * **ISRATIO**: **Rapporto Scorte/Vendite del Settore Manifatturiero e Commercio (Inventories to Sales Ratio: Manufacturing and Trade)**
            * Rapporto tra le scorte di magazzino delle imprese e le loro vendite. Un aumento può indicare vendite rallentate e futura contrazione della produzione.
        * **HOUST**: **Inizi di Costruzione di Nuove Case (Housing Starts: Total)**
            * Misura l'inizio effettivo della costruzione di nuove unità abitative, un indicatore anticipatore della fiducia nel settore immobiliare e nell'economia.
        * **TCU**: **Tasso di Utilizzo della Capacità Totale (Capacity Utilization: Total Industry)**
            * Misura la percentuale di capacità produttiva totale che viene effettivamente utilizzata nell'industria. Livelli elevati possono segnalare pressioni inflazionistiche e la necessità di nuovi investimenti.
        * **DSPIC96**: **Reddito Personale Disponibile Reale (Real Disposable Personal Income)**
            * Il reddito che gli individui hanno a disposizione per spendere o risparmiare dopo aver pagato le tasse, aggiustato per l'inflazione. Direttamente collegato alla spesa dei consumatori.
        * **CPILFESL**: **Inflazione "Core" (Consumer Price Index: All Items Less Food & Energy)**
            * Misura la variazione media nel tempo dei prezzi al consumo rimuovendo le componenti più volatili (cibo ed energia), fornendo una misura più stabile dell'inflazione di fondo.
        """)
        st.button("Chiudi", on_click=close_popup)


st.sidebar.markdown("---")
st.sidebar.markdown("Creato con Streamlit, yfinance e dati FRED")
