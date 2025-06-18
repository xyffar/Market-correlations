import streamlit as st
import pandas as pd
import itertools # Per generare combinazioni di identificatori
import matplotlib as plt

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

# 1. Text box per i ticker/serie dati
identifier_input = st.sidebar.text_area(
    "Inserisci i simboli (es. SPY, GLD) o ID serie FRED (es. UNRATE, DGS10) separati da virgole o spazi:",
    value="SPY, QQQ, UNRATE, DGS10", # Esempio di default
    help="Ad esempio: SPY, QQQ, UNRATE. Gli ID serie FRED supportati includono: UNRATE (tasso disoccupazione), DGS10 (rendimento 10Y), DGS02 (rendimento 2Y), DGS30 (rendimento 30Y), GDPC1 (PIL Reale), CPIAUCSL (Inflazione CPI), FEDFUNDS (Tasso Federal Funds), RSAFS (Vendite al Dettaglio), INDPRO (Produzione Industriale), UMICHCSII (Sentimento Consumatori), PERMIT (Permessi Costruzione), ICSA (Sussidi Disoccupazione Iniziali), AWHMAN (Ore Lavoro Manifatturiero), NEWORDER (Nuovi Ordini Beni Durevoli), IPMAN (Produzione Manifatturiera), ISRATIO (Rapporto Scorte/Vendite), HOUST (Inizi Costruzione Case), TCU (Tasso Utilizzo Capacità), DSPIC96 (Reddito Personale Disponibile Reale)."
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
    "Granularità dei Dati (per Yahoo Finance):",
    options=list(interval_options.keys()),
    index=0, # Default to '1 giorno'
    help="Seleziona l'intervallo di tempo per i dati Yahoo Finance. Le serie FRED useranno la loro frequenza nativa (es. mensile per UNRATE, giornaliera per DGS10). Saranno allineate temporalmente."
)
yf_interval = interval_options[selected_interval_label]

# 4. Selezione del tipo di prezzo per Yahoo Finance
price_type_options = ["Adj Close", "Close", "Open", "High", "Low"]
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
    help="Numero massimo di periodi da considerare per l'anticipo/ritardo. Un lag di 20 giorni/periodi rappresenta circa un mese lavorativo/periodi di osservazione."
)

# Prepara gli identificatori (ticker o FRED ID)
identifiers = [id.strip().upper() for id in identifier_input.replace(',', ' ').split() if id.strip()]

# Bottone per eseguire l'analisi
if st.sidebar.button("Esegui Analisi"):
    if len(identifiers) < 2:
        st.warning("Per eseguire l'analisi di cross-correlazione, devi inserire almeno due simboli/ID serie.")
    else:
        st.header("Risultati dell'Analisi di Cross-Correlazione")
        st.markdown("---")

        # Lista per raccogliere i risultati della tabella riassuntiva
        summary_data = []
        # Lista per raccogliere i contenuti grafici per la seconda tab
        graphical_results_list = []

        # Genera tutte le combinazioni uniche di identificatori
        id_combinations = list(itertools.combinations(identifiers, 2))

        if not id_combinations:
            st.info("Nessuna combinazione di simboli/ID serie valida da analizzare. Controlla i tuoi input.")
        else:
            with st.spinner("Calcolando le cross-correlazioni... Potrebbe volerci del tempo per scaricare i dati."):
                for id1, id2 in id_combinations:
                    # Ottieni i dati grezzi per il primo identificatore, passando il tipo di prezzo
                    series1_raw, error1 = get_data_for_identifier(id1, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), yf_interval, selected_price_type)
                    # Ottieni i dati grezzi per il secondo identificatore, passando il tipo di prezzo
                    series2_raw, error2 = get_data_for_identifier(id2, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), yf_interval, selected_price_type)

                    # Gestione degli errori durante il download
                    if error1:
                        st.error(f"Errore per '{id1}': {error1}")
                        graphical_results_list.append((id1, id2, f"Errore per '{id1}': {error1}", None, None))
                        continue
                    if error2:
                        st.error(f"Errore per '{id2}': {error2}")
                        graphical_results_list.append((id1, id2, f"Errore per '{id2}': {error2}", None, None))
                        continue
                    if series1_raw is None or series2_raw is None:
                        st.error(f"Impossibile ottenere i dati per '{id1}' o '{id2}'. Prova a controllare gli input.")
                        graphical_results_list.append((id1, id2, f"Impossibile ottenere i dati per '{id1}' o '{id2}'.", None, None))
                        continue

                    # Unisci le serie di dati raw per il plot dei trend.
                    # Questo gestirà l'allineamento di frequenze diverse (es. mensile vs giornaliero)
                    # riempiendo i valori mancanti (forward fill) per allineare gli indici temporali.
                    combined_raw_data_df = pd.DataFrame({series1_raw.name: series1_raw, series2_raw.name: series2_raw})
                    combined_raw_data_df = combined_raw_data_df.asfreq(yf_interval).ffill().dropna()

                    if combined_raw_data_df.empty:
                        error_msg = f"Nessun dato comune disponibile tra '{id1}' e '{id2}' nel periodo selezionato dopo l'allineamento. Prova un periodo o una granularità diversa."
                        st.error(error_msg)
                        graphical_results_list.append((id1, id2, error_msg, None, None))
                        continue

                    # Calcola la cross-correlazione sulle serie allineate e pre-processate
                    # Passa le serie raw e lasciale pre-processare internamente a calculate_cross_correlation
                    lags, correlations, best_info, corr_error = calculate_cross_correlation(series1_raw, series2_raw, max_lags, selected_interval_label)

                    if lags is not None:
                        best_lag, max_corr_value = best_info

                        # Aggiungi i risultati alla lista per la tabella
                        summary_data.append({
                            "Indice 1": id1,
                            "Indice 2": id2,
                            "Valore Max Correlazione": max_corr_value,
                            "Tau (Lag)": best_lag
                        })

                        # Prepara i contenuti grafici per essere visualizzati nella seconda tab
                        graphical_results_list.append((
                            id1, id2,
                            f"Correlazione Massima: `{max_corr_value:.4f}` al Lag: `{best_lag}`\n\n" +
                            (f"Questo suggerisce che **{id2}** tende ad anticipare **{id1}** di **{best_lag}** periodi ({selected_interval_label})." if best_lag > 0 else
                             (f"Questo suggerisce che **{id1}** tende ad anticipare **{id2}** di **{-best_lag}** periodi ({selected_interval_label})." if best_lag < 0 else
                              "Non è stata trovata una relazione di anticipo/ritardo significativa o la correlazione è massima a lag zero (movimenti contemporanei).")),
                            plot_cross_correlation(lags, correlations, id1, id2, selected_interval_label),
                            plot_data_trends(combined_raw_data_df, id1, id2)
                        ))
                    else:
                        st.error(corr_error) # Visualizza l'errore se il calcolo della correlazione fallisce
                        graphical_results_list.append((id1, id2, f"Errore calcolo correlazione: {corr_error}", None, None))

            st.success("Analisi completata!")

            # Crea le due tab
            tab1, tab2 = st.tabs(["📊 Tabella Riepilogo", "📈 Dettagli Grafici"])

            with tab1:
                st.subheader("Tabella Riepilogo Correlazioni")
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    # Ordina la tabella in ordine decrescente per Valore Max Correlazione (valore assoluto)
                    summary_df['Valore Max Correlazione Assoluto'] = summary_df['Valore Max Correlazione'].abs()
                    summary_df = summary_df.sort_values(by="Valore Max Correlazione Assoluto", ascending=False).drop(columns=['Valore Max Correlazione Assoluto'])
                    st.dataframe(summary_df)
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
                                st.write(text_summary)
                            with col2_res:
                                st.subheader("Cross-Correlazione")
                                st.pyplot(fig_corr)
                                plt.close(fig_corr) # Chiudi la figura per liberare memoria
                            with col3_res:
                                st.subheader("Andamento Dati Grezzi")
                                st.pyplot(fig_prices)
                                plt.close(fig_prices) # Chiudi la figura per liberare memoria
                        else:
                            st.error(text_summary) # Mostra il messaggio di errore completo se i grafici non sono disponibili
                        st.markdown("---") # Separatore tra le combinazioni
                else:
                    st.info("Nessun risultato grafico da mostrare.")


# --- Aggiungi il bottone per il popup nella sidebar ---
st.sidebar.markdown("---")
st.sidebar.button("Apri Informazioni Extra", on_click=open_popup)

# --- Contenitore per il popup ---
if st.session_state.show_popup:
    # Utilizza st.container() o st.empty() per il popup
    # Un contenitore è utile per raggruppare elementi
    with st.container(border=True): # border=True aggiunge un bordo per renderlo visibile come popup
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
        """)
        st.button("Chiudi", on_click=close_popup)


st.sidebar.markdown("---")
st.sidebar.markdown("Creato con Streamlit, yfinance e dati FRED")
