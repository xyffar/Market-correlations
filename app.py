import streamlit as st
import pandas as pd
import itertools # Per generare combinazioni di identificatori
import matplotlib.pyplot as plt # Assicurati che sia importato qui per plt.close()

# Importa le funzioni dai file di utilità
from data_loader import get_data_for_identifier
from correlation_analyzer import calculate_cross_correlation, plot_cross_correlation, plot_data_trends

# Imposta il titolo dell'applicazione Streamlit con un'emoji
st.set_page_config(
    page_title="📈 Cross-Correlazione Finanziaria", # Emoji aggiunta qui
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

indicators_info = """
### Indici FRED Disponibili

Di seguito trovi un elenco degli ID serie FRED che puoi utilizzare nell'input, con il loro nome esteso, una breve spiegazione e il motivo per cui potrebbero essere correlati con i mercati finanziari.

| Simbolo | Nome | Spiegazione | Motivo della Correlazione (con Mercati Finanziari) |
|---|---|---|---|
| **UNRATE** | Tasso di Disoccupazione Civile | Percentuale della forza lavoro disoccupata e in cerca di lavoro. | **Indicatore Lagging/Coincidente:** Un tasso di disoccupazione basso indica un'economia forte, con implicazioni positive per i consumi e gli utili aziendali (mercato azionario) e possibili pressioni inflazionistiche (tassi di interesse). |
| **DGS10** | Rendimento Tesoro 10 Anni | Rendimento annuale di un'obbligazione del Tesoro USA a 10 anni. | **Indicatore Leading/Coincidente:** Riflette le aspettative di inflazione e crescita economica a lungo termine. Un aumento suggerisce aspettative di crescita economica e inflazione più elevate, che possono influenzare i prezzi delle azioni (attraverso i tassi di sconto) e i mercati obbligazionari. |
| **DGS02** | Rendimento Tesoro 2 Anni | Rendimento annuale di un'obbligazione del Tesoro USA a 2 anni. | **Indicatore Leading:** Molto sensibile alle aspettative della Federal Reserve sui tassi di interesse a breve termine. Le sue variazioni possono anticipare le mosse della banca centrale, influenzando direttamente i tassi di prestito e l'appetito per il rischio. |
| **DGS30** | Rendimento Tesoro 30 Anni | Rendimento annuale di un'obbligazione del Tesoro USA a 30 anni. | **Indicatore Leading/Coincidente:** Riflette le aspettative a lunghissimo termine. Utile per analizzare la forma della curva dei rendimenti, che può predire recessioni o espansioni. |
| **GDPC1** | Prodotto Interno Lordo Reale | Misura il valore totale di beni e servizi finali prodotti, aggiustato per l'inflazione. | **Indicatore Coincidente/Lagging:** Il PIL è la misura più ampia dell'attività economica. La sua crescita indica salute economica, che generalmente supporta i profitti aziendali e i prezzi delle azioni. |
| **CPIAUCSL** | Indice dei Prezzi al Consumo - Tutti gli Articoli | Variazione media dei prezzi pagati dai consumatori per beni e servizi. | **Indicatore Coincidente:** L'inflazione erode il potere d'acquisto e può spingere le banche centrali ad alzare i tassi, influenzando i costi di finanziamento delle aziende e il valore degli asset. |
| **FEDFUNDS** | Tasso sui Federal Funds | Tasso di interesse overnight tra le banche per le riserve. | **Indicatore Coincidente/Lagging:** È il principale strumento della politica monetaria della Fed. Le sue modifiche (o aspettative di modifica) influenzano direttamente i tassi di interesse a breve termine e, di conseguenza, i mutui, i prestiti aziendali e la valutazione degli asset. |
| **RSAFS** | Vendite al Dettaglio: Totale (Escluse Food Services) | Valore totale delle vendite di beni al dettaglio. | **Indicatore Coincidente:** La spesa dei consumatori è un motore cruciale dell'economia. Forte crescita delle vendite al dettaglio indica fiducia dei consumatori e domanda robusta, positiva per le aziende del settore e il mercato azionario. |
| **INDPRO** | Indice di Produzione Industriale | Misura la produzione dei settori manifatturiero, minerario e delle utility. | **Indicatore Coincidente:** Riflette l'attività produttiva dell'economia. Una crescita indica espansione industriale, aumento degli ordini e potenziale aumento dell'occupazione. |
| **UMICHCSII** | Indice di Sentimento dei Consumatori | Riflette la fiducia dei consumatori nell'economia. | **Indicatore Leading:** Il sentimento dei consumatori spesso anticipa i loro futuri modelli di spesa e risparmio, influenzando la domanda aggregata e le prospettive di crescita economica. |
| **PERMIT** | Permessi di Costruzione | Numero di permessi rilasciati per la costruzione di nuove case private. | **Indicatore Leading:** L'attività edilizia è un motore economico importante. Un aumento dei permessi indica fiducia nel futuro dell'economia e della domanda, anticipando investimenti e creazione di posti di lavoro. |
| **ICSA** | Richieste Iniziali Sussidi Disoccupazione | Numero di persone che per la prima volta richiedono sussidi di disoccupazione. | **Indicatore Leading:** Un aumento improvviso può segnalare un indebolimento del mercato del lavoro e un rallentamento economico, portando a incertezza nei mercati. |
| **AWHMAN** | Ore Settimanali Medie, Manifatturiero | Ore medie lavorate nel settore manifatturiero. | **Indicatore Leading:** I datori di lavoro spesso modificano le ore di lavoro prima di assumere o licenziare. Un aumento può anticipare assunzioni future e maggiore produzione. |
| **NEWORDER** | Nuovi Ordini di Beni Durevoli | Valore monetario dei nuovi ordini ricevuti dai produttori di beni durevoli. | **Indicatore Leading:** È una misura della domanda futura per prodotti a lungo termine. Un aumento indica fiducia nelle prospettive economiche e intenzione di espansione aziendale. |
| **IPMAN** | Produzione Manifatturiera | Produzione del solo settore manifatturiero. | **Indicatore Coincidente:** Un sotto-componente di INDPRO, ma più specifico. La sua performance è un barometro della salute industriale e degli utili aziendali nel settore. |
| **ISRATIO** | Rapporto Scorte/Vendite | Rapporto tra scorte di magazzino e vendite delle imprese. | **Indicatore Leading/Coincidente:** Un aumento del rapporto può suggerire che le vendite stanno rallentando e che le aziende potrebbero ridurre la produzione in futuro per smaltire le scorte. |
| **HOUST** | Inizi di Costruzione Nuove Case | Inizi effettivi della costruzione di nuove unità abitative. | **Indicatore Leading:** Simile a PERMIT, conferma l'avvio di nuovi progetti. Un settore abitativo robusto è solitamente correlato a un'economia in crescita. |
| **TCU** | Tasso di Utilizzo della Capacità Totale | Percentuale della capacità produttiva totale utilizzata nell'industria. | **Indicatore Coincidente/Leading:** Alti livelli possono indicare che l'economia si sta surriscaldando e potrebbero esserci pressioni inflazionistiche future, o che le aziende investiranno per espandere la capacità. |
| **DSPIC96** | Reddito Personale Disponibile Reale | Reddito a disposizione degli individui dopo le tasse, aggiustato per l'inflazione. | **Indicatore Coincidente:** Direttamente legato al potere d'acquisto dei consumatori. Un aumento supporta la spesa e la crescita economica. |
| **CPILFESL** | Inflazione "Core" (CPI Less Food & Energy) | Variazione dei prezzi al consumo escludendo cibo ed energia (volatili). | **Indicatore Coincidente:** Spesso la metrica di inflazione preferita dalle banche centrali per le decisioni di politica monetaria, in quanto più stabile e rappresentativa delle pressioni inflazionistiche di fondo. |
"""

@st.dialog("Info about available economic indicators", width="large")
def show_infos():
    st.markdown(indicators_info, unsafe_allow_html=True) # Usa st.markdown e unsafe_allow_html per renderizzare la tabella
    if st.button("Chiudi", on_click=close_popup):
        st.rerun()

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
    value=pd.to_datetime("2025-12-31"),
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
price_type_options = ["Close", "Open", "High", "Low"]
selected_price_type = st.sidebar.selectbox(
    "Tipo di Prezzo (per Yahoo Finance):",
    options=price_type_options,
    index=0, # Default to 'Close'
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

if st.session_state.show_popup:
    show_infos()

st.sidebar.markdown("---")
st.sidebar.markdown("Creato con Streamlit, yfinance e dati FRED")
