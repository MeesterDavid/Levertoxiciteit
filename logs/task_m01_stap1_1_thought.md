# Denk- en Voortgangslog: Mijlpaal 1, Stap 1.1
- **Datum / Tijd:** 11/2/2026
- **Doel van de taak:** Echte humane data ophalen uit gevalideerde openbare databases voor zowel de gezonde baseline als pathologische (steatose/NAFLD) toestanden, gekoppeld aan officiële OECD AOP-literatuur.

## 1. Analyse & Probleemstelling
- Wat wordt er gevraagd?
  - Schrijf een Python-script (`fetch_real_hepatocyte_data.py`) dat echte humane data ophaalt voor 18 genen en 20 gerichte interacties.
  - Maak een Marimo-notebook (`m01_1_data_explorer.py`) om de data interactief te verkennen.
- Welke mogelijke valkuilen of randvoorwaarden zijn er?
  - API-limieten en rate-limits.
  - Data-normalisatie en afhandeling van ontbrekende waarden.
  - Correcte verwerking van de AOP-literatuurverwijzingen.

## 2. Methodologische & Wetenschappelijke Overwegingen
- Waarom kiezen we deze specifieke aanpak/bibliotheek/wiskundige formule?
  - MyGene.info en Ensembl API's bieden betrouwbare en gedetailleerde gen-informatie.
  - GTEx en Human Protein Atlas bieden basale expressie-informatie.
  - NCBI GEO biedt pathologische expressie-informatie.
  - STRING-db API biedt experimenteel gevalideerde interactiescores.
  - Floyd-Warshall Shortest-Path Distance matrix is een bekende en efficiënte methode voor het berekenen van kortste paden in netwerken.
- Hoe sluit dit aan bij het onderzoeksvoorstel (bijv. AOP's, attractor dynamics, API-bronnen)?
  - De data-acquisitie stelt ons in staat om de biologische netwerken en hun dynamiek nauwkeurig te modelleren.
  - De AOP-literatuurverwijzingen verankeren de modelopbouw in wetenschappelijke literatuur.

## 3. Stap-voor-stap Uitvoeringsplan
- Welke bestanden worden aangemaakt of gewijzigd?
  - `fetch_real_hepatocyte_data.py`
  - `m01_1_data_explorer.py`
  - `data/hepatocyte_baseline_data.json`
  - `data/hepatocyte_features.pt`
- Wat is de opbouw van de logica?
  - **Data-acquisitie script:**
    1. Importeer benodigde bibliotheken.
    2. Definieer de 18 genen en 20 gerichte interacties.
    3. Haal data op via MyGene.info, GTEx, HPA, en NCBI GEO.
    4. Haal interactiescores op via STRING-db.
    5. Bereken de Floyd-Warshall Shortest-Path Distance matrix.
    6. Sla de data op in JSON en PyTorch tensor format.
  - **Marimo-notebook:**
    1. Importeer benodigde bibliotheken.
    2. Laad de data uit `hepatocyte_baseline_data.json`.
    3. Voeg interactieve Marimo UI widgets toe.
    4. Maak interactieve Plotly visualisaties.
    5. Controleer de notebook op syntax-fouten.

## 4. Terminal Executie & Verificatie (GPU 1)
- Exact uitgevoerd commando: `CUDA_VISIBLE_DEVICES=1 uv run python fetch_real_hepatocyte_data.py`
- Output / Terminal log (inclusief geverifieerde tensor shapes en ontbreken van NaNs):
  - [Te vervolledigen na uitvoering]

## 5. Reflectie & Volgende Stap
- Werkt alles naar behoren?
  - [Te vervolledigen na uitvoering]
- Wat is het directe vervolg?
  - Voortzetten met Mijlpaal 1, Stap 1.2: Graphormer Encoder (`m01_graph_embedding.py`).