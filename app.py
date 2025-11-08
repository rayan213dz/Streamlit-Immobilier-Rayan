import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from streamlit_option_menu import option_menu

# -----------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------
st.set_page_config(page_title="Analyse Immobilière Étudiante", page_icon="🏠", layout="wide")

# -----------------------------
# BARRE DE NAVIGATION HORIZONTALE
# -----------------------------
menu = option_menu(
    menu_title=None,  # pas de titre
    options=[
        "Faisabilité d'achat étudiant",
        "Rendement brut minimal",
        "Répartition DVF par budget",
        "Tension locative (INSEE)",
        "Rendement par département",
        "Indice achat-location",
        "Comparateur DVF 2020–2024",
        "Carte d’accessibilité — DVF 2024"
    ],
    icons=[
        "person-check", "graph-up", "bar-chart", "city",
        "cash-coin", "house-heart", "columns-gap", "map"
    ],
    menu_icon="house",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0e1117"},
        "icon": {"color": "orange", "font-size": "16px"},
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#262730",
        },
        "nav-link-selected": {"background-color": "#ff4b4b"},
    },
)

# 🌟 --- En-tête ---
st.markdown("""
# 🏠 Portfolio Rayan - Projet immobilier interactif  
### Analyse et visualisation de données immobilières françaises  

Ce tableau de bord fait partie de mon portfolio, illustrant mes compétences en :
- data analysis (Python, Pandas, NumPy)  
- data visualization (Plotly, Streamlit)  
- gestion et nettoyage de données publiques (DVF, INSEE)  

Il combine **données publiques (DVF, INSEE, loyers 2024)** et outils analytiques pour explorer :
- la faisabilité d’un achat étudiant 
- les rendements bruts et tensions locatives 
- la relation entre **achat et location** selon les départements   

> Inspiré d’un projet collaboratif, puis retravaillé individuellement pour le portfolio.
""")

st.divider()




# -----------------------------
# CHARGEMENT DES DONNÉES (local)
# -----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

try:
    # ⚙️ Chargement depuis le dossier local
    df_insee = load_data("data/clean/insee_logement_2021_clean.csv")
    df_loyers = load_data("data/clean/loyers_clean_2024.csv")
    df_dvf = load_data("data/clean/dvf_clean_2024.csv")
    df_dvf_all = load_data("data/clean/dvf_clean_2020_2024_sample.csv")

    st.sidebar.success("✅ Données chargées depuis le dossier local : data/clean")

except Exception as e:
    st.sidebar.error(f"❌ Erreur de chargement des fichiers : {e}")

# -----------------------------
# 1️⃣ Faisabilité d'achat

# -----------------------------
if menu == "Faisabilité d'achat étudiant":
    st.header("📋 Faisabilité d'achat — profil étudiant")
    revenu = st.slider("Revenus mensuels (€)", 0, 3000, 800, step=100)
    statut = st.selectbox("Statut", ["Étudiant pur", "Étudiant avec CDI partiel", "Alternant"])
    logement = st.selectbox("Situation", ["Chez les parents", "Locataire"])
    duree_etudes = st.slider("Années d’études restantes", 0, 6, 2)
    salaire_sortie = st.slider("Salaire prévu à la sortie (€)", 1000, 4000, 1800, step=100)
    garant = st.checkbox("Garantie parentale", True)
    apport = st.slider("Apport (€)", 0, 30000, 5000, step=500)

    score = 50
    conseils = []

    score += 15 if revenu >= 1000 else (-15 if revenu < 500 else 0)
    if statut == "Alternant":
        score += 25
        conseils.append("Statut d’alternant : revenu régulier et expérience, profil rassurant.")
    elif statut == "Étudiant avec CDI partiel":
        score += 15
        conseils.append("CDI partiel : stabilité appréciée par les banques.")
    else:
        score -= 25
        conseils.append("Étudiant sans revenu fixe : viser garant et/ou prêt différé.")

    if logement == "Locataire":
        score -= 10
        conseils.append("Loyer existant : conserver une marge de sécurité.")
    else:
        score += 10
        conseils.append("Pas de loyer : meilleure capacité d’épargne et de remboursement.")

    if duree_etudes >= 3:
        score -= 10
        conseils.append("Plusieurs années d’études restantes : différé recommandé.")
    elif duree_etudes == 0:
        score += 5
        conseils.append("Fin d’études proche : crédibilité renforcée.")

    score += 10 if salaire_sortie >= 2000 else (-10 if salaire_sortie < 1500 else 0)

    if apport >= 10000:
        score += 15
        conseils.append("Apport ≥ 10 k€ : très bon signal pour la banque.")
    elif 5000 <= apport < 10000:
        score += 8
        conseils.append("Apport modéré : dossier solide.")
    elif 1 <= apport < 5000:
        score += 2
        conseils.append("Apport faible : garant conseillé.")
    else:
        score -= 10
        conseils.append("Sans apport : projet plus difficile à financer.")

    score += 10 if garant else -10
    score = max(0, min(100, score))

    if score >= 75:
        color, verdict = "🟢", "Faisabilité élevée"
    elif score >= 50:
        color, verdict = "🟡", "Faisabilité moyenne"
    else:
        color, verdict = "🔴", "Faisabilité faible"

    st.subheader(f"{color} Score : {score}/100 — {verdict}")
    st.progress(score / 100)
    st.write("### Conseils :")
    for c in conseils:
        st.markdown(f"- {c}")

# -----------------------------
# 2️⃣ Rendement brut minimal
# -----------------------------
elif menu == "Rendement brut minimal":
    st.header("📈 Rendement brut minimal — simulateur")
    apport = st.number_input("Apport (€)", 0, 100000, 10000, step=1000)
    emprunt = st.number_input("Montant emprunté (€)", 10000, 400000, 90000, step=1000)
    taux = st.number_input("Taux annuel (%)", 0.1, 8.0, 4.0, step=0.1)
    duree = st.slider("Durée du prêt (ans)", 5, 30, 20)
    charges = st.checkbox("Inclure 20% de charges", True)

    def mensualite(capital, taux_annuel_pct, duree_ans):
        t = taux_annuel_pct / 100 / 12
        n = duree_ans * 12
        return capital * (t * (1 + t)**n) / ((1 + t)**n - 1)

    m = mensualite(emprunt, taux, duree)
    annuite = m * 12
    prix = apport + emprunt
    coef_net = 0.8 if charges else 1.0
    loyer_annuel = annuite / coef_net
    rendement = (loyer_annuel / prix) * 100 if prix > 0 else np.nan

    st.metric("Prix total", f"{prix:,.0f} €".replace(",", " "))
    st.metric("Mensualité", f"{m:,.2f} € / mois".replace(",", " "))
    st.metric("Rendement brut requis", f"{rendement:.2f} %")

# -----------------------------
# 3️⃣ Répartition DVF par budget
# -----------------------------
elif menu == "Répartition DVF par budget":
    st.header("🏘️ Répartition des ventes — DVF 2024")
    try:
        df = df_dvf.copy()
        budget = st.slider("Budget maximum (€)", 50000, 300000, 100000, step=10000)
        d = df[df["valeur_fonciere"] <= budget]

        if d.empty:
            st.warning("Aucun bien sous ce budget.")
        else:
            d["type_simple"] = d["type_local"].str.upper().map(
                lambda x: "Appartement" if "APPART" in x else ("Maison" if "MAISON" in x else "Autre")
            )
            d = d[(d["nombre_pieces_principales"] >= 1) & (d["nombre_pieces_principales"] <= 8)]
            ventes = d.groupby(["type_simple", "nombre_pieces_principales"]).size().reset_index(name="ventes")

            fig = px.bar(
                ventes,
                x="nombre_pieces_principales",
                y="ventes",
                color="type_simple",
                barmode="group",
                text="ventes",
                template="plotly_white",
                title=f"Ventes ≤ {budget:,} € par taille du logement"
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(type="category", range=[0.5, 8.5])  # ✅ Fix abscisses
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), bargap=0.15)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Erreur : {e}")

elif menu == "Tension locative (INSEE)":
    st.header("🏙️ Tension locative — INSEE 2021")

    import os

    # Vérif fichier
    PATH = "data/clean/insee_logement_2021_clean.csv"
    if not os.path.exists(PATH):
        st.error("❌ Fichier manquant : data/clean/insee_logement_2021_clean.csv")
    else:
        df_insee = pd.read_csv(PATH, low_memory=False)
        df_insee["CODGEO"] = df_insee["CODGEO"].astype(str)
        df_insee["Code departement"] = df_insee["CODGEO"].str[:2]

        # Calcul taux de vacance si absent
        if "taux_vacance" not in df_insee.columns and {"P21_LOGVAC", "P21_LOG"} <= set(df_insee.columns):
            df_insee["taux_vacance"] = 100 * df_insee["P21_LOGVAC"] / df_insee["P21_LOG"]

        df_insee["taux_vacance"] = pd.to_numeric(df_insee["taux_vacance"], errors="coerce")
        df_insee = df_insee[df_insee["taux_vacance"].between(0, 50, inclusive="both")].copy()

        # Agrégation départementale
        df_dept = (
            df_insee.groupby("Code departement", as_index=False)
            .agg(
                logements=("P21_LOG", "sum"),
                communes=("CODGEO", "nunique"),
                vac_mean=("taux_vacance", "mean"),
                vac_med=("taux_vacance", "median"),
            )
            .sort_values("vac_mean", ascending=True)
            .reset_index(drop=True)
        )
        df_dept["vac_mean"] = df_dept["vac_mean"].round(2)
        df_dept["vac_med"] = df_dept["vac_med"].round(2)
        df_dept["rang_tendu"] = np.arange(1, len(df_dept) + 1)

        # Widgets Streamlit
        departements = sorted(df_insee["Code departement"].unique())
        dept = st.selectbox("Département", departements, index=departements.index("75") if "75" in departements else 0)
        nbins = st.slider("Nombre de classes (histogramme)", 10, 80, 30, step=5)
        dens = st.checkbox("Afficher densité", value=False)

        # Données département
        data = df_insee[df_insee["Code departement"] == dept].copy()
        if data.empty:
            st.warning("Aucune donnée disponible pour ce département.")
        else:
            # KPIs
            nb_com = data["CODGEO"].nunique()
            vac_med = data["taux_vacance"].median()
            vac_moy = data["taux_vacance"].mean()
            vac_q1 = data["taux_vacance"].quantile(0.25)
            vac_q3 = data["taux_vacance"].quantile(0.75)
            row = df_dept[df_dept["Code departement"] == dept].iloc[0]
            logements_dep = int(row["logements"])
            rang = int(row["rang_tendu"])

            # Badge
            if vac_med < 5:
                badge = "🟢 Contexte très tendu"
            elif vac_med < 8:
                badge = "🟡 Contexte plutôt tendu"
            else:
                badge = "🔴 Contexte plus détendu"

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader(f"Département {dept}")
                st.caption("Distribution du taux de vacance par commune (INSEE 2021)")
            with col2:
                st.metric("Communes", f"{nb_com:,}".replace(",", " "))
                st.metric("Logements", f"{logements_dep:,}".replace(",", " "))
            with col3:
                st.metric("Médiane vacance", f"{vac_med:.2f} %")
                st.metric("Moyenne vacance", f"{vac_moy:.2f} %")
            st.info(f"📊 {badge} — Rang national : **{rang}ᵉ**")

            # Graphique
            histnorm = "probability density" if dens else None
            fig = px.histogram(
                data,
                x="taux_vacance",
                nbins=nbins,
                histnorm=histnorm,
                template="plotly_white",
                title=f"Distribution — Département {dept}",
                labels={"taux_vacance": "Taux de vacance (%)"},
            )
            fig.add_vline(x=vac_med, line_dash="dash", annotation_text="Médiane", annotation_position="top left")
            fig.add_vline(x=vac_moy, line_dash="dot", annotation_text="Moyenne", annotation_position="top right")
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Tableau top 20
            st.subheader("🏆 Top 20 départements les plus tendus")
            top20 = df_dept.nsmallest(20, "vac_mean")[["rang_tendu", "Code departement", "logements", "communes", "vac_mean"]]
            st.dataframe(
                top20.style.format({
                    "logements": "{:,.0f}".format,
                    "communes": "{:,.0f}".format,
                    "vac_mean": "{:.2f} %",
                })
            )

# -----------------------------
# 5️⃣ Rendement par département
# -----------------------------
elif menu == "Rendement par département":
    st.header("💶 Rendement brut — par département")
    try:
        df_dvf["code_departement"] = df_dvf["code_departement"].astype(str).str.zfill(2)
        df_dvf = df_dvf[(df_dvf["prix_m2"] > 200) & (df_dvf["prix_m2"] < 15000)]
        prix_m2_dept = (
            df_dvf.groupby(["code_departement", "type_local"])
            .agg(prix_m2_median=("prix_m2", "median"))
            .reset_index()
        )

        df_loyers["dep"] = df_loyers["dep"].astype(str).str.zfill(2)
        df_loyers = df_loyers[df_loyers["type_détaillé"] != "Appartement - Tous"]
        df_loyers["type_local_normalise"] = df_loyers["type_détaillé"].apply(
            lambda x: "Appartement" if "Appartement" in x else "Maison"
        )

        loyer_dept = (
            df_loyers.groupby(["dep", "type_détaillé", "type_local_normalise"])
            .agg(loyer_m2_moyen=("loypredm2", "mean"))
            .reset_index()
        )

        # Fusion DVF + loyers
        fusion = prix_m2_dept.merge(
            loyer_dept,
            left_on=["code_departement", "type_local"],
            right_on=["dep", "type_local_normalise"],
            how="inner"
        )
        fusion["rendement"] = (fusion["loyer_m2_moyen"] * 12 / fusion["prix_m2_median"]) * 100
        fusion["rendement"] = fusion["rendement"].round(2)

        # Sélection du type de bien
        type_bien = st.selectbox(
            "🏘️ Type de bien",
            ["Appartement - 1 ou 2 pièces", "Appartement - 3 pièces ou plus", "Maison"]
        )
        seuil = st.slider("Seuil minimal (%)", 3.0, 10.0, 6.0, 0.5)

        data = fusion[fusion["type_détaillé"] == type_bien].copy()
        if data.empty:
            st.warning("Aucune donnée disponible pour ce type de bien.")
        else:
            data = data[data["rendement"] >= seuil].sort_values("rendement", ascending=False)

            fig = px.bar(
                data,
                x="code_departement",
                y="rendement",
                color="rendement",
                color_continuous_scale="RdYlGn",
                text="rendement",
                template="plotly_white",
                title=f"{type_bien} — Rendement ≥ {seuil:.1f} %"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            fig.update_xaxes(type="category", tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            top10 = data.nlargest(10, "rendement")[["code_departement", "prix_m2_median", "loyer_m2_moyen", "rendement"]]
            st.markdown("**🏆 Top 10 départements par rendement :**")
            st.dataframe(
                top10.style.background_gradient(subset=["rendement"], cmap="RdYlGn", vmin=4, vmax=10)
                .format({
                    "prix_m2_median": "{:.0f} €",
                    "loyer_m2_moyen": "{:.2f} €/m²",
                    "rendement": "{:.2f} %"
                })
            )

    except Exception as e:
        st.error(f"❌ Erreur : {e}")


# -----------------------------
# 7️⃣ Comparateur DVF 2020–2024
# -----------------------------
elif menu == "Comparateur DVF 2020–2024":
    st.header("🏘️ Comparateur DVF 2020–2024 — prix médians et volumes")

    if df_dvf_all is None or df_dvf_all.empty:
        st.error("❌ Fichier manquant ou vide : data/clean/dvf_clean_2020_2024.csv")
    else:
        dvf = df_dvf_all.copy()

        needed = {"annee", "type_local", "prix_m2", "valeur_fonciere"}
        miss = needed - set(dvf.columns)
        if miss:
            st.error(f"⚠️ Colonnes manquantes : {miss}")
        else:
            # Nettoyage
            def simplify_type(x: str):
                x = str(x).strip().upper()
                if "APPART" in x: return "Appartement"
                if "MAISON" in x:  return "Maison"
                return "Autre"

            dvf["type_simple"] = dvf["type_local"].map(simplify_type)
            dvf = dvf[dvf["prix_m2"].between(300, 20000)]
            dvf = dvf[dvf["annee"].between(2020, 2024)]

            years = sorted(dvf["annee"].unique().tolist())
            y_min, y_max = int(min(years)), int(max(years))

            # Widgets
            type_sel = st.selectbox("Type de bien", ["Tous", "Appartement", "Maison"])
            y1, y2 = st.slider("Période d'analyse", y_min, y_max, (y_min, y_max))
            show_index = st.checkbox("Afficher en indice (base 100 au début)", False)

            # Filtre + agrég
            d = dvf[(dvf["annee"] >= y1) & (dvf["annee"] <= y2)].copy()
            if type_sel != "Tous":
                d = d[d["type_simple"] == type_sel]

            if d.empty:
                st.warning("Aucune donnée sur cette période.")
            else:
                g = (d.groupby("annee", as_index=False)
                       .agg(prix_m2_median=("prix_m2","median"),
                            ventes=("valeur_fonciere","size"))
                       .sort_values("annee"))

                # KPIs
                def kpis(g):
                    if len(g) < 2:
                        return dict(delta_p=np.nan, delta_v=np.nan, vol=np.nan, cagr=np.nan)
                    p0, p1 = g["prix_m2_median"].iloc[0], g["prix_m2_median"].iloc[-1]
                    v0, v1 = g["ventes"].iloc[0], g["ventes"].iloc[-1]
                    n_years = int(g["annee"].iloc[-1] - g["annee"].iloc[0])
                    delta_p = (p1/p0 - 1)*100 if p0>0 else np.nan
                    delta_v = (v1/v0 - 1)*100 if v0>0 else np.nan
                    vol = (g["prix_m2_median"].std()/g["prix_m2_median"].mean()) if g["prix_m2_median"].mean()>0 else np.nan
                    cagr = ((p1/p0)**(1/n_years) - 1)*100 if p0>0 and n_years>0 else np.nan
                    return dict(delta_p=delta_p, delta_v=delta_v, vol=vol, cagr=cagr)

                k = kpis(g)
                c1, c2, c3, c4 = st.columns(4)
                # KPIs (affichage propre)
                if np.isfinite(k["delta_p"]):
                    c1.metric("Évolution prix", f"{k['delta_p']:.2f} %")
                else:
                    c1.metric("Évolution prix", "—")

                if np.isfinite(k["delta_v"]):
                    c2.metric("Évolution volume", f"{k['delta_v']:.2f} %")
                else:
                    c2.metric("Évolution volume", "—")

                if np.isfinite(k["vol"]):
                    c3.metric("Volatilité prix", f"{k['vol']*100:.1f} %")
                else:
                    c3.metric("Volatilité prix", "—")

                if np.isfinite(k["cagr"]):
                    c4.metric("CAGR prix", f"{k['cagr']:.2f} %")
                else:
                    c4.metric("CAGR prix", "—")


                # Graph prix
                if show_index:
                    base = g["prix_m2_median"].iloc[0]
                    g["indice_prix"] = (g["prix_m2_median"]/base)*100 if base>0 else np.nan
                    fig = px.line(g, x="annee", y="indice_prix", markers=True,
                                  title="Indice des prix (base 100 au début)",
                                  labels={"annee":"Année","indice_prix":"Indice"},
                                  template="plotly_white")
                else:
                    fig = px.line(g, x="annee", y="prix_m2_median", markers=True,
                                  title="Prix médian (€/m²)",
                                  labels={"annee":"Année","prix_m2_median":"€/m²"},
                                  template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # Graph volume
                fig2 = px.bar(g, x="annee", y="ventes", text="ventes",
                              title="Volume de ventes (nombre d’actes)",
                              labels={"annee":"Année","ventes":"Actes"},
                              template="plotly_white")
                fig2.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# 8️⃣ Carte d’accessibilité — DVF 2024
# -----------------------------
elif menu == "Carte d’accessibilité — DVF 2024":
    import pgeocode

    st.header("🗺️ Carte — % d’appartements ≤ budget par code postal (DVF 2024)")

    PATH = "data/clean/dvf_clean_2024.csv"
    if not os.path.exists(PATH):
        st.error("❌ Fichier manquant : data/clean/dvf_clean_2024.csv")
    else:
        df = pd.read_csv(PATH, low_memory=False)

        # Nettoyage
        if "Classe pièces" not in df.columns and "Nombre pieces principales" in df.columns:
            def _piece_bucket(n):
                try:
                    n = int(n)
                except Exception:
                    return np.nan
                if n == 1:
                    return "Studio/T1 (1p)"
                if n == 2:
                    return "T2 (2p)"
                return "Autre"
            df["Classe pièces"] = df["Nombre pieces principales"].apply(_piece_bucket)

        if "Code postal" in df.columns:
            df["Code postal"] = pd.to_numeric(df["Code postal"], errors="coerce").astype("Int64")
            df["Code postal"] = df["Code postal"].astype(str).str.replace("<NA>", "", regex=False).str[:5]

        df["Type local"] = df["Type local"].astype(str).str.strip().str.title()
        df = df.dropna(subset=["Code postal", "Commune", "Valeur fonciere", "Surface reelle bati"]).copy()
        df = df[(df["prix_m2"] >= 300) & (df["prix_m2"] <= 20000)].copy()

        # Géocodeur code postal → latitude / longitude
        nomi = pgeocode.Nominatim("fr")

        def aggreg_cp_appt(dfx, budget_max=100_000, pieces_label="Studio/T1 (1p)"):
            d = dfx[dfx["Type local"] == "Appartement"].copy()
            if pieces_label != "Tous":
                d = d[d["Classe pièces"] == pieces_label]
            d["accessible"] = d["Valeur fonciere"] <= float(budget_max)

            grp = (
                d.groupby(["Code postal", "Commune"], dropna=False)
                .agg(
                    nb_ventes=("Valeur fonciere", "size"),
                    nb_access=("accessible", "sum"),
                    pct_access=("accessible", lambda s: (s.sum() / max(len(s), 1)) * 100.0),
                    med_prix_m2=("prix_m2", "median"),
                    med_prix=("Valeur fonciere", "median"),
                )
                .reset_index()
            )
            grp = grp[grp["nb_ventes"] >= 3]
            grp = grp.dropna(subset=["med_prix_m2", "med_prix"])
            grp = grp[(grp["med_prix"] >= 15000) & (grp["med_prix_m2"] >= 300)].copy()
            return grp

        def add_latlon_from_cp(df_cp):
            geo = nomi.query_postal_code(df_cp["Code postal"].tolist())
            geo = geo[["postal_code", "latitude", "longitude"]].rename(columns={"postal_code": "Code postal"})
            out = df_cp.merge(geo, on="Code postal", how="left")
            out = out.dropna(subset=["latitude", "longitude"])
            out = out[(out["longitude"].between(-6, 10)) & (out["latitude"].between(41, 52))]
            return out

        # Widgets Streamlit
        budget = st.slider("Budget maximum (€)", 30000, 200000, 100000, step=5000)
        pieces = st.selectbox("Classe de pièces", ["Studio/T1 (1p)", "T2 (2p)", "Tous"])

        # Agrégation
        agg = aggreg_cp_appt(df, budget_max=budget, pieces_label=pieces)
        pts = add_latlon_from_cp(agg)

        if pts.empty:
            st.warning("Aucune zone trouvée. Augmentez le budget ou élargissez les filtres.")
        else:
            total_cp = len(pts)
            cp_50 = (pts["pct_access"] >= 50).sum()
            med_pct = pts["pct_access"].median()
            med_pm2 = pts["med_prix_m2"].median()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Codes postaux couverts", f"{total_cp:,}".replace(",", " "))
            col2.metric("CP avec ≥50% ≤ budget", f"{cp_50:,}".replace(",", " "))
            col3.metric("Médiane % ≤ budget", f"{med_pct:.1f} %")
            col4.metric("Prix médian €/m²", f"{med_pm2:,.0f} €".replace(",", " "))

            # Carte Plotly
            size_base = 4 + 18 * np.sqrt(pts["nb_ventes"] / pts["nb_ventes"].max())
            pts["_size"] = np.clip(size_base, 6, 22)
            fig = px.scatter_mapbox(
                pts,
                lat="latitude",
                lon="longitude",
                color="pct_access",
                size="_size",
                color_continuous_scale="Blues",
                range_color=(0, 100),
                hover_name="Commune",
                hover_data={
                    "Code postal": True,
                    "pct_access": ':.1f',
                    "med_prix_m2": ':.0f',
                    "med_prix": ':.0f',
                    "nb_ventes": True,
                },
                zoom=5.5,
                height=650,
                title=f"Accessibilité ≤ {budget:,} € — Appartements / {pieces}".replace(",", " "),
            )
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)






# -----------------------------
# 6️⃣ Indice achat-location
# -----------------------------
elif menu == "Indice achat-location":
    st.header("🏡 Indice achat-location — années de loyers nécessaires")
    try:
        df_dvf["code_departement"] = df_dvf["code_departement"].astype(str).str.zfill(2)
        df_loyers["dep"] = df_loyers["dep"].astype(str).str.zfill(2)

        # Normalisation des colonnes de type de bien
        df_loyers = df_loyers[df_loyers["type_détaillé"] != "Appartement - Tous"]
        df_loyers["type_local_normalise"] = df_loyers["type_détaillé"].apply(
            lambda x: "Appartement" if "Appartement" in x else "Maison"
        )

        # Calcul des prix et loyers moyens par département et type
        prix_dept = (
            df_dvf.groupby(["code_departement", "type_local"])
            .agg(prix_m2_median=("prix_m2", "median"))
            .reset_index()
        )

        loyers_dept = (
            df_loyers.groupby(["dep", "type_détaillé", "type_local_normalise"])
            .agg(loyer_m2_moyen=("loypredm2", "mean"))
            .reset_index()
        )

        # Fusion des deux sources
        fusion = prix_dept.merge(
            loyers_dept,
            left_on=["code_departement", "type_local"],
            right_on=["dep", "type_local_normalise"],
            how="inner"
        )

        # Calcul de l'indice achat-location (années de loyers nécessaires)
        fusion["annees_loyer"] = fusion["prix_m2_median"] / (fusion["loyer_m2_moyen"] * 12)

        # Sélection du type de bien
        type_bien = st.selectbox(
            "🏘️ Type de bien",
            ["Appartement - 1 ou 2 pièces", "Appartement - 3 pièces ou plus", "Maison"]
        )
        seuil = st.slider("Seuil max (années de loyers)", 10, 35, 20, 1)

        data = fusion[(fusion["annees_loyer"] <= seuil) & (fusion["type_détaillé"] == type_bien)]

        # Vérif
        if data.empty:
            st.warning("Aucune donnée disponible pour ce type de bien.")
        else:
            fig = px.bar(
                data,
                x="code_departement",
                y="annees_loyer",
                color="annees_loyer",
                color_continuous_scale="RdYlGn_r",
                title=f"{type_bien} — Indice ≤ {seuil} ans"
            )
            fig.update_xaxes(type="category", tickangle=-45)  # ✅ Fix affichage horizontal
            st.plotly_chart(fig, use_container_width=True)

            top10 = data.nsmallest(10, "annees_loyer")[["code_departement", "annees_loyer"]]
            st.markdown("**🏆 Top 10 départements les plus rentables à l'achat (moins d’années de loyers nécessaires)**")
            st.dataframe(
                top10.style.background_gradient(subset=["annees_loyer"], cmap="RdYlGn_r", vmin=10, vmax=30)
                .format({"annees_loyer": "{:.1f} ans"})
            )
    

    except Exception as e:
        st.error(f"❌ Erreur : {e}")

