import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------
st.set_page_config(page_title="Analyse Immobilière Étudiante", page_icon="🏠", layout="wide")

# 🌟 --- En-tête de ton portfolio ---
st.markdown("""
# 🏠 Tableau de bord immobilier interactif  
### Un projet développé par **Rayan** : visualisation et analyse des données immobilières françaises 

Ce tableau de bord interactif combine **données publiques (DVF, INSEE, loyers 2024)** et outils analytiques pour explorer :
- la faisabilité d’un achat étudiant 🧑‍🎓  
- les rendements bruts et tensions locatives 🏙️  
- la relation entre **achat et location** selon les départements 💰  

> Un projet personnel conçu pour démontrer mes compétences en **data analysis, Python et Streamlit** 🚀
""")

st.divider()

# -----------------------------
# MENU LATÉRAL
# -----------------------------
menu = st.sidebar.radio(
    "🧭 Navigation",
    [
        "Faisabilité d'achat étudiant",
        "Rendement brut minimal",
        "Répartition DVF par budget",
        "Tension locative (INSEE)",
        "Rendement par département",
        "Indice achat-location"
    ]
)

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
        df = pd.read_csv("../data/clean/dvf_clean_2024.csv")
        budget = st.slider("Budget maximum (€)", 50000, 300000, 100000, step=10000)
        d = df[df["Valeur fonciere"] <= budget]

        if d.empty:
            st.warning("Aucun bien sous ce budget.")
        else:
            d["Type_simple"] = d["Type local"].str.upper().map(
                lambda x: "Appartement" if "APPART" in x else ("Maison" if "MAISON" in x else "Autre")
            )
            d = d[(d["Nombre pieces principales"] >= 1) & (d["Nombre pieces principales"] <= 8)]
            ventes = d.groupby(["Type_simple", "Nombre pieces principales"]).size().reset_index(name="Ventes")

            fig = px.bar(
                ventes,
                x="Nombre pieces principales",
                y="Ventes",
                color="Type_simple",
                barmode="group",
                text="Ventes",
                template="plotly_white",
                title=f"Ventes ≤ {budget:,} € par taille du logement"
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), bargap=0.15)
            fig.update_xaxes(range=[0.5, 8.5])
            st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("❌ Fichier DVF manquant : ../data/clean/dvf_clean_2024.csv")

# -----------------------------
# 5️⃣ Rendement par département
# -----------------------------
elif menu == "Rendement par département":
    st.header("💶 Rendement brut — par département")
    try:
        df_dvf = pd.read_csv("../data/clean/dvf_clean_2024.csv", low_memory=False)
        df_loyers = pd.read_csv("../data/clean/loyers_clean_2024.csv", low_memory=False)

        df_dvf["Code departement"] = df_dvf["Code departement"].astype(str).str.zfill(2)
        df_dvf = df_dvf[(df_dvf["prix_m2"] > 200) & (df_dvf["prix_m2"] < 15000)]
        prix_m2_dept = (
            df_dvf.groupby(["Code departement", "Type local"])
            .agg(prix_m2_median=("prix_m2", "median"))
            .reset_index()
        )

        df_loyers["DEP"] = df_loyers["DEP"].astype(str).str.zfill(2)
        df_loyers = df_loyers[df_loyers["Type détaillé"] != "Appartement - Tous"]
        df_loyers["Type local normalisé"] = df_loyers["Type détaillé"].apply(
            lambda x: "Appartement" if "Appartement" in x else "Maison"
        )

        loyer_dept = (
            df_loyers.groupby(["DEP", "Type détaillé", "Type local normalisé"])
            .agg(loyer_m2_moyen=("loypredm2", "mean"))
            .reset_index()
        )

        fusion = prix_m2_dept.merge(
            loyer_dept,
            left_on=["Code departement", "Type local"],
            right_on=["DEP", "Type local normalisé"],
            how="inner"
        )
        fusion["rendement_brut_%"] = (fusion["loyer_m2_moyen"] * 12 / fusion["prix_m2_median"]) * 100
        fusion["rendement_brut_%"] = fusion["rendement_brut_%"].round(2)

        type_bien = st.selectbox(
            "Type de bien",
            ["Appartement - 1 ou 2 pièces", "Appartement - 3 pièces ou plus", "Maison"]
        )
        seuil = st.slider("Seuil minimal (%)", 3.0, 10.0, 6.0, 0.5)

        data = fusion[fusion["Type détaillé"] == type_bien].copy()
        if data.empty:
            st.warning("Aucune donnée disponible pour ce type de bien.")
        else:
            data = data[data["rendement_brut_%"] >= seuil].sort_values("rendement_brut_%", ascending=False)

            fig = px.bar(
                data,
                x="Code departement",
                y="rendement_brut_%",
                color="rendement_brut_%",
                color_continuous_scale="RdYlGn",
                text="rendement_brut_%",
                template="plotly_white",
                title=f"{type_bien} — Rendement ≥ {seuil:.1f} %"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

            top10 = data.nlargest(10, "rendement_brut_%")[["Code departement", "prix_m2_median", "loyer_m2_moyen", "rendement_brut_%"]]
            st.markdown("**🏆 Top 10 départements par rendement :**")
            st.dataframe(
                top10.style.background_gradient(subset=["rendement_brut_%"], cmap="RdYlGn", vmin=4, vmax=10)
                .format({
                    "prix_m2_median": "{:.0f} €",
                    "loyer_m2_moyen": "{:.2f} €/m²",
                    "rendement_brut_%": "{:.2f} %"
                })
            )

    except FileNotFoundError:
        st.error("❌ Données loyers/DVF manquantes.")

# -----------------------------
# 6️⃣ Indice achat-location
# -----------------------------
elif menu == "Indice achat-location":
    st.header("🏡 Indice achat-location — années de loyers nécessaires")
    try:
        df_dvf = pd.read_csv("../data/clean/dvf_clean_2024.csv")
        df_loyers = pd.read_csv("../data/clean/loyers_clean_2024.csv")

        df_dvf["Code departement"] = df_dvf["Code departement"].astype(str).str.zfill(2)
        df_loyers["DEP"] = df_loyers["DEP"].astype(str).str.zfill(2)

        fusion = (
            df_dvf.groupby(["Code departement", "Type local"])["prix_m2"].median().reset_index()
            .merge(
                df_loyers.groupby(["DEP", "Type détaillé"])
                .agg(loyer_m2=("loypredm2", "mean"))
                .reset_index(),
                left_on=["Code departement", "Type local"],
                right_on=["DEP", "Type détaillé"],
                how="inner"
            )
        )

        fusion["annees_loyer"] = fusion["prix_m2"] / (fusion["loyer_m2"] * 12)
        seuil = st.slider("Seuil max (années de loyers)", 10, 35, 20, 1)

        data = fusion[fusion["annees_loyer"] <= seuil].sort_values("Code departement")

        fig = px.bar(
            data,
            x="Code departement",
            y="annees_loyer",
            color="annees_loyer",
            color_continuous_scale="RdYlGn_r",
            title=f"Indice ≤ {seuil} ans",
            height=450,
        )
        fig.update_xaxes(type="category")
        fig.update_layout(
            xaxis_title="Code département",
            yaxis_title="Années de loyers nécessaires",
            margin=dict(l=20, r=20, t=50, b=40),
            bargap=0.25,
            width=900,
        )
        st.plotly_chart(fig, use_container_width=False)

        top10 = data.nsmallest(10, "annees_loyer")[["Code departement", "annees_loyer"]]
        st.markdown("**🏆 Top 10 départements où acheter est le plus rentable (moins d’années de loyers nécessaires)**")
        st.dataframe(
            top10.style.background_gradient(subset=["annees_loyer"], cmap="RdYlGn_r", vmin=10, vmax=30)
            .format({"annees_loyer": "{:.1f} ans"})
        )

    except FileNotFoundError:
        st.error("❌ Données nécessaires manquantes.")
