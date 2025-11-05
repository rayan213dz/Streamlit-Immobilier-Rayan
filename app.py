import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------
st.set_page_config(page_title="Analyse Immobilière Étudiante", page_icon="🏠", layout="wide")

# 🌟 --- En-tête ---
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
        "Rendement par département",
        "Indice achat-location"
    ]
)

# -----------------------------
# CHARGEMENT DES DONNÉES
# -----------------------------
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, low_memory=False)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

try:
    # ⚙️ Fichiers Google Drive
    df_insee = load_data("https://drive.google.com/uc?id=1JLN-q2a4HG2n2wiz3kUu_Jju5-tWtweW")   # INSEE
    df_loyers = load_data("https://drive.google.com/uc?id=1uV0wnR5jYIm9HES_mrEmue-lq8fem9vJ")  # Loyers
    df_dvf = load_data("https://drive.google.com/uc?id=1L9fRQqocd-JVUitI6e9_oA_pA0Y16XqQ")    # DVF

    st.sidebar.success("✅ Données chargées depuis Google Drive")

except Exception as e:
    st.sidebar.error(f"❌ Erreur de chargement : {e}")

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
# 6️⃣ Indice achat-location
# -----------------------------
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

