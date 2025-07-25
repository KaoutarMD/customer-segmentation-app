import streamlit as st
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster


df = pd.read_csv("data/Mall_Customers.csv")
df_scaled = StandardScaler().fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])




st.sidebar.title("Navigation")
st.sidebar.markdown("""
    - **Exploration** : Analyse exploratoire des données.
    - **Clustering K-Means** : Implémentation du clustering K-Means.
    - **Clustering DBSCAN** : Implémentation du clustering DBSCAN.
    - **Clustering AHC** : Implémentation du clustering par hiérarchie (
    AHC).
""")
page = st.sidebar.radio("Aller à", ["Accueil", "Exploration", "Clustering K-Means", "Clustering DBSCAN", "Clustering AHC"])

if page == "Accueil":
    st.title("🧠 Segmentation de clients - Analyse non supervisée")
    st.write("Bienvenue dans cette application de segmentation de clients utilisant des techniques d'analyse non supervisée.")
    st.write("Cette application vous permet d'explorer les données du dataset Mall_Customers et d'appliquer différentes méthodes de clustering pour segmenter les clients en fonction de leurs caractéristiques.")
    st.write("Utilisez le menu latéral pour naviguer entre les différentes sections de l'application.")


elif page == "Exploration":
    st.header("📊 Analyse exploratoire des données")
    st.write("Cette section vous permet d'explorer les données du dataset Mall_Customers.")

    st.subheader("📋 Aperçu du Dataset")
    st.dataframe(df.head())

    st.subheader("🔍 Statistiques descriptives")
    st.write(df.describe())

   #Afficher des graphiques interactifs (age, income, spending score)
    st.subheader("📈 Visualisation des données")
    st.write("Visualisez la distribution des âges, des revenus et des scores de dépenses.")
    st.write("graphique de la distribution des âges")
    st.bar_chart(df['Age'])
    st.write("graphique de la distribution des revenus")
    st.bar_chart(df['Annual Income (k$)'])
    st.write("graphique de la distribution des scores de dépenses")
    st.bar_chart(df['Spending Score (1-100)'])


    # Plus de graphiques avec seaborn
    st.subheader("📊 Graphiques avancés")
    # Pour le 1er graphique avec Seaborn
    st.write("Graphique de la distribution des âges par genre")
    fig1 = plt.figure()
    sns.histplot(data=df, x='Age', hue='Gender', multiple='stack')
    st.pyplot(fig1)

    # Pour le 2e
    st.write("Graphique de la distribution des revenus annuels par genre")
    fig2 = plt.figure()
    sns.histplot(data=df, x='Annual Income (k$)', hue='Gender', multiple='stack')
    st.pyplot(fig2)

    # Pour le 3e
    st.write("Graphique de la distribution des scores de dépenses par genre")
    fig3 = plt.figure()
    sns.histplot(data=df, x='Spending Score (1-100)', hue='Gender', multiple='stack')
    st.pyplot(fig3)



elif page == "Clustering K-Means":
    st.title("Clustering K-Means")
    st.write("Cette section est dédiée au clustering K-Means.")
  
    
    n_clusters = st.slider("Choisir le nombre de clusters", 2, 10, 5)
    
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(df_scaled)
    df['KMeans'] = labels

    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=labels, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Résultat du clustering K-Means')
    st.pyplot(fig)


    score = silhouette_score(df_scaled, labels)
    st.success(f"Silhouette Score: {score:.2f}")


elif page == "Clustering DBSCAN":
    
    st.title("Clustering DBSCAN")
    st.write("Cette section est dédiée au clustering DBSCAN.")

    eps = st.slider("Choisir la valeur de eps", 0.1, 5.0, 0.5)
    min_samples = st.slider("Choisir le nombre minimum de points", 1, 20, 5)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df_scaled)
    df['DBSCAN'] = labels

    fig2, ax = plt.subplots()
    scatter = ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                     c=labels, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Résultat du clustering DBSCAN')
    st.pyplot(fig2)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    st.info(f"Nombre de clusters trouvés (hors bruit) : {n_clusters}")
    st.info(f"Nombre de points bruit : {n_noise}")

    if n_clusters >= 2:
      score = silhouette_score(df_scaled, labels)
      st.success(f"Silhouette Score : {score:.2f}")
    else:
      st.warning("Impossible de calculer le Silhouette Score (moins de 2 clusters valides).")


elif page == "Clustering AHC":
    st.title("Clustering AHC")
    st.write("Cette section est dédiée au clustering AHC.")
    method = st.selectbox("Choisir la méthode de linkage", ["single", "complete", "average", "ward"])
    Z = linkage(df_scaled, method=method)
    fig = plt.figure(figsize=(10, 6))
    dendrogram(Z)
    st.pyplot(fig)
    st.write("Le dendrogramme ci-dessus montre la hiérarchie des clusters. Vous pouvez choisir un seuil pour déterminer le nombre de clusters.")
    threshold = st.slider("Choisir le seuil pour couper le dendrogramme", 0.0, 10.0, 5.0)
    clusters = fcluster(Z, threshold, criterion='distance')
    df['AHC'] = clusters

    fig2, ax = plt.subplots()
    scatter = ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                     c=clusters, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    st.pyplot(fig2)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    st.info(f"Nombre de clusters trouvés : {n_clusters}")
    if n_clusters >= 2:
        score = silhouette_score(df_scaled, clusters)
        st.success(f"Silhouette Score : {score:.2f}")
    else:
        st.warning("Impossible de calculer le Silhouette Score (moins de 2 clusters valides).")
