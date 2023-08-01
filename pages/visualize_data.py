import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import streamlit as st
from umap.umap_ import UMAP

def visualize_features():
    """ Visualize features """
    # 1. Flatten tensors and stack data
    flat_data = [t.sum(dim=0).numpy() for t in st.session_state.features.values()]
    stacked_data = np.vstack(flat_data)

    # 2. Normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(stacked_data)

    # 3. Apply PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(normalized_data)

    # 4. Perform KMeans clustering on PCA results
    kmeans = KMeans(n_clusters=3)  # adjust as needed
    labels = kmeans.fit_predict(pca_result)

    # 5. Create a DataFrame for PCA visualization
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
    df_pca['label'] = labels

    # 6. Create a 3D scatter plot for PCA
    fig_pca = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='label')
    st.plotly_chart(fig_pca)

    # 7. Apply t-SNE and KMeans
    tsne_pipeline = Pipeline([('scaling', StandardScaler()), 
                            ('t-SNE', TSNE(n_components=2, perplexity=30, n_iter=3000))])
    tsne_result = tsne_pipeline.fit_transform(stacked_data)
    labels_tsne = kmeans.fit_predict(tsne_result)

    # 8. Create a DataFrame for t-SNE visualization
    df_tsne = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])
    df_tsne['label'] = labels_tsne

    # 9. Create a 2D scatter plot for t-SNE
    fig_tsne = px.scatter(df_tsne, x='Dim1', y='Dim2', color='label')
    st.plotly_chart(fig_tsne)

    # 10. Apply UMAP and KMeans
    umap_pipeline = Pipeline([('scaling', StandardScaler()), 
                            ('umap', UMAP(n_neighbors=5, min_dist=0.3))])
    umap_result = umap_pipeline.fit_transform(stacked_data)
    labels_umap = kmeans.fit_predict(umap_result)

    # 11. Create a DataFrame for UMAP visualization
    df_umap = pd.DataFrame(umap_result, columns=['Dim1', 'Dim2'])
    df_umap['label'] = labels_umap

    # 12. Create a 2D scatter plot for UMAP
    fig_umap = px.scatter(df_umap, x='Dim1', y='Dim2', color='label')
    st.plotly_chart(fig_umap)

def visualize_embeddings():
    for filename, embeddings in st.session_state.embeddings.items():
        st.write(filename)
        st.write(embeddings)
    """ Visualize embeddings """
    # 1. Flatten embeddings and create list of labels
    flat_data = []
    labels = []
    # Get the list of names of the files
    names = list(st.session_state.embeddings.keys())

    for filename, embeddings in st.session_state.embeddings.items():
        flat_data.append(embeddings)
        labels.append(filename)
    stacked_data = np.vstack(flat_data)

    # 2. Normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(stacked_data)

    # 3. Apply PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(normalized_data)

    # 4. Perform KMeans clustering on PCA results
    kmeans = KMeans(n_clusters=3)  # adjust as needed
    labels = kmeans.fit_predict(pca_result)

    # 5. Create a DataFrame for PCA visualization
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
    df_pca['label'] = labels  
    df_pca['names'] = names  # use filenames as labels

    # Export the pandas dataframe to a csv file
    df_pca.to_csv('pca.csv', index=False)

    st.markdown("""**The 3D plot you see below is a visual representation of these voice prints. 
                Each point in the plot is a unique voice print extracted from an audio file. 
                The closer the points, the more similar the voice characteristics they represent. 
                This is achieved using a technique called 'Principal Component Analysis' or PCA, 
                which simplifies the complex data while retaining the essential differences between voices. 
                The colors represent different clusters or groups of similar voices identified by our model.**""")


    # 6. Create a 3D scatter plot for PCA
    fig_pca = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='label', hover_name='names')  # display filename on hover
    st.plotly_chart(fig_pca)

    # 10. Apply UMAP and KMeans
    umap_pipeline = Pipeline([('scaling', StandardScaler()), 
                            ('umap', UMAP(n_neighbors=5, min_dist=0.3))])
    umap_result = umap_pipeline.fit_transform(stacked_data)
    labels_umap = kmeans.fit_predict(umap_result)

    # 11. Create a DataFrame for UMAP visualization
    df_umap = pd.DataFrame(umap_result, columns=['Dim1', 'Dim2'])
    df_umap['label'] = labels_umap
    df_umap['names'] = names  # use filenames as labels

    # Export the pandas dataframe to a csv file
    df_umap.to_csv('umap.csv', index=False)

    st.markdown("""**This 2D plot is another view of these voice prints, created using a method called 'UMAP'. 
                This method also brings similar voice prints closer together, but it is more advanced and capable
                of handling more complex patterns. Again, the colors represent different groups of similar voices. 
                By looking at these plots, we can see how well our model is able to distinguish between different 
                speakers based on their unique voice prints.**""")
    
    # 12. Create a 2D scatter plot for UMAP
    fig_umap = px.scatter(df_umap, x='Dim1', y='Dim2', color='label', hover_name='names', color_discrete_map="reds")  # display filename on hover
    st.plotly_chart(fig_umap)

    # Create a button to return to the main visualization page
    if st.button('Return to main page', type='primary', use_container_width=True):
        st.session_state.visual_page = 'visual_home'
        st.experimental_rerun()

def app():
    st.markdown("""**The audio files we are working with are transformed into what we call 'embeddings'. 
                Think of these embeddings as unique fingerprints for each speaker's voice. They capture the unique characteristics 
                of a person's speech such as tone, pitch, accent, and more. By analyzing these embeddings, we can differentiate speakers, 
                even when they're saying the same words. This unique 'voice print' can be utilized in various applications, including speaker recognition, 
                personalized voice assistants, and even in security for voice-based authentication systems.**""")
    
    st.markdown('---')

    st.markdown("""**The 3D plot you see below is a visual representation of these voice prints. 
                Each point in the plot is a unique voice print extracted from an audio file. 
                The closer the points, the more similar the voice characteristics they represent. 
                This is achieved using a technique called 'Principal Component Analysis' or PCA, 
                which simplifies the complex data while retaining the essential differences between voices. 
                The colors represent different clusters or groups of similar voices identified by our model.**""")
    
    # Read in the pca csv file
    df_pca = pd.read_csv('pca.csv')
    # Create a 3D scatter plot for PCA
    fig_pca = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='label', hover_name='names',
                            color_continuous_scale="viridis", title="PCA Analysis of Speaker Embeddings")  # display filename on hover
    
    st.plotly_chart(fig_pca)

    st.markdown("""**From here, we can allow the user to generate a "voice print" on the blockchain,
                issue NFTs based on their voice print, train a model using this voice print, etc.**""")

 
    visualize_embeddings_button = st.button("Mint your voice print", type='primary', use_container_width=True)
    train_model_button = st.button("Train a model using your voice print", type='primary', use_container_width=True)
    mint_nft_button = st.button("Mint an NFT using your voice print", type='primary', use_container_width=True)
    if visualize_embeddings_button:
        st.session_state.visual_page = 'visualize_embeddings'
        st.experimental_rerun()

if st.session_state.visual_page == 'visual_home':
    app()
elif st.session_state.visual_page == 'visualize_embeddings':
    visualize_embeddings()