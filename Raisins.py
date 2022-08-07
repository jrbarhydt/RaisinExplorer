# ======================================================================================================================
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import numpy as np

# ======================================================================================================================
# Functions
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# Initialization
# ----------------------------------------------------------------------------------------------------------------------

data_filepath = "Raisin_Dataset.csv"

# init dataframe
df = pd.read_csv(data_filepath)
headers = list(df.columns)
headers.remove("Class")


# ======================================================================================================================
# Streamlit Generation
# ----------------------------------------------------------------------------------------------------------------------

# generate title
st.title("Raisin Explorer")
st.markdown("_Select features to plot from sidebar on the left._")

# generate sidebar
with st.sidebar:
    # init sidebar
    st.title("Select features to compare")

    # init and generate columns
    selected = ['Area', 'MajorAxisLength']
    st.markdown("---")
    st.header("Besni")
    selected[0] = st.radio(label="", options=headers, key=0, index=0)
    st.markdown("---")
    st.header("Kecimen")
    selected[1] = st.radio(label="", options=headers, key=0, index=1)

if selected[0] == selected[1]:
    st.write("Must chose two different axes to compare.")
else:
    pca = PCA(n_components=2)
    x, y = list(df[df["Class"] == "Kecimen"][selected[0]]), list(df[df["Class"] == "Kecimen"][selected[1]])
    pca.fit(list(zip(x, y)))
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=lambda_[0] * 2, height=lambda_[1] * 2,
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  linewidth=2, ec='#3d8ac0', linestyle='', label="Kecimen")
    ell.set_alpha(0.2)
    ell.set_facecolor('#3d8ac0')
    ell1 = Ellipse(xy=(np.mean(x), np.mean(y)),
                   width=lambda_[0] * 4, height=lambda_[1] * 4,
                   angle=np.rad2deg(np.arccos(v[0, 0])),
                   linewidth=2, ec='#3d8ac0', linestyle='', label="Kecimen")
    ell1.set_alpha(0.2)
    ell1.set_facecolor('#3d8ac0')

    x2, y2 = list(df[df["Class"] == "Besni"][selected[0]]), list(df[df["Class"] == "Besni"][selected[1]])
    pca.fit(list(zip(x2, y2)))
    cov2 = np.cov(x2, y2)
    lambda_, v = np.linalg.eig(cov2)
    lambda_ = np.sqrt(lambda_)
    ell2 = Ellipse(xy=(np.mean(x2), np.mean(y2)),
                   width=lambda_[0] * 2, height=lambda_[1] * 2,
                   angle=np.rad2deg(np.arccos(v[0, 0])),
                   linewidth=2, ec='#ffad64', linestyle='', label="Besni")
    ell2.set_alpha(0.2)
    ell2.set_facecolor('#ffad64')
    ell21 = Ellipse(xy=(np.mean(x2), np.mean(y2)),
                    width=lambda_[0] * 4, height=lambda_[1] * 4,
                    angle=np.rad2deg(np.arccos(v[0, 0])),
                    linewidth=2, ec='#ffad64', linestyle='', label="Besni")
    ell21.set_alpha(0.2)
    ell21.set_facecolor('#ffad64')

    fig = plt.figure(facecolor="#0E1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor('#262730')
    plt.scatter(x, y, s=2)
    plt.scatter(x2, y2, s=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(color='white', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.add_patch(ell)
    ax.add_patch(ell2)
    ax.add_patch(ell1)
    ax.add_patch(ell21)
    ax.set_xlim([min(min(x), min(x2)), max(max(x, x2))])
    ax.set_ylim([min(min(y), min(y2)), max(max(y, y2))])
    ax.set_xlabel(selected[0], color="white")
    ax.set_ylabel(selected[1], color="white")
    ax.legend(["Kecimen", "Besni"])

    st.pyplot(fig, use_container_width=True)
