# ======================================================================================================================
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


# ======================================================================================================================
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def gen_ellipse(center, width, height, angle, color, label):
    ellipse = Ellipse(xy=center,
                      width=width,
                      height=height,
                      angle=angle,
                      linewidth=2, ec=color, linestyle='', label=label)
    ellipse.set_alpha(0.2)
    ellipse.set_facecolor(color)
    return ellipse


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

    # link to other pages
    st.markdown("---")
    st.markdown("**More Dashboards:**")
    st.markdown("- [Guitar Chord Explorer](https://jrbarhydt-guitarfingering-guitarfingering-zljmbo.streamlitapp.com/)")
    st.markdown("- [Raisin Types Explorer](https://jrbarhydt-raisinexplorer-raisins-c3z4pe.streamlitapp.com/)")
    st.markdown("- [I-94 Traffic Explorer](https://jrbarhydt-i94-traffic-traffic-sespds.streamlitapp.com/)")

if selected[0] == selected[1]:
    st.write("Must chose two different axes to compare.")
else:
    # kecimen ellipses from eigenvectors/values
    xy = df[df["Class"] == "Kecimen"][[selected[0], selected[1]]]
    cov = np.cov(xy.transpose())
    e_val, e_vec = np.linalg.eig(cov)
    w, h = 2 * np.sqrt(e_val)
    theta = np.rad2deg(np.arccos(e_vec[0, 0]))
    ell = gen_ellipse(xy.mean(), w, h, theta, '#3d8ac0', "Kecimen")
    ell1 = gen_ellipse(xy.mean(), 2*w, 2*h, theta, '#3d8ac0', "Kecimen")
    # besni ellipses
    xy2 = df[df["Class"] == "Besni"][[selected[0], selected[1]]]
    cov2 = np.cov(xy2.transpose())
    e_val, e_vec = np.linalg.eig(cov2)
    w, h = 2 * np.sqrt(e_val)
    theta = np.rad2deg(np.arccos(e_vec[0, 0]))
    ell2 = gen_ellipse(xy2.mean(), w, h, theta, '#ffad64', "Besni")
    ell21 = gen_ellipse(xy2.mean(), 2*w, 2*h, theta, '#ffad64', "Besni")

    # plot setup
    fig = plt.figure(facecolor="#0E1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor('#262730')
    plt.scatter(xy[selected[0]], xy[selected[1]], s=2)
    plt.scatter(xy2[selected[0]], xy2[selected[1]], s=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(color='white', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.add_patch(ell)
    ax.add_patch(ell2)
    ax.add_patch(ell1)
    ax.add_patch(ell21)

    ax.set_xlabel(selected[0], color="white")
    ax.set_ylabel(selected[1], color="white")
    ax.legend(["Kecimen", "Besni"])

    st.pyplot(fig, use_container_width=True)
