import os
import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Can our algorithm guesstimate your IQ only based on MRI volumes?")
st.subheader("""We tried our best but are probably still a bit far off :) 
            \n Please let us know how we did at AWEERASEKERA@mgh.harvard.edu
""")

st.info(
        """
    **Note:** This study was peer-reviewed and published in Human Brain Mapping under the title 
    **Predictive models demonstrate age-dependent association of subcortical volumes and cognitive measures**
    \n The study used the open-source LEMON dataset containing MRI scans of 145 young adults and 48 old adults situated in the Leipzig region in Germany.
    """
    )

file = st.file_uploader("Please upload a spreadsheet with the FreeSurfer output", type=["csv", "xls", "xlsx"])

# iq_long = ['Fluid Intelligence', 'Crystallized Intelligence', 'Cognitive Flexibility', 'Working Memory']

iq_long = ['Crystallized Intelligence']
old_young = ["young", "old"]
str_feats = ['subcortical_feats', 'etiv_feats', 'all_feats']

subcort_reg = ['Caudate',
               'Putamen',
               'Thalamus-Proper',
               'Pallidum',
               'Amygdala',
               'Accumbens-area',
               'Hippocampus'
              ]

etiv_reg =  ['CorticalWhiteMatterVol',
             'TotalGrayVol',
             'CSF'
            ]

etiv = "EstimatedTotalIntraCranialVol"

def load_models():
    models_folder = "full_RF_models_saved"
    models_dict = {}
    for oy in old_young:
        models_dict[oy] = {}
        for iq in iq_long:
            models_dict[oy][iq] = {}
            for feat in str_feats:
                models_dict[oy][iq][feat] = []
                model_name = "_".join(["Full_RF", iq, feat, oy]) + ".sav"
                fullname = os.path.join(models_folder, model_name)
                loaded_model = pickle.load(open(fullname, 'rb'))
                models_dict[oy][iq][feat].append(loaded_model)
    return models_dict

models_dict = load_models()

adults_radio_list = ["Young Adults", "Old Adults"]
features_radio_list = ["Subcortical", "Cortical (GM, WM, CSF)", "Subcortical+Cortical"]

if file:
    ## read spreadsheet file
    if str(file.name).endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    # st.write(df.columns)
    ## sum up the values left and right
    for column in subcort_reg:
        df[column] = df["Left-"+column] + df["Right-"+column]
    df["CorticalWhiteMatterVol"] = df["lhCorticalWhiteMatterVol"] + df["rhCorticalWhiteMatterVol"]
    ## normalize if necessary
    if etiv in df.columns:
        st.write("Found eTIV - normalizing..")
        for column in subcort_reg+etiv_reg:
            df[column] = df[column] / df[etiv]
    else:
        st.write("eTIV not detected - brain normalization is expected in the uploaded dataset")
    ## select young vs old
    adults_type = st.radio("Are the subjects in the spreadsheet considered young or old?", adults_radio_list)
    ## select subcortival vs cortical vs all features
    features = st.radio("Features used to predict IQ", features_radio_list)
    ## perform prediction using the selected model
    # st.subheader("The used combination is..")
    # st.write(adults_type, features)
    oy_index = adults_radio_list.index(adults_type)
    feat_index = features_radio_list.index(features)
    selected_oy = old_young[oy_index]
    selected_feat = str_feats[feat_index]

    selected_model = models_dict[selected_oy]['Crystallized Intelligence'][selected_feat][0]

    if feat_index == 0:
        iq_pred = selected_model.predict(df[subcort_reg])
    elif feat_index == 1:
        iq_pred = selected_model.predict(df[etiv_reg])
    elif feat_index == 2:
        iq_pred = selected_model.predict(df[subcort_reg+etiv_reg])
    else:
        st.write("did not select any Feature radio button ?")
        iq_pred = []
    iq_pred = np.round(iq_pred, 1)
    st.subheader("Predicted IQ numbers are..")
    st.write(iq_pred)