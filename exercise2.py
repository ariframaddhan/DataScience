import streamlit as st
import pandas as pd
import pickle
import time
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""# Latihan DQLAB""")

add_selectitem = st.sidebar.selectbox("Want to open about?", ("Iris species!", "Heart Disease!"))

def heart():
    st.write("""
    This app predicts the **Heart Disease**

    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML.
    """)

    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1, 4, 2)
            if cp == 1:
                wcp = "Nyeri dada tipe angina"
            elif cp == 2:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien", wcp)

            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Kemiringan segmen ST pada EKG", 0, 2, 1)
            oldpeak = st.sidebar.slider("Penurunan ST segmen", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
            sex = 0 if sex == "Perempuan" else 1
            age = st.sidebar.slider("Usia", 29, 77, 30)

            data = {
                'cp': cp,
                'thalach': thalach,
                'slope': slope,
                'oldpeak': oldpeak,
                'exang': exang,
                'ca': ca,
                'thal': thal,
                'sex': sex,
                'age': age
            }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()

    # ⬇️ Prediction only happens if 'Predict!' is clicked
    if st.sidebar.button('Predict!'):
        st.write("Input data:")
        st.write(input_df)

        try:
            with open("output_decision_tree.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return

        prediction = loaded_model.predict(input_df)
        result = 'No Heart Disease' if prediction[0] == 0 else 'Yes Heart Disease'

        st.subheader('Prediction:')
        with st.spinner('Wait for it...'):
            time.sleep(2)
            st.success(f"Prediction: {result}")

# Placeholder for Iris model
def iris():
    st.write("🔬 Iris model not implemented yet.")

if add_selectitem == "Iris species!":
    iris()
elif add_selectitem == "Heart Disease!":
    heart()
