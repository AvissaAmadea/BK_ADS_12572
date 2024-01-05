import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle


# 1) Pengumpulan Data
# Dataset dengan nama file "hungarian.data" bersumber dari link berikut:
# https://archive.ics.uci.edu/dataset/45/heart+disease

# 2) Menelaah Data --> membaca dataset
with open("data/hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]

# menyimpan dataset ke dalam variabel data, 
# mengambil data menjadi baris-baris yang memiliki panjang 76 karakter 
# secara berturut-turut dalam kelompok 10 baris.
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

# membuat variabel df berisi data tersusun dalam struktur tabel use DataFrame
df = pd.DataFrame.from_records(data)

# Mengambil semua baris dan kolom dari DataFrame df, kecuali kolom terakhir.
df = df.iloc[:, :-1]

# Menghapus kolom pertama dari DataFrame df.
df = df.drop(df.columns[0], axis=1)

# Mengonversi semua nilai dalam DataFrame df menjadi tipe data float.
df = df.astype(float)


# 3) Validasi Data
#Mengganti semua nilai -9.0 dalam DataFrame df 
# dengan nilai NaN (Not a Number) dari NumPy.
df.replace(-9.0, np.NaN, inplace=True)


# 4) Menentukan Object Data
# Membuat DataFrame baru df_selected yang berisi kolom-kolom yang dipilih 
# dari DataFrame df dengan menggunakan indeks kolom yang spesifik 
# ([1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]).
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

# mengganti nama kolom sesuai dengan 14 nama kolom yg ada pada deskripsi target
column_mapping = {
  2: 'age',
  3: 'sex',
  8: 'cp',
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}
# rename column
df_selected.rename(columns=column_mapping, inplace=True)


# 5) Membersihkan Data
# terdapat fitur yang hampir 90% datanya memiliki nilai null 
# sehingga perlu menghapus fitur tsb menggunakan fungsi drop
# menghapus data dari 3 column
columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

# mengisi field yang masih terisi null dengan mean di setiap kolomnya
# memilih kolom dan menghapus nilai nullnya
meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

# mengambil dan mengubah nilai dataset menjadi float
meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

# mengubah dan membulatkan nilai mean
meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

# mengisi nilai null menjadi nilai mean di setiap kolomnya
fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach':meanthalach,
  'exang':meanexang,
  'restecg':meanRestCG
}

df_clean = df_selected.fillna(value=fill_values)

# menghapus data yang mangandung duplikasi 
df_clean.drop_duplicates(inplace=True)


# 6) Konstruksi Data
# memisahkan antara fitur dan target lalu simpan ke dalam variabel baru
X = df_clean.drop("target", axis=1)
y = df_clean['target']

# Karena persebaran jumlah target tidak seimbang maka diseimbangkan 
# menggunakan Metode Oversampling SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


# 7) Membuat Model --> XGBoost
# Membuat model dg nama file xgb_model.pkl
# import pickle
# with open('model/xgb_model.pkl', 'wb') as file:
#     pickle.dump(xgb_model, file)

# membuka file xgb di dalam folder model
model = pickle.load(open("model/xgb_model.pkl", 'rb'))

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y

# -------------------------------------------------------------------------------------------------
# STREAMLIT
# -------------------------------------------------------------------------------------------------

# judul halaman streamlit
st.set_page_config(
  page_title = "Hungarian Heart Disease",
  page_icon = ":heart:"
)

st.markdown(
    """
    <div style="display: flex; align-items: center; border-radius: 10px;">
        <img src="https://raw.githubusercontent.com/AvissaAmadea/BK_ADS_12572/main/jantung.png" alt="Heart Image" width="60" style="margin-right: 7px;">
        <h1 style="margin-bottom: 0; color: white;"><span style="color: white;">Penyakit Jantung Hungarian</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)
# st.title("Penyakit Jantung Hungarian")

st.write(f"**_Akurasi Model_** :  :green[**{accuracy}**]%")
# st.write("")

# membuat 2 halaman terpisah di halaman utama
tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

# fungsi sidebar
with tab1:
  st.sidebar.header("Sidebar untuk **Input Data**")

  # input data di sidebar
  age = st.sidebar.number_input(label=":blue[**Umur**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
  st.sidebar.write(f":orange[Min] nilai: :orange[**{df_final['age'].min()}**], :red[Max] nilai: :red[**{df_final['age'].max()}**]")
  st.sidebar.write("")

  sex_sb = st.sidebar.selectbox(label=":blue[**Jenis Kelamin**]", options=["Laki-laki", "Perempuan"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Laki-laki":
    sex = 1
  elif sex_sb == "Perempuan":
    sex = 0
  # -- Value 0: Perempuan
  # -- Value 1: Laki-laki

  cp_sb = st.sidebar.selectbox(label=":blue[**Tipe Nyeri Dada**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic

  trestbps = st.sidebar.number_input(label=":blue[**Tekanan darah istirahat** (mm Hg pada saat masuk ke rumah sakit)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
  st.sidebar.write(f":orange[Min] nilai: :orange[**{df_final['trestbps'].min()}**], :red[Max] nilai: :red[**{df_final['trestbps'].max()}**]")
  st.sidebar.write("")

  chol = st.sidebar.number_input(label=":blue[**Jumlah Kolesterol dalam darah** (mg/dl)]", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())
  st.sidebar.write(f":orange[Min] nilai: :orange[**{df_final['chol'].min()}**], :red[Max] nilai: :red[**{df_final['chol'].max()}**]")
  st.sidebar.write("")

  fbs_sb = st.sidebar.selectbox(label=":blue[**Kadar Gula dalam darah > 120 mg/dl?**]", options=["Tidak", "Ya"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "Tidak":
    fbs = 0
  elif fbs_sb == "Ya":
    fbs = 1
  # -- Value 0: tidak
  # -- Value 1: ya

  restecg_sb = st.sidebar.selectbox(label=":blue[**Hasil Resting Electrocardiographic**]", options=["Normal", "Mengalami kelainan gelombang ST-T", "Menunjukkan hipertrofi ventrikel kiri"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Mengalami kelainan gelombang ST-T":
    restecg = 1
  elif restecg_sb == "Menunjukkan hipertrofi ventrikel kiri":
    restecg = 2
  # -- Value 0: Normal
  # -- Value 1: Mengalami kelainan gelombang ST-T (Inversi gelombang T dan/atau elevasi atau depresi ST > 0,05 mV)
  # -- Value 2: Menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri berdasarkan kriteria Estes

  thalach = st.sidebar.number_input(label=":blue[**Denyut jantung maksimum mencapai**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
  st.sidebar.write(f":orange[Min] nilai: :orange[**{df_final['thalach'].min()}**], :red[Max] nilai: :red[**{df_final['thalach'].max()}**]")
  st.sidebar.write("")

  exang_sb = st.sidebar.selectbox(label=":blue[**Angina akibat olahraga?**]", options=["Tidak", "Ya"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "Tidak":
    exang = 0
  elif exang_sb == "Ya":
    exang = 1
  # -- Value 0: Tidak
  # -- Value 1: Ya

  oldpeak = st.sidebar.number_input(label=":blue[**Depresi ST disebabkan oleh olahraga dibandingkan istirahat**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
  st.sidebar.write(f":orange[Min] nilai: :orange[**{df_final['oldpeak'].min()}**], :red[Max] nilai: :red[**{df_final['oldpeak'].max()}**]")
  st.sidebar.write("")

  # inputan data dimasukkan ke dalam variabel data
  data = {
    'Umur': age,
    'Jenis Kelamin': sex_sb,
    'Tipe Nyeri Dada': cp_sb,
    'RPB': f"{trestbps} mm Hg",
    'Kolesterol Serum': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Resting ECG': restecg_sb,
    'Denyut jantung maksimum': thalach,
    'Angina akibat olahraga?': exang_sb,
    'ST depression': oldpeak,
  }

  # memasukkan data menjadi DataFrame
  preview_df = pd.DataFrame(data, index=['input'])

  # menampilkan data inputan user ke st
  st.header("Inputan User menjadi DataFrame")
  st.write("")
  st.dataframe(preview_df.iloc[:, :6])
  st.write("")
  st.dataframe(preview_df.iloc[:, 6:])
  st.write("")

  # Variabel result diatur sebagai string ":violet[-]".
  result = ":violet[-]"

  predict_btn = st.button("**Prediksi**", type="primary")

  st.write("")
  if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]

    # Melakukan prediksi menggunakan model yang sudah dilatih sebelumnya
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    # menampilkan proses prediksi
    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    # menetapkan hasil prediksi
    if prediction == 0:
      result = ":green[**Sehat Wal Afiat**]"
    elif prediction == 1:
      result = ":yellow[**Penyakit Jantung level 1**]"
    elif prediction == 2:
      result = ":orange[**Penyakit Jantung level 2**]"
    elif prediction == 3:
      result = ":red[**Penyakit Jantung level 3**]"
    elif prediction == 4:
      result = ":red[**Penyakit Jantung level 4**]"

  st.write("")
  st.write("")
  st.subheader("Hasil Prediksi:")
  st.subheader(result)

with tab2:
  st.header("Predict multiple data:")

  # Membuat Contoh CSV dari Data Awal dari lima baris pertama dari DataFrame df_final kecuali kolom terakhir
  sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download Contoh Dataset CSV", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  # Mengunggah dan Memprediksi dari File CSV yang Diunggah
  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload file CSV", type='csv')

  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    # melakukan prediksi menggunakan model machine learning (model.predict)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Sehat Wal Afiat"
      elif prediction == 1:
        result = "Penyakit Jantung level 1"
      elif prediction == 2:
        result = "Penyakit Jantung level 2"
      elif prediction == 3:
        result = "Penyakit Jantung level 3"
      elif prediction == 4:
        result = "Penyakit Jantung level 4"
      result_arr.append(result)

    # Menampilkan Hasil Prediksi dengan membagi menjadi 2 kolom: result dan parameter
    uploaded_result = pd.DataFrame({'Hasil Prediksi': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)