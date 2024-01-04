import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle

# membaca dataset
with open("data/hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]

# menyimpan dataset ke dalam variabel data, mengambil data menjadi baris-baris yang memiliki panjang 76 karakter secara berturut-turut dalam kelompok 10 baris.
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

# membuat variabel df berisi data yang disusun dalam struktur tabel use DataFrame
df = pd.DataFrame.from_records(data)

# Mengambil semua baris dan kolom dari DataFrame df, kecuali kolom terakhir.
df = df.iloc[:, :-1]

# Menghapus kolom pertama dari DataFrame df.
df = df.drop(df.columns[0], axis=1)

# Mengonversi semua nilai dalam DataFrame df menjadi tipe data float.
df = df.astype(float)

#Mengganti semua nilai -9.0 dalam DataFrame df dengan nilai NaN (Not a Number) dari NumPy.
df.replace(-9.0, np.NaN, inplace=True)

# Membuat DataFrame baru df_selected yang berisi kolom-kolom yang dipilih dari DataFrame df dengan menggunakan indeks kolom yang spesifik ([1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]).
df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

# membuat column mapping untuk merename column dari dataset
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

# menghapus data dari 3 column
columns_to_drop = ['ca', 'slope','thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

# mengisi field yang masih terisi null dengan mean di setiap kolomnya
meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

meanTBPS = round(meanTBPS.mean())
meanChol = round(meanChol.mean())
meanfbs = round(meanfbs.mean())
meanthalach = round(meanthalach.mean())
meanexang = round(meanexang.mean())
meanRestCG = round(meanRestCG.mean())

# mengubah nilai null menjadi nilai mean di setiap kolomnya
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

X = df_clean.drop("target", axis=1)
y = df_clean['target']

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

model = pickle.load(open("model/xgb_model.pkl", 'rb'))

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y
