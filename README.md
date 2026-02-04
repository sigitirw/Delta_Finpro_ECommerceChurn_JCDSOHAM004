# Delta_Finpro_ECommerceChurn_JCDSOHAM004
# Customer Churn Prediction – E-Commerce

**Team Delta JCDSOHAM004:**
- Medina Alifia
- Sigit Irwanto
- Salsabila Adinda

## Deskripsi Proyek

Proyek ini bertujuan membangun model *machine learning* untuk memprediksi **customer churn** pada sektor e-commerce serta menerjemahkan hasil model menjadi **insight bisnis dan rekomendasi strategi retensi**.

Model dikembangkan sebagai **early warning system** untuk mengidentifikasi pelanggan berisiko churn secara proaktif dengan fokus utama pada minimisasi *False Negative*.

---

## Ruang Lingkup Analisis

Analisis dalam notebook mencakup:

1. *Business Understanding* dan perumusan masalah churn
2. *Exploratory Data Analysis (EDA)*
3. *Data preprocessing* dan *feature engineering*
4. Pemodelan *machine learning*
5. Evaluasi model berbasis metrik yang relevan secara bisnis
6. Interpretasi model menggunakan *feature importance* dan SHAP
7. Simulasi dampak bisnis dan rekomendasi retensi

---

## Pendekatan Pemodelan

- Jenis masalah: *Binary classification* (Churn / Non-Churn)
- Model yang dievaluasi:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model final: **XGBoost Classifier**
- Optimasi model menggunakan **F2-Score** untuk memprioritaskan *recall* pada kelas churn

Pendekatan ini dipilih karena biaya kehilangan pelanggan (*False Negative*) dinilai lebih tinggi dibandingkan biaya intervensi pada pelanggan non-churn.

---

## Evaluasi Model

Evaluasi model dalam notebook dilakukan menggunakan:

- Confusion Matrix
- Precision, Recall, dan F2-Score
- Analisis implikasi bisnis (Cost–Benefit Simulation)

Model final menunjukkan kemampuan tinggi dalam menangkap pelanggan churn aktual dengan jumlah kesalahan prediksi yang minimal.

---

## Interpretabilitas Model

Untuk memastikan hasil model dapat digunakan sebagai dasar pengambilan keputusan, dilakukan:

- Analisis *feature importance* dari model XGBoost
- Analisis SHAP untuk:
  - Interpretasi global kontribusi fitur
  - Interpretasi lokal pada level individual pelanggan

Hasil interpretasi digunakan untuk menghubungkan prediksi model dengan pola perilaku pelanggan dan risiko churn.

---

## Insight Bisnis Utama

Berdasarkan hasil analisis dan interpretasi model dalam notebook:

- Pelanggan dengan **tenure rendah** memiliki risiko churn lebih tinggi
- **Complain** merupakan indikator risiko churn yang sangat kuat
- Pola perilaku seperti frekuensi transaksi dan jarak transaksi terakhir berkontribusi signifikan terhadap churn
- Terdapat perbedaan risiko churn berdasarkan karakteristik demografis dan wilayah

Insight ini digunakan sebagai dasar penyusunan strategi retensi yang lebih terarah.

---

## Limitasi Model

Limitasi model yang diidentifikasi dalam notebook meliputi:

- Ketergantungan pada pola data historis
- Optimasi F2-score yang meningkatkan potensi *False Positive*
- Representasi variabel *Complain* yang bersifat biner
- Fitur perilaku yang bersifat agregat, bukan time-series
- Model bersifat prediktif dan tidak bersifat kausal
- Generalisasi model antar segmen pelanggan masih terbatas

---

## Tujuan Penggunaan

Model dan analisis dalam notebook ini ditujukan sebagai:

- Sistem pendukung keputusan (*decision support system*)
- Alat bantu prioritisasi strategi retensi pelanggan
- Dasar analisis risiko churn berbasis data

Model tidak dimaksudkan sebagai satu-satunya dasar pengambilan keputusan tanpa pertimbangan bisnis lanjutan.
readme_customer_churn_prediction_ml_shap.md

- Link Tableau : 

1). Dashboard: https://public.tableau.com/views/FinalTeamDelta-MedinaSigitSalsa-OnlineEcommerce/Dashboard12?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

2). Story: https://public.tableau.com/views/FinalStoryTeamDelta-MedinaSigitSalsa-OnlineEcommerce/ChurnFlowChart?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

- Link Streamlit : https://deltafinproecommercechurnjcdsoham004-macvgkdkdtaudqeyknc78b.streamlit.app
