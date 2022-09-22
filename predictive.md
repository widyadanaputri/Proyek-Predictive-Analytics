# Proyek Predictive Analytics (Ardiyanti Widyadana Prastuti)
## M03

Implementasi pembelajaran modul *Machine Learning* Terapan

## Project Domain

Asuransi kesehatan atau *health insurance* merupakan salah satu jenis asuransi yang paling banyak dimiliki oleh orang-orang saat ini. Selain karena tujuan serta perlindungan yang ditawarkan, jenis asuransi yang satu ini juga merupakan yang paling dekat dengan kehidupan banyak orang.
Asuransi Kesehatan adalah jenis asuransi yang menanggung biaya pengobatan. Seseorang yang telah mengambil polis asuransi kesehatan mendapatkan perlindungan asuransi kesehatan dengan membayar sejumlah premi tertentu. Ada banyak faktor yang menentukan premi asuransi kesehatan. 
Jumlah premi untuk polis asuransi kesehatan tergantung dari orang ke orang, karena banyak faktor yang mempengaruhi jumlah premi untuk polis asuransi kesehatan. Katakanlah usia, orang muda sangat kecil kemungkinannya untuk memiliki masalah kesehatan utama dibandingkan dengan orang yang lebih tua. Dengan demikian, merawat orang yang lebih tua akan lebih mahal dibandingkan dengan yang masih muda. Itu sebabnya orang yang lebih tua diharuskan membayar premi yang tinggi dibandingkan dengan orang yang lebih muda.
Besar atau kecil tanggungan yang ditawarkan juga biasanya tergantung dengan berapa biaya premi yang harus dibayarkan setiap bulannya. Semakin besar premi yang diangsur, maka semakin besar nominal pertanggungan yang akan diberikan oleh pihak jasa asuransi.
Sama seperti usia, ada banyak faktor lain yang dapat mempengaruhi premi untuk polis asuransi kesehatan. Jadi, jumlah premi polis asuransi kesehatan tergantung pada orang ke orang karena banyak faktor yang mempengaruhi jumlah premi polis asuransi kesehatan.


## Business Understanding

#### 1. Problem statements
Dari *background* di atas dapat disimpulkan beberapa rumusan masalah di antaranya:
- Bagaimana cara menganalisis data pada asurasi kesehatan ?
- Bagaimana cara mengolah data agar dapat dilatih dengan baik oleh model ?
- Bagaimana cara membangun model yang dapat memprediksi *time series medical insurance* dengan akurasi yang baik ?
##### 2. Goals
Tujuan dari proyek ini adalah:
- Dapat memprediksi harga *premium medical insurance* dengan menggunakan model *machine learning*
- Dapat mengolah data dengan optimal agar dapat dilatih dengan baik oleh model *machine learning*
- Dapat menemukan model yang dapat memprediksi *time series medical insurance* dengan tingkat akurasi yang baik
#### 3. Solution Statements
Dari *problem statements* dan *goals* yang telah dijabarkan, berikut solusi yang dapat dilakukan:
- Melakukan analisa dengan cara menangani *missing value* pada data, kemudian mencari korelasi pada data, menangani *outlier* pada data, dan melakukan normalisasi pada data. Selain itu dilakukan eksplorasi dan pemrosesan pada data dengan memvisualisasikannya.
- Membuat model regresi untuk memprediksi harga asuransi kesehatan. Dalam hal ini akan digunakan algoritma *Support Vector Regression, K-Nearest Neighbor,* dan *Gradient Boosting Regression*.
- Melakukan *hyperparameter tuning* agar model dapat bekerja dengan performa yang terbaik.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah dataset yang diambil dari website https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction. Dataset yang digunakan berbentuk .csv dengan total 986 *records* dan 6 *columns*. Terdiri dari :
1. *Age* : usia *customer*
2. Diabetes : apakah orang tersebut memiliki kadar gula darah abnormal
3. *BloodPressureProblems* : apakah orang tersebut memiliki tingkat tekanan darah abnormal
4. *Height* : tinggi *customer*
5. *Weight* : berat *customer*
6. *PremiumPrice* : harga dari *medical insurance*

## Data Preparation
Teknik *data preparation* yang dilakukan di antaranya :
1. Menghapus kolom *'AnyTransplants', 'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'* dengan fungsi *drop* karena pada kolom-kolom yang tersebut tidak diperlukan.
2. Melakukan *splitting* dataset. Disini dataset dibagi 2 yaitu *train data* yang digunakan sebagai model *training* dan *test data* diperuntukkan validasi untuk mengecek model yang digunakan sudah akurat atau belum. Perbandingan yang umum digunakan untuk *splitting* dataset yaitu 80% *train data* dan 20% *test data*.
3. Melakukan normalisasi data yang ditujukan supaya model dapat bekerja secara optimal. Data yang dinormalisasikan yaitu *X_train* dan *X_text* dalam hal ini digunakan *library MinMaxScaler*.
4. Untuk mendeteksi outlier digunakan metode IQR (*Interquartile Range*).

## Modelling
1. *Support Vector Regression Support Vector Regression* mempunyai prinsip yang sama dengan SVM. SVM sendiri adalah salah satu algoritma *machine learning* dengan pendekatan *supervised learning* yang bekerja dengan mencari *hyperplane* atau fungsi pemisah terbaik untuk memisahkan kelas. Algoritma SVM memiliki konsep dan dasar matematis yang mapan sehingga menjadi algoritma yang populer. Sementara SVR mencari jalan yang dapat menampung sebanyak mungkin sampel di jalan. *Hyperparameter* yang digunakan pada model ini sebagai berikut :
- *C* : parameter regularisasi yang digunakan untuk menukar klasifikasi yang benar dalam hal ini dari contoh training terhadap maksimalisasi margin dengan fungsi keputusan, dengan nilai *hyperparameter* C: [0.001, 0.01, 0.1, 10, 100, 1000]
- *kernel* : berfungsi untuk menghitung kernel matriks sebelumnya, dengan nilai *hyperparameter* kernel: ['rbf']
- *gamma* : parameter yang digunakan dalam menentukan seberapan jauh pengaruh dari contoh pelatihan atau training yang mana ketika nilainya rendah berarti jauh dan jika nilainya tinggi itu mengartikan dekat, dengan nilai *hyperparameter* gamma: [0.3, 0.03, 0.003, 0.0003]

2. *Gradient Boosting Regression Gradient Boosting* merupakan algoritma *machine learning* yang menggunakan teknik *ensembel learning* dari *decision tree* untuk memprediksi nilai. *Gradient Boosting Regression* sangat mampu menangani *pattern* yang kompleks dan data ketika linear model tidak dapat menangani. Berikut *hyperparameter* yang digunakan pada model ini:
- *learning_rate* : digunakan untuk menghitung nilai koreksi bobot ketika waktu proses training yang pada umumnya nilai dari *learning rate* berkisar di antara 0 hingga 1
- *n_estimators* : jumlah dari tahapan *boosting* yang akan dilakukan, nilai tiap *hyperparameter* tiap algoritma yaitu *criterion='squared_error', learning_rate*=0.01, *n_estimators*=1000 
- *criterion* : digunakan untuk menemukan fitur serta ambang batas yang optimal dalam membagi sebuah data
Kelebihannya yaitu menangkap hubungan linear ataupun non linear pada data dan model yang dihasilkan lebih akurat. Sementara untuk kekurangannya, waktu komputasi serta desain tinggi, kemudian tingkat kesulitan tinggi ketika pemilihan model.

3. *K-Nearest Neighbors* atau KNN adalah algoritma yang berfungsi untuk melakukan klasifikasi suatu data berdasarkan data pembelajaran (*train data sets*), yang diambil dari k tetangga terdekatnya (*nearest neighbors*). Dengan k merupakan banyaknya tetangga terdekat. Dimana *n_neighbors* memiliki nilai sebesar 1 dan 10. Pada proyek ini nilai n_neighbors yang digunakan yaitu 9.
Kelebihan KNN yaitu algoritma ini mudah untuk dipahami serta diimplementasikan, *memory based approach* atau mudah beradaptasi dengan *training data* yang baru sehingga memudahkan *developer* untuk mengimplementasikan *training data* yang baru, dan *variety of distance metrics* atau adanya fleksibilitas dari user untuk menggunakan beberapa metode yang paling sesuai. Kekurangannya adalah sensitif terhadap outlier dan rentan pada fitur yang kurang informatif.


## Evaluation
Pada tahap evaluasi disini digunakan *mse* atau *mean squared error* yang mana digunakan untuk mengetahui serta mengukur seberapa dekat garis pas dengan titik data.

![image](https://user-images.githubusercontent.com/112928081/191669295-5ceb01ae-8f91-468d-9dd7-9e934119e548.png)
Keterangan :
n = jumlah titik data
Yi = nilai sesungguhnya
Yi_hat = nilai prediksi

Dari hasil perbandingan tiga model yang telah digunakan, maka didapatkan bahwa model *SVR* (Support Vector Regression)* menghasilkan performa lebih baik atau optimal dibandingkan dengan *Gradient Boosting model* dan *K-Nearest Neighbor*. Model tersebut dapat membantu untuk memprediksi asuransi kesehatan yang dibutuhkan.

|                  | *train_mse* | *test_mse* |
|------------------|-----------|----------|
| SVR              | 36083714.028427  | 38274258.620152 |
| KNN              | 12902609.86375   | 17283886.650035 |
| *GradientBoosting* | 8867749.642456   | 13164618.870484 |

Berdasarkan tabel *mse* di atas dapat dilihat bahwa algoritma SVR memiliki nilai *mse* pada *data train* sebesar 36083714.028427 dan pada *data test* sebesar 38274258.620152. Kemudian pada algoritma KNN memiliki nilai *mse* pada *data train* sebesar 12902609.86375 dan pada *data test* sebesar 17283886.650035. Dan pada algoritma *Gradient Boosting* memiliki nilai *mse* pada *data train* sebesar 8867749.642456 dan pada *data test* sebesar 13164618.870484. Sehingga, dapat disimpulkan bahwa pada proyek kali ini penggunaan model SVR menghasilkan performa yang optimal. 
Berikut ini hasil dari plot visualisasi *mse* *(mean squared error)*.

![image](https://user-images.githubusercontent.com/112928081/191668908-75dc9462-1f5c-44e2-8221-bb08c2616bb3.png)


## References
1. PT Mahir Teknologi (SMART), Semua. 2022. Kenali Hyperparameter Tuning dalam Machine Learning. https://codingstudio.id/hyperparameter-tuning/
2. PT. NOMIC INDONESIA, JOJO. 2021. *Health Insurance*: Definisi, Jenis dan Manfaat yang Ditawarkan. https://www.jojonomic.com/blog/health-insurance/
