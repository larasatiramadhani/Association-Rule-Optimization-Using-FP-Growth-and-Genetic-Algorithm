# Association-Rule-Optimization-Using-FP-Growth-and-Genetic-Algorithm
This repository implements association rule mining using FP-Growth optimized with a Genetic Algorithm, focusing on improving rule quality based on lift and novelty metrics. Developed for academic research on culinary transaction data.
Siap. Di bawah ini **README versi rapi, formal, dan siap dikumpulkan** (cocok untuk repo skripsi / dosen / penguji).
Bahasanya **jelas untuk khalayak umum**, tapi **tetap teknis & akademik**.

---

## Association Rule Mining dengan FP-Growth dan Genetic Algorithm Berbasis Novelty & Lift

---

## 1. Gambaran Umum Program

Program ini bertujuan untuk **menemukan aturan asosiasi (association rules)** dari data transaksi penjualan dengan pendekatan **hybrid**:

1. **FP-Growth**
   Digunakan untuk mengekstraksi *frequent itemsets* secara efisien tanpa membangkitkan kandidat.

2. **Genetic Algorithm (GA)**
   Digunakan untuk mengoptimasi dan mengeksplorasi aturan asosiasi berdasarkan:

   * **Confidence**
   * **Lift**
   * **Novelty (keunikan aturan)**

Pendekatan ini dirancang untuk:

* Menghindari aturan yang redundan
* Menghasilkan aturan yang **informatif, bernilai bisnis, dan tidak monoton**
* Cocok untuk **analisis pola pembelian konsumen**

---

## 2. Data yang Diperlukan

Program membutuhkan **2 file Excel (.xlsx)**:

---

### 📄 1. Data Transaksi (`dtl_jual.xlsx`)

Berisi detail transaksi penjualan.

**Kolom wajib:**

| Nama Kolom | Deskripsi        |
| ---------- | ---------------- |
| `nonota`   | ID transaksi     |
| `kodebrg`  | Kode produk      |
| `jumlah`   | Jumlah pembelian |
| `harga`    | Harga produk     |

📌 Contoh:

| nonota | kodebrg | jumlah | harga |
| ------ | ------- | ------ | ----- |
| T001   | A01     | 2      | 15000 |
| T001   | B02     | 1      | 8000  |

---

### 📄 2. Data Menu / Produk (`deskripsi menu.xlsx`)

Berisi metadata produk.

**Kolom wajib:**

| Nama Kolom  | Deskripsi        |
| ----------- | ---------------- |
| `kode`      | Kode produk      |
| `nama`      | Nama produk      |
| `Deskripsi` | Deskripsi produk |

📌 Contoh:

| kode | nama        | Deskripsi           |
| ---- | ----------- | ------------------- |
| A01  | nasi goreng | nasi goreng spesial |
| B02  | es teh      | minuman teh dingin  |

---

📌 **Relasi data:**
`kodebrg (transaksi)` ↔ `kode (menu)`

---

## 3. Cara Menggunakan Program

### 3.1. Instalasi Library

Pastikan Python ≥ **3.9**

```bash
pip install pandas numpy scikit-learn mlxtend openpyxl
```

---

### 3.2. Struktur Folder

```
project/
│
├── main.py
├── Data/
│   ├── dtl_jual.xlsx
│   └── deskripsi menu.xlsx
```

---

### 3.3. Menjalankan Program

Jalankan melalui terminal:

```bash
python main.py
```

Atau dengan path custom:

```bash
python main.py --transaksi Data/dtl_jual.xlsx --menu Data/deskripsi menu.xlsx
```

---

### 3.4. Output

Program akan menampilkan **tabel aturan asosiasi** dengan kolom:

| Kolom      | Deskripsi                    |
| ---------- | ---------------------------- |
| antecedent | Item sebelum (jika membeli…) |
| consequent | Item sesudah (maka membeli…) |
| confidence | Tingkat kepercayaan aturan   |
| lift       | Kekuatan asosiasi            |
| novelty    | Tingkat keunikan aturan      |

📌 Output sudah diurutkan berdasarkan **Lift tertinggi**.

---

## 4. Parameter Penting yang Bisa Disesuaikan

Parameter dapat diubah di bagian **SETTING PARAMETER**:

```python
minsup = 0.01          # Minimum support FP-Growth
max_k = 4              # Maksimum ukuran itemset / rule
population_size = 100  # Ukuran populasi GA
MIN_CONF = 0.2         # Minimum confidence rule
MIN_LIFT = 1.0         # Minimum lift rule
TV = 0.5               # Threshold novelty
CR = 0.9               # Crossover rate
MR = 0.9               # Mutation rate
```
---

### 📌 Penjelasan Singkat Parameter

| Parameter         | Fungsi                                       |
| ----------------- | -------------------------------------------- |
| `minsup`          | Mengontrol seberapa sering item harus muncul |
| `max_k`           | Membatasi kompleksitas aturan                |
| `population_size` | Jumlah kandidat aturan GA                    |
| `MIN_CONF`        | Menyaring aturan lemah                       |
| `MIN_LIFT`        | Menjamin aturan bermakna                     |
| `TV`              | Menjaga keunikan aturan                      |
| `CR`              | Intensitas crossover                         |
| `MR`              | Intensitas eksplorasi aturan                 |

---

## 5. Metodologi Singkat

1. Data transaksi diproses menjadi list item per transaksi
2. FP-Growth menghasilkan frequent itemsets
3. Semua kemungkinan aturan dibentuk
4. Genetic Algorithm mengoptimasi aturan
5. Novelty digunakan sebagai fungsi fitness
6. Aturan terbaik ditampilkan sebagai hasil akhir

