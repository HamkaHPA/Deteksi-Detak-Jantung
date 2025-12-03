# Sistem Monitor Detak Jantung rPPG Real-time
## Metode ICA (Independent Component Analysis) & Eulerian Video Magnification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-Educational-yellow" alt="License">
</p>

---

## 📋 Deskripsi Proyek

Proyek ini merupakan implementasi sistem **Remote Photoplethysmography (rPPG)** untuk tugas kuliah **Sistem Teknologi Multimedia**. Sistem ini mampu mengukur detak jantung secara **non-kontak** hanya menggunakan webcam standar, tanpa memerlukan sensor fisik yang ditempelkan pada tubuh.

### Apa itu rPPG?

rPPG adalah teknik pengukuran sinyal fisiologis secara remote menggunakan kamera. Prinsip kerjanya:

1. **Photoplethysmography (PPG)** tradisional mengukur perubahan volume darah menggunakan sensor optik yang ditempelkan pada kulit (seperti pulse oximeter di jari)
2. **Remote PPG (rPPG)** melakukan hal yang sama tetapi dari jarak jauh menggunakan kamera video
3. Perubahan volume darah menyebabkan perubahan warna kulit yang **sangat halus** (tidak terlihat mata telanjang)
4. Dengan algoritma pemrosesan sinyal yang tepat, perubahan ini dapat dideteksi dan dikonversi menjadi detak jantung

### Fitur Utama

- ✅ **Ekstraksi sinyal BVP** menggunakan algoritma **ICA (Independent Component Analysis)**
- ✅ **Visualisasi denyut nadi** pada wajah dengan **Eulerian Video Magnification (EVM)**
- ✅ **Real-time processing** dengan framerate webcam standar (30 FPS)
- ✅ **User Interface** interaktif dengan plot waveform dan spektrum FFT
- ✅ **Arsitektur modular** dengan separation of concerns

---

## 🔬 Perbandingan Metode

### Tabel Perbandingan: Demo Kelas (POS) vs Proyek Ini (ICA + EVM)

| Aspek | Demo Kelas (POS) | Proyek Ini (ICA + EVM) |
|:------|:-----------------|:-----------------------|
| **Algoritma Utama** | POS (Plane-Orthogonal-to-Skin) | FastICA (Independent Component Analysis) |
| **ROI (Region of Interest)** | Single (full face) | Multi-region (dahi + skin masking YCbCr) |
| **Window Processing** | Fixed single window | Sliding window dengan buffer dinamis |
| **Motion Handling** | ❌ Tidak ada penanganan | ✅ ICA memisahkan komponen gerakan sebagai noise |
| **Noise Separation** | ❌ Tidak ada pemisahan noise | ✅ Blind Source Separation otomatis |
| **FPS Handling** | Fixed (hardcoded) | Adaptive dengan FPS_TARGET konfigurabel |
| **Threading** | Single thread | Single thread (optimized pipeline) |
| **Visualisasi** | Text overlay sederhana | Sidebar panel dengan grafik BVP & FFT real-time |
| **Visualisasi Denyut** | ❌ Tidak ada | ✅ Eulerian Video Magnification (EVM) |
| **Logging/Debug** | Print statements | Print statements + status bar informatif |
| **Error Recovery** | Crash on disconnect | ✅ Graceful handling dengan auto-reset |
| **Akurasi Estimasi** | ⚠️ Hanya kondisi ideal | ✅ Robust terhadap variasi pencahayaan |

### Perbandingan Teknis Algoritma

| Fitur Teknis | POS (Demo Kelas) | ICA (Proyek Ini) |
|:-------------|:-----------------|:-----------------|
| **Prinsip Kerja** | Proyeksi sinyal ke bidang ortogonal terhadap skin-tone | Blind Source Separation untuk memisahkan sumber independen |
| **Input** | RGB dari satu ROI | RGB dari ROI dahi dengan skin masking |
| **Output** | 1 sinyal pulse | 3 komponen independen (dipilih yang paling pulse-like) |
| **Filtering** | Bandpass sederhana | Butterworth bandpass + FFT analysis |
| **Estimasi HR** | Peak counting atau FFT basic | FFT dengan analisis spektral full |
| **Kompleksitas** | O(n) | O(n²) untuk ICA decomposition |
| **Kelebihan** | Ringan, cepat | Robust, akurat, adaptive |
| **Kelemahan** | Sensitif noise & cahaya | Lebih berat komputasi |

### Mengapa ICA Lebih Baik?

**Basic Green Channel:**
```
Sinyal_Terukur = Sinyal_BVP + Noise_Cahaya + Noise_Gerakan + Noise_Sensor
```
Semua komponen tercampur dan sulit dipisahkan.

**ICA (Independent Component Analysis):**
```
[Komponen_1]   [Mixing_Matrix]   [Sumber_1 (BVP)]
[Komponen_2] = [    ...      ] × [Sumber_2 (Cahaya)]
[Komponen_3]   [    ...      ]   [Sumber_3 (Gerakan)]
```
ICA menemukan `Mixing_Matrix` invers sehingga sumber-sumber independen dapat dipisahkan.

### Mengapa EVM Penting?

**Eulerian Video Magnification** adalah teknik pemrosesan video dari MIT yang dapat:
1. Memperbesar perubahan warna yang tidak terlihat mata
2. Memvisualisasikan aliran darah di bawah kulit
3. Memberikan **feedback visual** yang intuitif kepada pengguna
4. Merupakan fitur **multimedia** yang membedakan proyek ini dari demo kelas

---

## 📁 Struktur Proyek

```
MULMETDETAK/
├── run.py                 # Program utama (entry point)
├── requirements.txt       # Daftar dependensi Python
├── README.md              # Dokumentasi (file ini)
└── src/
    ├── __init__.py        # Inisialisasi package
    ├── config.py          # Konstanta dan konfigurasi global
    ├── face_utils.py      # Deteksi wajah dan ekstraksi ROI
    ├── ica_processor.py   # Algoritma ICA dan analisis FFT
    ├── evm_processor.py   # Eulerian Video Magnification
    └── ui_renderer.py     # Rendering antarmuka pengguna
```

### Penjelasan Setiap Modul

| Modul | Fungsi Utama |
|-------|--------------|
| `config.py` | Menyimpan semua konstanta: ukuran buffer, rentang frekuensi HR, parameter EVM, dll. |
| `face_utils.py` | Deteksi wajah (Haar Cascade), ekstraksi ROI dahi, skin masking (YCbCr) |
| `ica_processor.py` | Implementasi FastICA untuk ekstraksi sinyal, Butterworth bandpass filter, FFT untuk estimasi HR |
| `evm_processor.py` | Gaussian pyramid, temporal filtering, amplifikasi warna untuk visualisasi denyut |
| `ui_renderer.py` | Rendering waveform BVP, spektrum FFT, display HR, bounding box, status bar |
| `run.py` | Main loop yang mengorkestrasi seluruh sistem |

---

## 🚀 Cara Instalasi

### Prasyarat
- Python 3.8 atau lebih baru
- Webcam (internal laptop atau USB external)
- Sistem operasi: Windows, macOS, atau Linux

### Langkah Instalasi

1. **Clone atau download proyek ini**
   ```bash
   cd path/to/your/folder
   ```

2. **Buat virtual environment (opsional tapi disarankan)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan program**
   ```bash
   python run.py
   ```

---

## 🎮 Cara Penggunaan

### Persiapan

1. **Pencahayaan**: Pastikan ruangan memiliki pencahayaan yang cukup dan merata. Hindari backlight (cahaya dari belakang).

2. **Posisi**: Duduk dengan wajah menghadap kamera secara langsung (frontal). Jaga jarak sekitar 50-70 cm dari kamera.

3. **Stabilitas**: Usahakan untuk tetap diam selama pengukuran. Gerakan akan mengganggu sinyal.

### Hotkeys

| Tombol | Fungsi |
|--------|--------|
| `Q` atau `ESC` | Keluar dari program |
| `R` | Reset buffer sinyal (mulai ulang pengukuran) |
| `E` | Toggle EVM (aktif/nonaktif) |
| `+` atau `=` | Tingkatkan amplifikasi EVM |
| `-` atau `_` | Kurangi amplifikasi EVM |

### Interpretasi Tampilan

1. **Bounding Box Biru**: Area wajah yang terdeteksi
2. **Bounding Box Kuning (corners)**: ROI dahi yang digunakan untuk ekstraksi sinyal
3. **Plot BVP (hijau)**: Sinyal Blood Volume Pulse - setiap puncak = satu detak
4. **Plot FFT (cyan)**: Spektrum frekuensi - garis merah menunjukkan peak HR
5. **Display HR**: Detak jantung dalam BPM (Beats Per Minute)
6. **Status Bar**: Informasi status pemrosesan

---

## ⚙️ Konfigurasi Lanjutan

Parameter dapat diubah di file `src/config.py`:

### Parameter Utama

```python
BUFFER_SIZE = 150        # Jumlah frame dalam buffer (150 = 5 detik pada 30 FPS)
FPS_TARGET = 30          # Target frame rate
FREQ_MIN = 0.7           # Frekuensi minimum HR (0.7 Hz = 42 BPM)
FREQ_MAX = 4.0           # Frekuensi maksimum HR (4.0 Hz = 240 BPM)
```

### Parameter EVM

```python
EVM_PYRAMID_LEVELS = 4   # Level Gaussian pyramid (lebih tinggi = lebih detail)
EVM_AMPLIFICATION = 50   # Faktor amplifikasi (20-100 disarankan)
EVM_FREQ_LOW = 0.83      # Batas bawah frequency band (~50 BPM)
EVM_FREQ_HIGH = 1.0      # Batas atas frequency band (~60 BPM)
```

## ⚠️ Disclaimer

Sistem ini dibuat untuk **tujuan edukasi** dalam konteks tugas kuliah Sistem Teknologi Multimedia. 

**PERINGATAN:**
- Hasil pengukuran **TIDAK** untuk diagnosis medis
- Akurasi bergantung pada kondisi pencahayaan, stabilitas, dan kualitas webcam
- Untuk pengukuran detak jantung yang akurat, gunakan perangkat medis yang tersertifikasi

---

## 👨‍🎓 Informasi Tugas

- **Mata Kuliah**: Sistem Teknologi Multimedia
- **Topik**: Remote Photoplethysmography (rPPG) dengan ICA dan EVM
- **Bahasa Pemrograman**: Python 3
- **Framework/Library**: OpenCV, NumPy, SciPy, Scikit-learn

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan edukasi. Silakan gunakan dan modifikasi sesuai kebutuhan pembelajaran.

---

<p align="center">
  <i>Dibuat untuk tugas kuliah Sistem Teknologi Multimedia</i>
</p>
