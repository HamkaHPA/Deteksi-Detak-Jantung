# =============================================================================
# KONFIGURASI GLOBAL: Sistem Monitor Detak Jantung rPPG Real-time
# =============================================================================
# File ini berisi semua konstanta dan parameter konfigurasi yang digunakan
# di seluruh modul aplikasi. Menggunakan pendekatan terpusat untuk memudahkan
# penyesuaian parameter tanpa mengubah kode di banyak tempat.
# =============================================================================

import numpy as np


# =============================================================================
# KONFIGURASI VIDEO CAPTURE
# =============================================================================
# Parameter untuk pengambilan video dari webcam

CAMERA_INDEX = 0            # Indeks kamera (0 = kamera default)
FRAME_WIDTH = 640           # Lebar frame dalam piksel
FRAME_HEIGHT = 480          # Tinggi frame dalam piksel
FPS_TARGET = 30             # Target frame per second (standar webcam)


# =============================================================================
# KONFIGURASI PEMROSESAN SINYAL
# =============================================================================
# Parameter untuk buffer sinyal dan analisis frekuensi detak jantung

BUFFER_SIZE = 150           # Jumlah frame dalam buffer (~5 detik pada 30 FPS)
                            # Buffer yang lebih besar = hasil lebih stabil
                            # Buffer yang lebih kecil = respons lebih cepat

FREQ_MIN = 0.7              # Frekuensi minimum detak jantung (Hz) = 42 BPM
FREQ_MAX = 4.0              # Frekuensi maksimum detak jantung (Hz) = 240 BPM
                            # Rentang ini mencakup detak jantung normal manusia
                            # (istirahat ~60-100 BPM, olahraga ~100-180 BPM)

FILTER_ORDER = 5            # Orde filter Butterworth untuk bandpass filter
                            # Orde lebih tinggi = filter lebih tajam tapi delay lebih besar


# =============================================================================
# KONFIGURASI ICA (Independent Component Analysis)
# =============================================================================
# Parameter untuk algoritma FastICA dari scikit-learn

ICA_N_COMPONENTS = 3        # Jumlah komponen independen (sesuai dengan R, G, B)
ICA_MAX_ITER = 500          # Iterasi maksimum untuk konvergensi ICA (ditingkatkan)
ICA_TOL = 1e-3              # Toleransi konvergensi (lebih longgar untuk stabilitas)
ICA_RANDOM_STATE = 42       # Seed untuk reprodusibilitas hasil


# =============================================================================
# KONFIGURASI EULERIAN VIDEO MAGNIFICATION (EVM)
# =============================================================================
# Parameter untuk amplifikasi visual denyut nadi pada wajah

EVM_PYRAMID_LEVELS = 4      # Jumlah level pada Gaussian pyramid
                            # Level lebih banyak = detail lebih halus tapi lebih lambat

EVM_AMPLIFICATION = 50      # Faktor amplifikasi untuk perubahan warna kulit
                            # Nilai lebih tinggi = efek lebih terlihat tapi lebih noise

EVM_FREQ_LOW = 0.83         # Frekuensi bawah untuk temporal filter (~50 BPM)
EVM_FREQ_HIGH = 1.0         # Frekuensi atas untuk temporal filter (~60 BPM)
                            # Rentang sempit untuk mengurangi noise

EVM_CHROM_ATTENUATION = 1.0 # Faktor atenuasi warna (1.0 = penuh)


# =============================================================================
# KONFIGURASI DETEKSI WAJAH (Haar Cascade)
# =============================================================================
# Parameter untuk algoritma Haar Cascade dari OpenCV

FACE_SCALE_FACTOR = 1.1     # Faktor skala untuk multi-scale detection
FACE_MIN_NEIGHBORS = 5      # Minimum tetangga untuk konfirmasi deteksi
FACE_MIN_SIZE = (100, 100)  # Ukuran minimum wajah yang dideteksi (piksel)


# =============================================================================
# KONFIGURASI REGION OF INTEREST (ROI) DAHI
# =============================================================================
# Parameter untuk ekstraksi area dahi dari bounding box wajah
# Dahi dipilih karena memiliki pembuluh darah yang dekat dengan permukaan kulit
# dan relatif stabil (tidak banyak bergerak seperti area mulut)

FOREHEAD_TOP_RATIO = 0.05       # Jarak dari atas wajah (5% dari tinggi wajah)
FOREHEAD_HEIGHT_RATIO = 0.25    # Tinggi ROI dahi (25% dari tinggi wajah)
FOREHEAD_LEFT_RATIO = 0.2       # Jarak dari kiri wajah (20% dari lebar wajah)
FOREHEAD_WIDTH_RATIO = 0.6      # Lebar ROI dahi (60% dari lebar wajah)


# =============================================================================
# KONFIGURASI SKIN MASKING (YCbCr)
# =============================================================================
# Threshold untuk deteksi warna kulit menggunakan ruang warna YCbCr
# Nilai ini dioptimalkan untuk berbagai warna kulit manusia

SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)    # Batas bawah (Y, Cr, Cb)
SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8) # Batas atas (Y, Cr, Cb)


# =============================================================================
# KONFIGURASI TAMPILAN UI
# =============================================================================
# Parameter untuk rendering visualisasi pada frame output

# Dimensi area plot
PLOT_WIDTH = 300            # Lebar area plot dalam piksel
PLOT_HEIGHT = 100           # Tinggi area plot dalam piksel
PLOT_MARGIN = 10            # Margin antar elemen UI

# Warna (format BGR untuk OpenCV)
COLOR_BVP_WAVEFORM = (0, 255, 0)        # Hijau untuk sinyal BVP
COLOR_FFT_SPECTRUM = (255, 255, 0)      # Cyan untuk spektrum FFT
COLOR_HR_PEAK = (0, 0, 255)             # Merah untuk peak detak jantung
COLOR_ROI_BOX = (0, 255, 255)           # Kuning untuk bounding box ROI
COLOR_FACE_BOX = (255, 0, 0)            # Biru untuk bounding box wajah
COLOR_TEXT = (255, 255, 255)            # Putih untuk teks
COLOR_BACKGROUND = (40, 40, 40)         # Abu-abu gelap untuk background plot

# Font
FONT = 0                    # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7            # Skala ukuran font
FONT_THICKNESS = 2          # Ketebalan font


# =============================================================================
# KONFIGURASI STABILITAS & ERROR HANDLING
# =============================================================================
# Parameter untuk graceful degradation ketika kondisi tidak ideal

HR_DISPLAY_TIMEOUT = 2.0    # Waktu (detik) untuk tetap menampilkan HR terakhir
                            # ketika wajah tidak terdeteksi

MIN_SKIN_PIXELS = 100       # Minimum piksel kulit yang valid untuk pemrosesan
                            # Jika kurang dari ini, skip frame tersebut

SMOOTHING_WINDOW = 5        # Ukuran window untuk smoothing hasil HR
                            # Mengurangi fluktuasi hasil yang terlalu cepat
