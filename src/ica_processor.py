# =============================================================================
# MODUL ICA PROCESSOR: Ekstraksi Sinyal BVP dengan Independent Component Analysis
# =============================================================================
# Modul ini bertanggung jawab untuk:
# 1. Menerapkan algoritma FastICA untuk memisahkan sinyal independen
# 2. Melakukan filtering sinyal dengan Butterworth bandpass filter
# 3. Menghitung detak jantung menggunakan analisis FFT
# =============================================================================

import numpy as np
from scipy.signal import butter, filtfilt, detrend
from sklearn.decomposition import FastICA
from typing import Tuple, Optional

# Import konfigurasi
from src.config import (
    ICA_N_COMPONENTS, ICA_MAX_ITER, ICA_TOL, ICA_RANDOM_STATE,
    FREQ_MIN, FREQ_MAX, FILTER_ORDER, FPS_TARGET
)


def _create_bandpass_filter(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Membuat koefisien filter Butterworth bandpass.
    
    Filter bandpass digunakan untuk:
    1. Menghilangkan komponen DC (frekuensi sangat rendah)
    2. Menghilangkan noise frekuensi tinggi
    3. Hanya meloloskan frekuensi dalam rentang detak jantung (0.7-4 Hz)
    
    Butterworth filter dipilih karena:
    - Memiliki respons frekuensi yang flat di passband
    - Tidak ada ripple di passband maupun stopband
    - Cocok untuk aplikasi biomedis
    
    Parameters
    ----------
    lowcut : float
        Frekuensi cutoff bawah dalam Hz
    highcut : float
        Frekuensi cutoff atas dalam Hz
    fs : float
        Sampling frequency dalam Hz
    order : int
        Orde filter (semakin tinggi = transisi semakin tajam)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Koefisien filter (b, a) untuk digunakan dengan filtfilt()
    
    Notes
    -----
    Frekuensi Nyquist = fs / 2
    Frekuensi normalized = f_cutoff / f_nyquist
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Pastikan frekuensi normalized dalam range (0, 1)
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    # Pastikan low < high
    if low >= high:
        low = high * 0.5
    
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _apply_bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 5
) -> np.ndarray:
    """
    Menerapkan filter bandpass pada sinyal.
    
    Menggunakan filtfilt() untuk zero-phase filtering:
    - Filter diterapkan maju dan mundur
    - Tidak ada phase shift (delay) pada output
    - Penting untuk preservasi timing sinyal biomedis
    
    Parameters
    ----------
    signal : np.ndarray
        Sinyal input 1D
    lowcut : float
        Frekuensi cutoff bawah dalam Hz
    highcut : float
        Frekuensi cutoff atas dalam Hz
    fs : float
        Sampling frequency dalam Hz
    order : int
        Orde filter
    
    Returns
    -------
    np.ndarray
        Sinyal yang sudah difilter
    """
    b, a = _create_bandpass_filter(lowcut, highcut, fs, order)
    
    # Minimum panjang sinyal untuk filtfilt
    # filtfilt membutuhkan minimal 3 * max(len(a), len(b)) sampel
    min_length = 3 * max(len(a), len(b))
    
    if len(signal) < min_length:
        return signal  # Return tanpa filter jika sinyal terlalu pendek
    
    try:
        filtered = filtfilt(b, a, signal)
        return filtered
    except ValueError:
        # Jika filtfilt gagal, return sinyal asli
        return signal


def apply_ica_extraction(
    rgb_buffer: np.ndarray,
    fs: float = FPS_TARGET
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Menerapkan Independent Component Analysis (ICA) pada buffer sinyal RGB.
    
    Konsep ICA dalam konteks rPPG:
    ================================
    Sinyal RGB yang direkam dari kulit adalah campuran dari beberapa sumber:
    1. Sinyal Blood Volume Pulse (BVP) - yang kita cari
    2. Perubahan pencahayaan ambient
    3. Gerakan (motion artifacts)
    4. Noise sensor kamera
    
    ICA bekerja dengan asumsi bahwa sumber-sumber ini independen secara statistik.
    Dengan memisahkan sinyal menjadi komponen independen, kita dapat mengisolasi
    komponen BVP dari noise lainnya.
    
    Algoritma FastICA:
    ==================
    1. Preprocessing: centering (mean removal) dan whitening
    2. Iterasi untuk memaksimalkan non-Gaussianity
    3. Menggunakan fungsi negentropy (logcosh) sebagai ukuran non-Gaussianity
    
    Parameters
    ----------
    rgb_buffer : np.ndarray
        Buffer sinyal RGB dengan shape (n_samples, 3)
        Setiap baris adalah [R_mean, G_mean, B_mean] dari satu frame
    fs : float
        Sampling frequency dalam Hz (default: FPS_TARGET)
    
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[int]]
        - sources: Array komponen independen dengan shape (n_samples, 3)
        - best_idx: Index komponen dengan sinyal BVP terkuat (0, 1, atau 2)
        
        Mengembalikan (None, None) jika:
        - Buffer terlalu pendek
        - ICA gagal konvergen
    
    Notes
    -----
    - Komponen terbaik dipilih berdasarkan power spektral di rentang HR
    - Detrending dilakukan untuk menghilangkan drift DC
    """
    # Validasi input
    if rgb_buffer is None or len(rgb_buffer) < 30:
        # Minimal 30 sampel (1 detik pada 30 FPS) untuk analisis yang bermakna
        return None, None
    
    # Pastikan shape correct (n_samples, 3)
    if rgb_buffer.ndim != 2 or rgb_buffer.shape[1] != 3:
        return None, None
    
    # Step 1: Preprocessing - Detrend untuk menghilangkan drift linear
    # Drift bisa terjadi karena perubahan pencahayaan gradual
    try:
        rgb_detrended = detrend(rgb_buffer, axis=0, type='linear')
    except ValueError:
        rgb_detrended = rgb_buffer.copy()
    
    # Step 2: Normalisasi - zero mean, unit variance untuk setiap channel
    # Ini membantu konvergensi ICA
    mean = np.mean(rgb_detrended, axis=0)
    std = np.std(rgb_detrended, axis=0)
    std[std == 0] = 1  # Hindari division by zero
    rgb_normalized = (rgb_detrended - mean) / std
    
    # Step 3: Terapkan FastICA
    try:
        ica = FastICA(
            n_components=ICA_N_COMPONENTS,
            algorithm='parallel',
            whiten='unit-variance',
            fun='logcosh',  # Fungsi untuk estimasi negentropy
            max_iter=ICA_MAX_ITER,
            tol=ICA_TOL,
            random_state=ICA_RANDOM_STATE
        )
        
        # Suppress convergence warning - hasil masih bisa digunakan
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # fit_transform mengembalikan sumber independen
            sources = ica.fit_transform(rgb_normalized)
        
    except Exception as e:
        # ICA bisa gagal jika sinyal terlalu noise atau singular
        print(f"[ICA] Gagal: {e}")
        return None, None
    
    # Step 4: Pilih komponen dengan power terkuat di rentang HR
    best_idx = _select_best_component(sources, fs)
    
    return sources, best_idx


def _select_best_component(
    sources: np.ndarray,
    fs: float
) -> int:
    """
    Memilih komponen ICA dengan sinyal BVP terkuat.
    
    Metode seleksi:
    1. Hitung FFT untuk setiap komponen
    2. Hitung total power dalam rentang frekuensi HR (0.7-4 Hz)
    3. Pilih komponen dengan power tertinggi di rentang tersebut
    
    Rasionalisasi:
    - Sinyal BVP memiliki komponen frekuensi dominan di rentang HR
    - Komponen noise cenderung memiliki spektrum yang lebih tersebar
    
    Parameters
    ----------
    sources : np.ndarray
        Komponen independen dari ICA dengan shape (n_samples, n_components)
    fs : float
        Sampling frequency
    
    Returns
    -------
    int
        Index komponen terbaik (0, 1, atau 2)
    """
    n_samples, n_components = sources.shape
    
    # Hitung frekuensi untuk FFT
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
    
    # Mask untuk rentang frekuensi HR
    hr_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    
    best_idx = 0
    best_power = 0
    
    for i in range(n_components):
        # FFT dari komponen
        fft_vals = np.abs(np.fft.rfft(sources[:, i]))
        
        # Total power dalam rentang HR
        hr_power = np.sum(fft_vals[hr_mask] ** 2)
        
        if hr_power > best_power:
            best_power = hr_power
            best_idx = i
    
    return best_idx


def compute_heart_rate_fft(
    bvp_signal: np.ndarray,
    fs: float = FPS_TARGET,
    apply_filter: bool = True
) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    """
    Menghitung detak jantung dari sinyal BVP menggunakan analisis FFT.
    
    Proses komputasi:
    =================
    1. (Opsional) Apply bandpass filter untuk menghilangkan noise
    2. Hitung FFT (Fast Fourier Transform)
    3. Cari peak frekuensi dalam rentang detak jantung valid (0.7-4 Hz)
    4. Konversi frekuensi ke BPM: HR = f_peak × 60
    
    FFT (Fast Fourier Transform):
    =============================
    FFT mengubah sinyal dari domain waktu ke domain frekuensi.
    - Input: sinyal BVP (amplitudo vs waktu)
    - Output: spektrum (amplitudo vs frekuensi)
    
    Detak jantung akan muncul sebagai peak pada frekuensi tertentu.
    Misalnya, 60 BPM = 1 Hz, 90 BPM = 1.5 Hz.
    
    Parameters
    ----------
    bvp_signal : np.ndarray
        Sinyal Blood Volume Pulse (1D array)
    fs : float
        Sampling frequency dalam Hz (default: FPS_TARGET)
    apply_filter : bool
        Apakah menerapkan bandpass filter sebelum FFT
    
    Returns
    -------
    Tuple[Optional[float], np.ndarray, np.ndarray]
        - heart_rate: Detak jantung dalam BPM, atau None jika gagal
        - freqs: Array frekuensi dari FFT
        - fft_magnitude: Magnitude spektrum FFT
    
    Notes
    -----
    - Resolusi frekuensi = fs / n_samples
    - Untuk buffer 150 sampel pada 30 FPS: resolusi = 30/150 = 0.2 Hz = 12 BPM
    """
    # Validasi input
    if bvp_signal is None or len(bvp_signal) < 30:
        return None, np.array([]), np.array([])
    
    # Step 1: Bandpass filter (opsional tapi sangat disarankan)
    if apply_filter:
        bvp_filtered = _apply_bandpass_filter(
            bvp_signal,
            FREQ_MIN,
            FREQ_MAX,
            fs,
            FILTER_ORDER
        )
    else:
        bvp_filtered = bvp_signal
    
    # Step 2: Hitung FFT
    n_samples = len(bvp_filtered)
    
    # rfft untuk sinyal real (lebih efisien dari fft)
    fft_vals = np.fft.rfft(bvp_filtered)
    fft_magnitude = np.abs(fft_vals)
    
    # Frekuensi yang bersesuaian dengan setiap bin FFT
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
    
    # Step 3: Cari peak dalam rentang HR valid
    # Buat mask untuk rentang frekuensi detak jantung
    valid_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    
    if not np.any(valid_mask):
        return None, freqs, fft_magnitude
    
    valid_freqs = freqs[valid_mask]
    valid_magnitude = fft_magnitude[valid_mask]
    
    # Cari index dengan magnitude tertinggi
    peak_idx = np.argmax(valid_magnitude)
    peak_freq = valid_freqs[peak_idx]
    
    # Step 4: Konversi frekuensi ke BPM
    # HR (BPM) = frekuensi (Hz) × 60 (detik/menit)
    heart_rate = peak_freq * 60.0
    
    return heart_rate, freqs, fft_magnitude


def process_rgb_buffer(
    rgb_buffer: np.ndarray,
    fs: float = FPS_TARGET
) -> Tuple[Optional[float], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Pipeline lengkap untuk memproses buffer RGB menjadi detak jantung.
    
    Menggabungkan semua langkah:
    1. ICA extraction
    2. Seleksi komponen terbaik
    3. FFT analysis
    4. HR computation
    
    Parameters
    ----------
    rgb_buffer : np.ndarray
        Buffer sinyal RGB dengan shape (n_samples, 3)
    fs : float
        Sampling frequency
    
    Returns
    -------
    Tuple dengan 4 elemen:
        - heart_rate: Detak jantung dalam BPM atau None
        - bvp_signal: Sinyal BVP yang diekstrak atau None
        - freqs: Array frekuensi FFT
        - fft_magnitude: Magnitude spektrum FFT
    
    Example
    -------
    >>> hr, bvp, freqs, fft = process_rgb_buffer(rgb_buffer, fs=30)
    >>> if hr is not None:
    >>>     print(f"Detak jantung: {hr:.1f} BPM")
    """
    # Step 1: ICA extraction
    sources, best_idx = apply_ica_extraction(rgb_buffer, fs)
    
    if sources is None or best_idx is None:
        return None, None, np.array([]), np.array([])
    
    # Step 2: Ambil komponen BVP terbaik
    bvp_signal = sources[:, best_idx]
    
    # Step 3: Hitung detak jantung dengan FFT
    heart_rate, freqs, fft_magnitude = compute_heart_rate_fft(bvp_signal, fs)
    
    return heart_rate, bvp_signal, freqs, fft_magnitude
