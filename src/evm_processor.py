# =============================================================================
# MODUL EVM PROCESSOR: Eulerian Video Magnification untuk Visualisasi Denyut
# =============================================================================
# Modul ini mengimplementasikan teknik Eulerian Video Magnification (EVM)
# yang dikembangkan oleh MIT CSAIL untuk memperbesar perubahan warna kulit
# yang tidak terlihat oleh mata telanjang, sehingga denyut nadi dapat
# divisualisasikan secara real-time.
#
# Referensi Paper:
# Wu, H.-Y., et al. "Eulerian Video Magnification for Revealing Subtle
# Changes in the World." ACM Transactions on Graphics, 2012.
# =============================================================================

import cv2
import numpy as np
from collections import deque
from scipy.signal import butter, lfilter
from typing import Tuple, Optional, List

# Import konfigurasi
from src.config import (
    EVM_PYRAMID_LEVELS, EVM_AMPLIFICATION,
    EVM_FREQ_LOW, EVM_FREQ_HIGH,
    FPS_TARGET
)


class EulerianMagnifier:
    """
    Kelas untuk melakukan Eulerian Video Magnification secara real-time.
    
    Konsep Dasar EVM:
    =================
    EVM adalah teknik pemrosesan video yang dapat memperbesar perubahan
    kecil dalam video yang tidak terlihat oleh mata manusia. Berbeda dengan
    metode optik flow (Lagrangian), EVM bekerja dengan menganalisis perubahan
    intensitas piksel dari waktu ke waktu pada lokasi tetap (Eulerian).
    
    Pipeline EVM:
    =============
    1. Spatial Decomposition: Membangun Gaussian pyramid untuk memisahkan
       informasi spasial pada berbagai skala
    2. Temporal Filtering: Menerapkan bandpass filter pada setiap piksel
       untuk mengisolasi frekuensi yang menarik (denyut nadi: 0.83-1.0 Hz)
    3. Amplification: Memperbesar sinyal yang sudah difilter
    4. Reconstruction: Menggabungkan kembali pyramid untuk menghasilkan
       video yang sudah di-magnify
    
    Implementasi Simplified:
    ========================
    Untuk performa real-time, implementasi ini menggunakan pendekatan
    yang disederhanakan:
    - Gaussian pyramid (bukan Laplacian) untuk kecepatan
    - IIR filter untuk temporal filtering (bukan ideal bandpass)
    - Buffer circular untuk efisiensi memori
    
    Attributes
    ----------
    pyramid_levels : int
        Jumlah level pada Gaussian pyramid
    amplification : float
        Faktor amplifikasi untuk perubahan warna
    freq_low : float
        Frekuensi cutoff bawah untuk temporal filter (Hz)
    freq_high : float
        Frekuensi cutoff atas untuk temporal filter (Hz)
    fps : float
        Frame rate video
    buffer_size : int
        Ukuran buffer untuk temporal filtering
    
    Example
    -------
    >>> evm = EulerianMagnifier(amplification=50)
    >>> for frame in video_frames:
    >>>     magnified = evm.process_frame(frame)
    >>>     cv2.imshow("EVM", magnified)
    """
    
    def __init__(
        self,
        pyramid_levels: int = EVM_PYRAMID_LEVELS,
        amplification: float = EVM_AMPLIFICATION,
        freq_low: float = EVM_FREQ_LOW,
        freq_high: float = EVM_FREQ_HIGH,
        fps: float = FPS_TARGET,
        buffer_size: int = 30
    ):
        """
        Inisialisasi EulerianMagnifier dengan parameter yang ditentukan.
        
        Parameters
        ----------
        pyramid_levels : int
            Jumlah level pada Gaussian pyramid (default: 4)
            Level lebih banyak = detail lebih halus tapi lebih lambat
        amplification : float
            Faktor amplifikasi (default: 50)
            Nilai lebih tinggi = efek lebih terlihat tapi lebih noise
        freq_low : float
            Frekuensi cutoff bawah dalam Hz (default: 0.83 Hz ≈ 50 BPM)
        freq_high : float
            Frekuensi cutoff atas dalam Hz (default: 1.0 Hz ≈ 60 BPM)
        fps : float
            Frame rate video (default: 30)
        buffer_size : int
            Ukuran buffer temporal (default: 30 frame ≈ 1 detik)
        """
        self.pyramid_levels = pyramid_levels
        self.amplification = amplification
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.fps = fps
        self.buffer_size = buffer_size
        
        # Buffer untuk menyimpan pyramid frames
        # Menggunakan deque untuk efisiensi penambahan/penghapusan
        self._pyramid_buffer: List[deque] = []
        
        # Filter state untuk IIR filtering
        self._filter_state: List[dict] = []
        
        # Koefisien filter Butterworth
        self._b, self._a = self._create_temporal_filter()
        
        # Flag untuk menandakan sudah diinisialisasi
        self._initialized = False
        
        # Frame terakhir untuk fallback
        self._last_frame: Optional[np.ndarray] = None
    
    def _create_temporal_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Membuat koefisien filter Butterworth bandpass untuk temporal filtering.
        
        Filter bandpass digunakan untuk mengisolasi frekuensi denyut nadi
        dan menghilangkan komponen DC (pencahayaan statis) serta noise
        frekuensi tinggi (gerakan cepat).
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Koefisien filter (b, a)
        """
        nyquist = 0.5 * self.fps
        low = self.freq_low / nyquist
        high = self.freq_high / nyquist
        
        # Pastikan frekuensi normalized dalam range valid
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))
        
        if low >= high:
            high = low + 0.1
        
        # Orde 2 untuk respons yang smooth dan stabil
        b, a = butter(2, [low, high], btype='band')
        return b, a
    
    def _build_gaussian_pyramid(
        self,
        frame: np.ndarray
    ) -> List[np.ndarray]:
        """
        Membangun Gaussian pyramid dari frame.
        
        Gaussian pyramid adalah representasi multi-resolusi dari gambar:
        - Level 0: Gambar asli
        - Level 1: Downsampled 2x (blur + subsample)
        - Level 2: Downsampled 4x dari asli
        - dst.
        
        Setiap level merepresentasikan informasi spasial pada skala berbeda.
        Level bawah = detail halus, level atas = struktur kasar.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame input BGR
        
        Returns
        -------
        List[np.ndarray]
            List of pyramid levels, dari resolusi penuh ke terendah
        """
        pyramid = [frame.astype(np.float32)]
        
        current = frame.astype(np.float32)
        for _ in range(self.pyramid_levels - 1):
            # pyrDown: Gaussian blur + downsampling 2x
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        return pyramid
    
    def _reconstruct_from_pyramid(
        self,
        pyramid: List[np.ndarray],
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Merekonstruksi gambar dari Gaussian pyramid.
        
        Untuk simplified EVM, kita hanya menggunakan level tertentu
        (biasanya level tengah) dan meng-upscale ke resolusi asli.
        
        Parameters
        ----------
        pyramid : List[np.ndarray]
            Gaussian pyramid
        original_shape : Tuple[int, int]
            Shape (height, width) dari gambar asli
        
        Returns
        -------
        np.ndarray
            Gambar yang direkonstruksi
        """
        # Gunakan level tengah untuk keseimbangan detail dan kecepatan
        level_idx = min(2, len(pyramid) - 1)
        reconstructed = pyramid[level_idx]
        
        # Upscale ke resolusi asli
        target_h, target_w = original_shape[:2]
        reconstructed = cv2.resize(
            reconstructed,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        return reconstructed
    
    def _initialize_buffers(self, pyramid: List[np.ndarray]) -> None:
        """
        Menginisialisasi buffer dan filter state untuk setiap level pyramid.
        
        Dipanggil saat frame pertama diproses untuk menyiapkan struktur
        data yang diperlukan untuk temporal filtering.
        
        Parameters
        ----------
        pyramid : List[np.ndarray]
            Gaussian pyramid dari frame pertama
        """
        self._pyramid_buffer = []
        self._filter_state = []
        
        for level in pyramid:
            # Buffer circular untuk menyimpan frame pada level ini
            buffer = deque(maxlen=self.buffer_size)
            buffer.append(level.copy())
            self._pyramid_buffer.append(buffer)
            
            # State untuk IIR filter (menyimpan nilai sebelumnya)
            state = {
                'x_prev': [np.zeros_like(level, dtype=np.float32) for _ in range(2)],
                'y_prev': [np.zeros_like(level, dtype=np.float32) for _ in range(2)]
            }
            self._filter_state.append(state)
        
        self._initialized = True
    
    def _apply_temporal_filter_iir(
        self,
        current: np.ndarray,
        level_idx: int
    ) -> np.ndarray:
        """
        Menerapkan IIR bandpass filter pada satu level pyramid.
        
        IIR (Infinite Impulse Response) filter dipilih karena:
        - Lebih efisien untuk real-time (tidak perlu menyimpan semua sampel)
        - Dapat diimplementasikan secara rekursif
        - Latency rendah dibanding FIR filter
        
        Implementasi menggunakan Direct Form II:
        y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] 
               - a[1]*y[n-1] - a[2]*y[n-2]
        
        Parameters
        ----------
        current : np.ndarray
            Frame current pada level ini
        level_idx : int
            Index level pyramid
        
        Returns
        -------
        np.ndarray
            Frame yang sudah difilter secara temporal
        """
        state = self._filter_state[level_idx]
        b, a = self._b, self._a
        
        # Direct Form II Transposed
        # y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] - a[1]*y[n-1] - a[2]*y[n-2]
        
        x_curr = current.astype(np.float32)
        x_prev1 = state['x_prev'][0]
        x_prev2 = state['x_prev'][1]
        y_prev1 = state['y_prev'][0]
        y_prev2 = state['y_prev'][1]
        
        # Cek apakah ukuran frame berubah (deteksi wajah bergeser)
        # Jika berubah, reset state ke zeros dengan ukuran baru (lebih aman dari resize)
        if x_curr.shape != x_prev1.shape:
            # Reset state dengan zeros - lebih stabil daripada resize
            target_shape = x_curr.shape
            x_prev1 = np.zeros(target_shape, dtype=np.float32)
            x_prev2 = np.zeros(target_shape, dtype=np.float32)
            y_prev1 = np.zeros(target_shape, dtype=np.float32)
            y_prev2 = np.zeros(target_shape, dtype=np.float32)
            
            # Update state dengan ukuran baru
            state['x_prev'][0] = x_prev1
            state['x_prev'][1] = x_prev2
            state['y_prev'][0] = y_prev1
            state['y_prev'][1] = y_prev2
        
        # Hitung output filter dengan penanganan overflow
        try:
            y_curr = (b[0] * x_curr + 
                      b[1] * x_prev1 + 
                      b[2] * x_prev2 - 
                      a[1] * y_prev1 - 
                      a[2] * y_prev2)
            
            # Clip nilai untuk mencegah overflow dan NaN
            y_curr = np.clip(y_curr, -1e6, 1e6)
            
            # Ganti NaN/Inf dengan 0
            y_curr = np.nan_to_num(y_curr, nan=0.0, posinf=0.0, neginf=0.0)
            
        except (FloatingPointError, ValueError):
            # Jika terjadi error, return zeros
            y_curr = np.zeros_like(x_curr)
        
        # Update state
        state['x_prev'][1] = x_prev1.copy()
        state['x_prev'][0] = x_curr.copy()
        state['y_prev'][1] = y_prev1.copy()
        state['y_prev'][0] = y_curr.copy()
        
        return y_curr
    
    def process_frame(
        self,
        frame: np.ndarray,
        roi_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Memproses satu frame dan mengembalikan frame dengan EVM applied.
        
        Pipeline untuk setiap frame:
        1. (Opsional) Crop ke ROI jika disediakan
        2. Bangun Gaussian pyramid
        3. Apply temporal filter pada setiap level
        4. Amplifikasi sinyal yang difilter
        5. Rekonstruksi dan gabungkan dengan frame asli
        
        Parameters
        ----------
        frame : np.ndarray
            Frame BGR input
        roi_rect : Optional[Tuple[int, int, int, int]]
            (x, y, w, h) untuk membatasi EVM ke area tertentu (misal: wajah)
            Jika None, proses seluruh frame
        
        Returns
        -------
        np.ndarray
            Frame dengan perubahan warna yang sudah di-magnify
        """
        if frame is None or frame.size == 0:
            return self._last_frame if self._last_frame is not None else frame
        
        self._last_frame = frame.copy()
        
        # Jika ROI disediakan, proses hanya area tersebut
        if roi_rect is not None:
            x, y, w, h = roi_rect
            # Validasi bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            
            if w <= 0 or h <= 0:
                return frame
            
            roi = frame[y:y+h, x:x+w].copy()
            magnified_roi = self._process_region(roi)
            
            # Gabungkan kembali ke frame
            output = frame.copy()
            output[y:y+h, x:x+w] = magnified_roi
            return output
        else:
            return self._process_region(frame)
    
    def _process_region(self, region: np.ndarray) -> np.ndarray:
        """
        Memproses satu region dengan EVM pipeline.
        
        Parameters
        ----------
        region : np.ndarray
            Region BGR untuk diproses
        
        Returns
        -------
        np.ndarray
            Region dengan EVM applied
        """
        original_shape = region.shape
        
        # Step 1: Bangun Gaussian pyramid
        pyramid = self._build_gaussian_pyramid(region)
        
        # Step 2: Inisialisasi buffer jika belum
        if not self._initialized:
            self._initialize_buffers(pyramid)
            # Frame pertama, belum bisa filter
            return region
        
        # Step 3: Apply temporal filter dan amplifikasi pada setiap level
        filtered_pyramid = []
        for level_idx, level in enumerate(pyramid):
            # Temporal filtering
            filtered = self._apply_temporal_filter_iir(level, level_idx)
            
            # Amplifikasi
            amplified = filtered * self.amplification
            
            # Tambahkan ke level asli
            result = level + amplified
            
            filtered_pyramid.append(result)
        
        # Step 4: Rekonstruksi
        magnified = self._reconstruct_from_pyramid(filtered_pyramid, original_shape)
        
        # Step 5: Handle NaN/Inf values dan clip ke range valid
        # Ganti NaN/Inf dengan nilai dari region asli
        if np.any(~np.isfinite(magnified)):
            magnified = np.where(np.isfinite(magnified), magnified, region.astype(np.float32))
        
        magnified = np.clip(magnified, 0, 255).astype(np.uint8)
        
        return magnified
    
    def reset(self) -> None:
        """
        Reset semua buffer dan state filter.
        
        Panggil method ini jika:
        - Berganti ke video/webcam baru
        - Ingin memulai ulang pemrosesan
        - Terjadi perubahan signifikan dalam scene
        """
        self._pyramid_buffer = []
        self._filter_state = []
        self._initialized = False
        self._last_frame = None
    
    def set_amplification(self, value: float) -> None:
        """
        Mengubah faktor amplifikasi secara dinamis.
        
        Berguna untuk fine-tuning efek selama runtime.
        
        Parameters
        ----------
        value : float
            Faktor amplifikasi baru (disarankan: 20-100)
        """
        self.amplification = max(1, min(value, 200))  # Clamp ke range aman
    
    def set_frequency_band(self, low: float, high: float) -> None:
        """
        Mengubah band frekuensi untuk temporal filter.
        
        Parameters
        ----------
        low : float
            Frekuensi cutoff bawah dalam Hz
        high : float
            Frekuensi cutoff atas dalam Hz
        """
        self.freq_low = low
        self.freq_high = high
        self._b, self._a = self._create_temporal_filter()
        
        # Reset state filter karena koefisien berubah
        for state in self._filter_state:
            for key in state:
                for i in range(len(state[key])):
                    state[key][i].fill(0)


def create_color_magnifier(
    amplification: float = 50,
    freq_range: Tuple[float, float] = (0.83, 1.0)
) -> EulerianMagnifier:
    """
    Factory function untuk membuat EulerianMagnifier dengan preset untuk
    visualisasi perubahan warna (seperti denyut nadi).
    
    Parameters
    ----------
    amplification : float
        Faktor amplifikasi (default: 50)
    freq_range : Tuple[float, float]
        (freq_low, freq_high) dalam Hz (default: 0.83-1.0 Hz untuk pulse)
    
    Returns
    -------
    EulerianMagnifier
        Instance yang sudah dikonfigurasi untuk color magnification
    
    Example
    -------
    >>> magnifier = create_color_magnifier(amplification=40)
    >>> magnified_frame = magnifier.process_frame(frame, face_roi)
    """
    return EulerianMagnifier(
        pyramid_levels=4,
        amplification=amplification,
        freq_low=freq_range[0],
        freq_high=freq_range[1],
        fps=FPS_TARGET,
        buffer_size=30
    )
