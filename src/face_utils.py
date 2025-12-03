# =============================================================================
# MODUL FACE UTILITIES: Deteksi Wajah dan Ekstraksi ROI
# =============================================================================
# Modul ini bertanggung jawab untuk:
# 1. Mendeteksi wajah menggunakan Haar Cascade
# 2. Mengekstrak Region of Interest (ROI) area dahi
# 3. Membuat skin mask menggunakan ruang warna YCbCr
# 4. Menghitung rata-rata nilai RGB dari piksel kulit
# =============================================================================

import cv2
import numpy as np
from typing import Tuple, Optional

# Import konfigurasi dari modul config
from src.config import (
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    FOREHEAD_TOP_RATIO, FOREHEAD_HEIGHT_RATIO,
    FOREHEAD_LEFT_RATIO, FOREHEAD_WIDTH_RATIO,
    SKIN_YCRCB_LOWER, SKIN_YCRCB_UPPER,
    MIN_SKIN_PIXELS
)


# =============================================================================
# INISIALISASI DETEKTOR WAJAH
# =============================================================================
# Menggunakan Haar Cascade yang sudah tersedia di OpenCV
# Model ini cukup cepat untuk pemrosesan real-time

# Load Haar Cascade classifier untuk deteksi wajah frontal
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Mendeteksi wajah pada frame menggunakan Haar Cascade.
    
    Algoritma Haar Cascade bekerja dengan cara:
    1. Mengkonversi gambar ke grayscale
    2. Mencari pola-pola Haar features (tepi, garis, dll)
    3. Menggunakan cascade classifier untuk memfilter kandidat wajah
    
    Parameters
    ----------
    frame : np.ndarray
        Frame BGR dari webcam dengan shape (height, width, 3)
    
    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        Bounding box wajah dalam format (x, y, width, height)
        Mengembalikan None jika tidak ada wajah terdeteksi
    
    Notes
    -----
    - Hanya mengembalikan wajah terbesar jika ada multiple deteksi
    - Parameter SCALE_FACTOR dan MIN_NEIGHBORS dari config.py
    """
    # Konversi ke grayscale untuk deteksi Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah dengan multi-scale detection
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )
    
    # Jika tidak ada wajah terdeteksi
    if len(faces) == 0:
        return None
    
    # Ambil wajah dengan area terbesar (kemungkinan wajah terdekat/utama)
    # faces adalah array dengan shape (n, 4) dimana setiap baris adalah [x, y, w, h]
    largest_face_idx = np.argmax(faces[:, 2] * faces[:, 3])  # Area = width * height
    x, y, w, h = faces[largest_face_idx]
    
    return (int(x), int(y), int(w), int(h))


def extract_forehead_roi(
    frame: np.ndarray,
    face_rect: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Mengekstrak Region of Interest (ROI) area dahi dari bounding box wajah.
    
    Area dahi dipilih karena:
    1. Memiliki pembuluh darah yang dekat dengan permukaan kulit
    2. Relatif stabil dan tidak banyak bergerak (dibanding mulut/mata)
    3. Biasanya tidak tertutup rambut atau kacamata
    
    Posisi ROI dihitung berdasarkan proporsi dari bounding box wajah:
    - Top: 5% dari atas wajah (menghindari garis rambut)
    - Height: 25% dari tinggi wajah
    - Left: 20% dari kiri wajah (menghindari tepi)
    - Width: 60% dari lebar wajah (area tengah)
    
    Parameters
    ----------
    frame : np.ndarray
        Frame BGR dari webcam
    face_rect : Tuple[int, int, int, int]
        Bounding box wajah (x, y, width, height)
    
    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int, int, int]]
        - ROI image (crop dari frame)
        - Koordinat ROI absolut (x, y, width, height) dalam frame
    """
    x, y, w, h = face_rect
    
    # Hitung koordinat ROI dahi berdasarkan proporsi
    roi_x = int(x + w * FOREHEAD_LEFT_RATIO)
    roi_y = int(y + h * FOREHEAD_TOP_RATIO)
    roi_w = int(w * FOREHEAD_WIDTH_RATIO)
    roi_h = int(h * FOREHEAD_HEIGHT_RATIO)
    
    # Pastikan koordinat tidak keluar dari batas frame
    frame_h, frame_w = frame.shape[:2]
    roi_x = max(0, min(roi_x, frame_w - 1))
    roi_y = max(0, min(roi_y, frame_h - 1))
    roi_w = min(roi_w, frame_w - roi_x)
    roi_h = min(roi_h, frame_h - roi_y)
    
    # Crop ROI dari frame
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    
    return roi, (roi_x, roi_y, roi_w, roi_h)


def create_skin_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Membuat binary mask untuk piksel kulit menggunakan ruang warna YCbCr.
    
    Ruang warna YCbCr dipilih karena:
    1. Memisahkan luminance (Y) dari chrominance (Cb, Cr)
    2. Warna kulit manusia memiliki rentang Cb dan Cr yang relatif konsisten
       terlepas dari intensitas pencahayaan (Y)
    3. Lebih robust terhadap variasi pencahayaan dibanding RGB
    
    Proses masking:
    1. Konversi BGR ke YCrCb (OpenCV menggunakan urutan YCrCb, bukan YCbCr)
    2. Threshold berdasarkan rentang warna kulit yang sudah dioptimalkan
    3. Morphological opening untuk menghilangkan noise
    
    Parameters
    ----------
    roi_bgr : np.ndarray
        Region of Interest dalam format BGR dengan shape (height, width, 3)
    
    Returns
    -------
    np.ndarray
        Binary mask dengan shape (height, width)
        Nilai 255 untuk piksel kulit, 0 untuk non-kulit
    
    References
    ----------
    Rentang warna kulit YCbCr berdasarkan penelitian:
    - Y: 0-255 (semua nilai luminance diterima)
    - Cr: 133-173 (komponen merah-kuning kulit)
    - Cb: 77-127 (komponen biru-kuning kulit)
    """
    # Validasi input
    if roi_bgr is None or roi_bgr.size == 0:
        return np.array([], dtype=np.uint8)
    
    # Konversi BGR ke YCrCb
    # OpenCV menggunakan urutan YCrCb, bukan YCbCr
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    
    # Threshold berdasarkan rentang warna kulit
    # SKIN_YCRCB_LOWER dan SKIN_YCRCB_UPPER dari config.py
    mask = cv2.inRange(ycrcb, SKIN_YCRCB_LOWER, SKIN_YCRCB_UPPER)
    
    # Morphological opening untuk menghilangkan noise kecil
    # Kernel ellipse lebih natural untuk bentuk kulit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def extract_rgb_means(
    roi_bgr: np.ndarray,
    skin_mask: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """
    Menghitung rata-rata nilai R, G, B dari piksel kulit dalam ROI.
    
    Fungsi ini adalah inti dari ekstraksi sinyal rPPG:
    1. Mengambil hanya piksel yang termasuk kulit (berdasarkan mask)
    2. Menghitung rata-rata spasial untuk setiap channel warna
    3. Rata-rata ini akan berfluktuasi seiring dengan denyut nadi
       karena perubahan volume darah mempengaruhi warna kulit
    
    Parameters
    ----------
    roi_bgr : np.ndarray
        Region of Interest dalam format BGR
    skin_mask : np.ndarray
        Binary mask untuk piksel kulit
    
    Returns
    -------
    Optional[Tuple[float, float, float]]
        Tuple (R_mean, G_mean, B_mean) jika cukup piksel kulit tersedia
        None jika piksel kulit kurang dari MIN_SKIN_PIXELS
    
    Notes
    -----
    - Minimal MIN_SKIN_PIXELS piksel kulit diperlukan untuk hasil yang valid
    - Urutan return adalah RGB (bukan BGR) untuk konsistensi dengan ICA
    """
    # Validasi input
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    if skin_mask is None or skin_mask.size == 0:
        return None
    
    # Pastikan dimensi mask sesuai dengan ROI
    if roi_bgr.shape[:2] != skin_mask.shape:
        return None
    
    # Hitung jumlah piksel kulit yang valid
    skin_pixel_count = cv2.countNonZero(skin_mask)
    
    # Jika piksel kulit kurang dari threshold, return None
    if skin_pixel_count < MIN_SKIN_PIXELS:
        return None
    
    # Ekstrak piksel kulit menggunakan mask
    # mask > 0 menghasilkan boolean array
    skin_pixels = roi_bgr[skin_mask > 0]
    
    # Hitung rata-rata untuk setiap channel
    # OpenCV menggunakan urutan BGR, konversi ke RGB
    b_mean = np.mean(skin_pixels[:, 0])
    g_mean = np.mean(skin_pixels[:, 1])
    r_mean = np.mean(skin_pixels[:, 2])
    
    return (r_mean, g_mean, b_mean)


def get_face_and_roi(
    frame: np.ndarray
) -> Tuple[
    Optional[Tuple[int, int, int, int]],
    Optional[np.ndarray],
    Optional[Tuple[int, int, int, int]],
    Optional[np.ndarray]
]:
    """
    Fungsi wrapper yang menggabungkan semua langkah deteksi dan ekstraksi.
    
    Urutan pemrosesan:
    1. Deteksi wajah → face_rect
    2. Ekstrak ROI dahi → forehead_roi, roi_rect
    3. Buat skin mask → skin_mask
    
    Parameters
    ----------
    frame : np.ndarray
        Frame BGR dari webcam
    
    Returns
    -------
    Tuple dengan 4 elemen:
        - face_rect: Bounding box wajah atau None
        - forehead_roi: Image ROI dahi atau None
        - roi_rect: Koordinat ROI absolut atau None
        - skin_mask: Binary mask kulit atau None
    
    Notes
    -----
    Jika wajah tidak terdeteksi, semua return values adalah None
    """
    # Step 1: Deteksi wajah
    face_rect = detect_face(frame)
    
    if face_rect is None:
        return None, None, None, None
    
    # Step 2: Ekstrak ROI dahi
    forehead_roi, roi_rect = extract_forehead_roi(frame, face_rect)
    
    # Validasi ROI tidak kosong
    if forehead_roi.size == 0:
        return face_rect, None, None, None
    
    # Step 3: Buat skin mask
    skin_mask = create_skin_mask(forehead_roi)
    
    return face_rect, forehead_roi, roi_rect, skin_mask
