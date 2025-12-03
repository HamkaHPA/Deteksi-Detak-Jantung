#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# PROGRAM UTAMA: Sistem Monitor Detak Jantung rPPG Real-time
# =============================================================================
# 
# Deskripsi:
# ----------
# Program ini mengimplementasikan sistem Remote Photoplethysmography (rPPG)
# untuk mengukur detak jantung secara non-kontak menggunakan webcam.
#
# Metode yang digunakan:
# 1. ICA (Independent Component Analysis) - untuk ekstraksi sinyal BVP
# 2. Eulerian Video Magnification (EVM) - untuk visualisasi denyut pada wajah
#
# Komponen Sistem:
# ----------------
# - Deteksi wajah menggunakan Haar Cascade (OpenCV)
# - Ekstraksi ROI (Region of Interest) pada area dahi
# - Skin masking dengan ruang warna YCbCr
# - Pemrosesan sinyal dengan FastICA (scikit-learn)
# - Analisis frekuensi dengan FFT (scipy)
# - Visualisasi real-time dengan OpenCV
#
# Hotkeys:
# --------
# Q     : Keluar dari program
# R     : Reset buffer sinyal
# E     : Toggle Eulerian Video Magnification on/off
# +/=   : Tingkatkan amplifikasi EVM
# -/_   : Kurangi amplifikasi EVM
#
# Penggunaan:
# -----------
# python run.py
#
# =============================================================================

import cv2
import numpy as np
import time
from collections import deque
from typing import Optional, Tuple

# Import modul-modul dari package src
from src.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET,
    BUFFER_SIZE, HR_DISPLAY_TIMEOUT, SMOOTHING_WINDOW,
    EVM_AMPLIFICATION
)
from src.face_utils import get_face_and_roi, extract_rgb_means
from src.ica_processor import process_rgb_buffer
from src.evm_processor import create_color_magnifier
from src.ui_renderer import create_composite_display


class HeartRateMonitor:
    """
    Kelas utama untuk sistem monitoring detak jantung rPPG.
    
    Kelas ini mengorkestrasi seluruh pipeline pemrosesan:
    1. Capture video dari webcam
    2. Deteksi wajah dan ekstraksi ROI
    3. Pengumpulan sinyal RGB ke buffer
    4. Pemrosesan ICA untuk ekstraksi BVP
    5. Analisis FFT untuk estimasi detak jantung
    6. Visualisasi dengan EVM (opsional)
    7. Rendering UI
    
    Attributes
    ----------
    cap : cv2.VideoCapture
        Object untuk capture video dari webcam
    rgb_buffer : deque
        Buffer circular untuk menyimpan nilai RGB
    hr_history : deque
        Buffer untuk smoothing hasil detak jantung
    evm_magnifier : EulerianMagnifier
        Object untuk Eulerian Video Magnification
    evm_enabled : bool
        Flag untuk mengaktifkan/menonaktifkan EVM
    last_valid_hr : Optional[float]
        Detak jantung valid terakhir untuk display timeout
    last_hr_time : float
        Timestamp terakhir HR valid terdeteksi
    """
    
    def __init__(self):
        """
        Inisialisasi HeartRateMonitor dengan konfigurasi default.
        """
        # =====================================================================
        # INISIALISASI VIDEO CAPTURE
        # =====================================================================
        print("[INFO] Menginisialisasi webcam...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        # Set resolusi dan FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        
        # Validasi webcam berhasil dibuka
        if not self.cap.isOpened():
            raise RuntimeError(
                "[ERROR] Tidak dapat membuka webcam! "
                "Pastikan webcam terhubung dan tidak digunakan aplikasi lain."
            )
        
        # Dapatkan FPS aktual dari webcam
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps <= 0:
            self.actual_fps = FPS_TARGET
        print(f"[INFO] Webcam aktif pada {self.actual_fps:.1f} FPS")
        
        # =====================================================================
        # INISIALISASI BUFFER
        # =====================================================================
        # Buffer untuk menyimpan nilai rata-rata RGB dari setiap frame
        # Menggunakan deque dengan maxlen untuk efisiensi (auto-discard oldest)
        self.rgb_buffer = deque(maxlen=BUFFER_SIZE)
        
        # Buffer untuk smoothing hasil HR (mengurangi fluktuasi)
        self.hr_history = deque(maxlen=SMOOTHING_WINDOW)
        
        # =====================================================================
        # INISIALISASI EVM (Eulerian Video Magnification)
        # =====================================================================
        self.evm_magnifier = create_color_magnifier(amplification=EVM_AMPLIFICATION)
        self.evm_enabled = True  # EVM aktif secara default
        self.evm_amplification = EVM_AMPLIFICATION
        
        # =====================================================================
        # STATE UNTUK DISPLAY DAN TIMEOUT
        # =====================================================================
        self.last_valid_hr: Optional[float] = None
        self.last_hr_time: float = time.time()
        self.last_face_rect: Optional[Tuple[int, int, int, int]] = None
        self.last_roi_rect: Optional[Tuple[int, int, int, int]] = None
        
        # =====================================================================
        # STATE UNTUK HASIL PEMROSESAN
        # =====================================================================
        self.current_bvp: Optional[np.ndarray] = None
        self.current_freqs: Optional[np.ndarray] = None
        self.current_fft: Optional[np.ndarray] = None
        self.current_hr: Optional[float] = None
        
        # =====================================================================
        # TIMING
        # =====================================================================
        self.frame_count = 0
        self.start_time = time.time()
        
        print("[INFO] Sistem siap!")
        print("[INFO] Tekan 'Q' untuk keluar, 'E' untuk toggle EVM")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Memproses satu frame dari webcam.
        
        Pipeline pemrosesan:
        1. Deteksi wajah dan ekstraksi ROI dahi
        2. Skin masking untuk mendapatkan piksel kulit valid
        3. Ekstraksi nilai rata-rata RGB
        4. Tambahkan ke buffer
        5. Jika buffer cukup, proses dengan ICA
        6. Hitung detak jantung dengan FFT
        7. (Opsional) Apply EVM untuk visualisasi
        8. Render UI
        
        Parameters
        ----------
        frame : np.ndarray
            Frame BGR dari webcam
        
        Returns
        -------
        np.ndarray
            Frame dengan overlay UI dan (opsional) EVM
        """
        # =====================================================================
        # STEP 1: DETEKSI WAJAH DAN EKSTRAKSI ROI
        # =====================================================================
        face_rect, forehead_roi, roi_rect, skin_mask = get_face_and_roi(frame)
        
        # Update state wajah terakhir untuk display
        if face_rect is not None:
            self.last_face_rect = face_rect
            self.last_roi_rect = roi_rect
        
        # =====================================================================
        # STEP 2: EKSTRAKSI NILAI RGB DARI PIKSEL KULIT
        # =====================================================================
        is_valid = False
        
        if forehead_roi is not None and skin_mask is not None:
            rgb_means = extract_rgb_means(forehead_roi, skin_mask)
            
            if rgb_means is not None:
                # Tambahkan ke buffer
                self.rgb_buffer.append(rgb_means)
                is_valid = True
        
        # =====================================================================
        # STEP 3: PROSES ICA DAN FFT JIKA BUFFER CUKUP
        # =====================================================================
        # Minimal 50% dari BUFFER_SIZE untuk mulai analisis
        min_samples = BUFFER_SIZE // 2
        
        if len(self.rgb_buffer) >= min_samples:
            # Konversi buffer ke numpy array
            rgb_array = np.array(self.rgb_buffer)
            
            # Proses dengan ICA dan FFT
            hr, bvp, freqs, fft_mag = process_rgb_buffer(rgb_array, self.actual_fps)
            
            # Update state hasil
            if hr is not None:
                # Validasi rentang HR yang masuk akal (40-200 BPM)
                if 40 <= hr <= 200:
                    # Smoothing dengan moving average
                    self.hr_history.append(hr)
                    smoothed_hr = np.mean(self.hr_history)
                    
                    self.current_hr = smoothed_hr
                    self.last_valid_hr = smoothed_hr
                    self.last_hr_time = time.time()
            
            # Update sinyal untuk visualisasi
            if bvp is not None:
                self.current_bvp = bvp
            if freqs is not None and len(freqs) > 0:
                self.current_freqs = freqs
                self.current_fft = fft_mag
        
        # =====================================================================
        # STEP 4: CHECK HR DISPLAY TIMEOUT
        # =====================================================================
        # Jika sudah lama tidak ada HR valid, tampilkan None
        if time.time() - self.last_hr_time > HR_DISPLAY_TIMEOUT:
            display_hr = None
        else:
            display_hr = self.last_valid_hr
        
        # =====================================================================
        # STEP 5: APPLY EVM JIKA AKTIF
        # =====================================================================
        output_frame = frame.copy()
        
        if self.evm_enabled and face_rect is not None:
            # Apply EVM hanya pada area wajah untuk performa lebih baik
            output_frame = self.evm_magnifier.process_frame(output_frame, face_rect)
        
        # =====================================================================
        # STEP 6: HITUNG METRICS
        # =====================================================================
        buffer_progress = len(self.rgb_buffer) / BUFFER_SIZE * 100
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        current_fps = self.frame_count / elapsed if elapsed > 0 else 30
        
        # =====================================================================
        # STEP 7: RENDER UI MODERN
        # =====================================================================
        output_frame = create_composite_display(
            output_frame,
            bvp_signal=self.current_bvp,
            freqs=self.current_freqs,
            fft_magnitude=self.current_fft,
            heart_rate=display_hr,
            face_rect=self.last_face_rect,
            roi_rect=self.last_roi_rect,
            is_valid=is_valid and len(self.rgb_buffer) >= min_samples,
            show_instructions=True,
            buffer_progress=buffer_progress,
            fps=current_fps,
            evm_enabled=self.evm_enabled,
            evm_amp=self.evm_amplification,
            frame_count=self.frame_count
        )
        
        return output_frame
    
    def handle_key(self, key: int) -> bool:
        """
        Menangani input keyboard.
        
        Parameters
        ----------
        key : int
            Kode tombol dari cv2.waitKey()
        
        Returns
        -------
        bool
            True jika program harus berhenti, False untuk lanjut
        """
        # Q atau ESC: Keluar
        if key == ord('q') or key == ord('Q') or key == 27:
            print("[INFO] Keluar dari program...")
            return True
        
        # R: Reset buffer
        if key == ord('r') or key == ord('R'):
            print("[INFO] Mereset buffer sinyal...")
            self.rgb_buffer.clear()
            self.hr_history.clear()
            self.current_bvp = None
            self.current_freqs = None
            self.current_fft = None
            self.current_hr = None
            self.evm_magnifier.reset()
        
        # E: Toggle EVM
        if key == ord('e') or key == ord('E'):
            self.evm_enabled = not self.evm_enabled
            status = "aktif" if self.evm_enabled else "nonaktif"
            print(f"[INFO] EVM {status}")
            if self.evm_enabled:
                self.evm_magnifier.reset()
        
        # + atau =: Tingkatkan amplifikasi EVM
        if key == ord('+') or key == ord('='):
            self.evm_amplification = min(150, self.evm_amplification + 10)
            self.evm_magnifier.set_amplification(self.evm_amplification)
            print(f"[INFO] Amplifikasi EVM: {self.evm_amplification}")
        
        # - atau _: Kurangi amplifikasi EVM
        if key == ord('-') or key == ord('_'):
            self.evm_amplification = max(10, self.evm_amplification - 10)
            self.evm_magnifier.set_amplification(self.evm_amplification)
            print(f"[INFO] Amplifikasi EVM: {self.evm_amplification}")
        
        return False
    
    def run(self) -> None:
        """
        Main loop untuk menjalankan sistem monitoring.
        
        Loop ini berjalan terus-menerus sampai user menekan Q/ESC:
        1. Baca frame dari webcam
        2. Proses frame
        3. Tampilkan hasil
        4. Handle input keyboard
        """
        print("\n" + "=" * 60)
        print("SISTEM MONITOR DETAK JANTUNG rPPG REAL-TIME")
        print("Metode: ICA (Independent Component Analysis)")
        print("Visualisasi: Eulerian Video Magnification")
        print("=" * 60)
        print("\nInstruksi:")
        print("  - Posisikan wajah Anda di depan kamera")
        print("  - Tetap diam dan hindari gerakan berlebihan")
        print("  - Pastikan pencahayaan cukup dan merata")
        print("  - Tunggu beberapa detik sampai buffer terisi")
        print("\nHotkeys:")
        print("  Q     : Keluar")
        print("  R     : Reset buffer")
        print("  E     : Toggle EVM on/off")
        print("  +/-   : Atur amplifikasi EVM")
        print("=" * 60 + "\n")
        
        try:
            while True:
                # Baca frame dari webcam
                ret, frame = self.cap.read()
                
                if not ret:
                    print("[WARNING] Gagal membaca frame dari webcam")
                    continue
                
                # Flip horizontal untuk efek mirror (lebih natural)
                frame = cv2.flip(frame, 1)
                
                # Proses frame
                output = self.process_frame(frame)
                
                # Tampilkan hasil
                cv2.imshow("rPPG Heart Rate Monitor - ICA & EVM", output)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if self.handle_key(key):
                    break
                    
        except KeyboardInterrupt:
            print("\n[INFO] Dihentikan oleh user (Ctrl+C)")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """
        Membersihkan resources saat program berakhir.
        """
        print("[INFO] Membersihkan resources...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Selesai.")


def main():
    """
    Entry point untuk program.
    """
    try:
        monitor = HeartRateMonitor()
        monitor.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
