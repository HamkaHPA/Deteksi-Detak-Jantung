# =============================================================================
# MODUL UI RENDERER: Visualisasi dengan Panel Samping
# =============================================================================
# UI dengan layout:
# - Area KIRI: Video feed wajah (tidak terhalangi)
# - Area KANAN: Panel dengan background solid berisi grafik dan info
#
# Wajah akan terlihat jelas tanpa overlay grafik yang mengganggu
# =============================================================================

import cv2
import numpy as np
from typing import Tuple, Optional, List

from src.config import FREQ_MIN, FREQ_MAX


# =============================================================================
# TEMA WARNA
# =============================================================================
THEME_PRIMARY = (76, 175, 80)       # Hijau
THEME_SECONDARY = (33, 150, 243)    # Biru
THEME_ACCENT = (255, 87, 34)        # Oranye
THEME_BG_DARK = (25, 25, 25)        # Background panel
THEME_BG_CARD = (35, 35, 35)        # Card background
THEME_BORDER = (60, 60, 60)         # Border
THEME_TEXT = (255, 255, 255)        # Teks putih
THEME_TEXT_DIM = (140, 140, 140)    # Teks redup

# Layout
PANEL_WIDTH = 300                    # Lebar panel samping


def create_sidebar_layout(frame: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Membuat layout dengan sidebar di kanan.
    Mengembalikan frame yang sudah diperlebar dengan panel kosong.
    
    Returns:
        Tuple (frame_baru, x_posisi_panel)
    """
    h, w = frame.shape[:2]
    
    # Buat canvas baru yang lebih lebar
    new_width = w + PANEL_WIDTH
    canvas = np.zeros((h, new_width, 3), dtype=np.uint8)
    
    # Isi panel kanan dengan background gelap
    canvas[:, w:] = THEME_BG_DARK
    
    # Salin frame asli ke sisi kiri
    canvas[:, :w] = frame
    
    # Garis pemisah
    cv2.line(canvas, (w, 0), (w, h), THEME_BORDER, 2)
    
    return canvas, w


def draw_card(img, x, y, w, h, title="", header_color=THEME_PRIMARY):
    """Menggambar card dengan header berwarna."""
    # Card background
    cv2.rectangle(img, (x, y), (x + w, y + h), THEME_BG_CARD, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), THEME_BORDER, 1)
    
    if title:
        # Header
        header_h = 28
        cv2.rectangle(img, (x, y), (x + w, y + header_h), header_color, -1)
        cv2.putText(img, title, (x + 10, y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, THEME_TEXT, 1, cv2.LINE_AA)
        return y + header_h + 5
    return y + 5


def draw_bvp_chart(img, signal, x, y, w, h):
    """Menggambar chart BVP dalam area yang ditentukan."""
    # Grid
    for i in range(4):
        gy = y + int(i * h / 3)
        cv2.line(img, (x, gy), (x + w, gy), THEME_BORDER, 1)
    
    if signal is None or len(signal) < 2:
        cv2.putText(img, "Mengumpulkan data...", (x + 10, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, THEME_TEXT_DIM, 1, cv2.LINE_AA)
        return
    
    # Normalisasi
    sig_min, sig_max = np.min(signal), np.max(signal)
    if sig_max - sig_min > 1e-6:
        norm = (signal - sig_min) / (sig_max - sig_min)
    else:
        norm = np.zeros_like(signal) + 0.5
    
    # Points
    n = len(norm)
    points = []
    for i, v in enumerate(norm):
        px = x + int(i * w / (n - 1))
        py = y + int((1 - v) * h)
        points.append((px, py))
    
    pts = np.array(points, dtype=np.int32)
    
    # Area fill
    fill_pts = list(points) + [(x + w, y + h), (x, y + h)]
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(fill_pts, dtype=np.int32)], (30, 70, 32))
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    
    # Line
    cv2.polylines(img, [pts], False, THEME_PRIMARY, 2, cv2.LINE_AA)


def draw_fft_chart(img, freqs, magnitude, heart_rate, x, y, w, h):
    """Menggambar chart FFT dalam area yang ditentukan."""
    if freqs is None or magnitude is None or len(freqs) < 2:
        cv2.putText(img, "Menunggu data...", (x + 10, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, THEME_TEXT_DIM, 1, cv2.LINE_AA)
        return
    
    # Filter ke rentang HR
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    if not np.any(mask):
        return
    
    valid_f = freqs[mask]
    valid_m = magnitude[mask]
    
    # Normalisasi
    mag_max = np.max(valid_m)
    norm_m = valid_m / mag_max if mag_max > 1e-6 else np.zeros_like(valid_m)
    
    # Bars
    n_bars = min(len(valid_f), 35)
    indices = np.linspace(0, len(valid_f) - 1, n_bars, dtype=int)
    bar_w = max(4, w // n_bars - 2)
    
    for i, idx in enumerate(indices):
        f, m = valid_f[idx], norm_m[idx]
        bx = x + int(i * w / n_bars)
        bh = int(m * h)
        by = y + h - bh
        
        # Highlight peak HR
        color = THEME_ACCENT if (heart_rate and abs(f * 60 - heart_rate) < 5) else THEME_SECONDARY
        cv2.rectangle(img, (bx, by), (bx + bar_w, y + h), color, -1)
    
    # Frequency labels
    for hz in [1.0, 2.0, 3.0]:
        if FREQ_MIN <= hz <= FREQ_MAX:
            lx = x + int((hz - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) * w)
            cv2.putText(img, f"{hz:.0f}Hz", (lx - 8, y + h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, THEME_TEXT_DIM, 1, cv2.LINE_AA)


def draw_hr_display(img, heart_rate, is_valid, x, y, w, h):
    """Menggambar display detak jantung."""
    center_x = x + w // 2
    
    if heart_rate is not None and is_valid:
        # Warna berdasarkan HR
        if heart_rate < 60:
            color = THEME_SECONDARY
            status = "Rendah"
        elif heart_rate > 100:
            color = THEME_ACCENT
            status = "Tinggi"
        else:
            color = THEME_PRIMARY
            status = "Normal"
        
        # Nilai HR besar di tengah
        hr_text = f"{int(heart_rate)}"
        text_size = cv2.getTextSize(hr_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
        cv2.putText(img, hr_text, (center_x - text_size[0] // 2, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3, cv2.LINE_AA)
        
        # BPM label
        cv2.putText(img, "BPM", (center_x - 20, y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, THEME_TEXT, 1, cv2.LINE_AA)
        
        # Status
        cv2.putText(img, status, (center_x - 25, y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Heart icon
        hx, hy = x + w - 35, y + 35
        cv2.circle(img, (hx - 5, hy), 7, color, -1)
        cv2.circle(img, (hx + 5, hy), 7, color, -1)
        tri = np.array([[hx - 12, hy + 2], [hx + 12, hy + 2], [hx, hy + 18]], dtype=np.int32)
        cv2.fillPoly(img, [tri], color)
        
    else:
        cv2.putText(img, "--", (center_x - 25, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, THEME_TEXT_DIM, 3, cv2.LINE_AA)
        cv2.putText(img, "BPM", (center_x - 20, y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, THEME_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, "Mendeteksi...", (center_x - 40, y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, THEME_TEXT_DIM, 1, cv2.LINE_AA)


def draw_face_corners(img, face_rect, roi_rect=None, is_valid=True):
    """Menggambar corner brackets pada wajah (tidak menutupi)."""
    if face_rect is None:
        return img
    
    x, y, w, h = face_rect
    corner_len = min(w, h) // 5
    thickness = 3
    color = THEME_PRIMARY if is_valid else THEME_ACCENT
    
    # Hanya corner brackets
    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)
    
    # ROI dengan border tipis
    if roi_rect is not None:
        rx, ry, rw, rh = roi_rect
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), THEME_SECONDARY, 1)
        cv2.putText(img, "ROI", (rx + 2, ry + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, THEME_SECONDARY, 1, cv2.LINE_AA)
    
    return img


def draw_info_section(img, x, y, w, buffer_pct, fps, evm_on, evm_amp):
    """Menggambar section info di panel."""
    # Buffer progress bar
    cv2.putText(img, "Buffer:", (x + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, THEME_TEXT_DIM, 1, cv2.LINE_AA)
    
    bar_x, bar_y = x + 60, y + 5
    bar_w, bar_h = w - 80, 14
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), THEME_BORDER, -1)
    fill_w = int(bar_w * min(buffer_pct / 100, 1.0))
    if fill_w > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), THEME_PRIMARY, -1)
    cv2.putText(img, f"{buffer_pct:.0f}%", (bar_x + bar_w + 5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, THEME_TEXT, 1, cv2.LINE_AA)
    
    # EVM status
    evm_color = THEME_PRIMARY if evm_on else THEME_TEXT_DIM
    cv2.putText(img, f"EVM: {'ON' if evm_on else 'OFF'} (amp={evm_amp:.0f})", (x + 10, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, evm_color, 1, cv2.LINE_AA)
    
    # FPS
    cv2.putText(img, f"FPS: {fps:.1f}", (x + 180, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, THEME_TEXT, 1, cv2.LINE_AA)


def draw_instructions_small(img, x, y):
    """Instruksi hotkey kecil."""
    instr = "[Q]Keluar [R]Reset [E]EVM [+/-]Amp"
    cv2.putText(img, instr, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, THEME_TEXT_DIM, 1, cv2.LINE_AA)


def create_composite_display(
    frame: np.ndarray,
    bvp_signal: Optional[np.ndarray],
    freqs: Optional[np.ndarray],
    fft_magnitude: Optional[np.ndarray],
    heart_rate: Optional[float],
    face_rect: Optional[Tuple[int, int, int, int]],
    roi_rect: Optional[Tuple[int, int, int, int]],
    is_valid: bool = True,
    show_instructions: bool = True,
    buffer_progress: float = 0,
    fps: float = 30,
    evm_enabled: bool = True,
    evm_amp: float = 50,
    frame_count: int = 0
) -> np.ndarray:
    """
    Membuat tampilan dengan PANEL SAMPING.
    
    Layout:
    ┌────────────────────┬─────────────────┐
    │                    │  [HR Display]   │
    │    VIDEO FEED      │  [BVP Chart]    │
    │    (Wajah clear)   │  [FFT Chart]    │
    │                    │  [Info/Status]  │
    └────────────────────┴─────────────────┘
    """
    # 1. Buat layout dengan sidebar
    canvas, panel_x = create_sidebar_layout(frame)
    h = canvas.shape[0]
    
    # 2. Gambar face overlay di area video (kiri)
    if face_rect is not None:
        canvas = draw_face_corners(canvas, face_rect, roi_rect, is_valid)
    else:
        # Pesan jika wajah tidak terdeteksi
        msg = "Posisikan wajah di depan kamera"
        ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = (panel_x - ts[0]) // 2
        cv2.putText(canvas, msg, (tx, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, THEME_ACCENT, 2, cv2.LINE_AA)
    
    # 3. Panel kanan - posisi awal
    px = panel_x + 10
    pw = PANEL_WIDTH - 20
    current_y = 10
    
    # 4. Card: Heart Rate Display
    card_h = 110
    content_y = draw_card(canvas, px, current_y, pw, card_h, "Detak Jantung", THEME_ACCENT)
    draw_hr_display(canvas, heart_rate, is_valid and heart_rate is not None,
                    px, content_y, pw, card_h - 35)
    current_y += card_h + 10
    
    # 5. Card: BVP Signal
    card_h = 110
    content_y = draw_card(canvas, px, current_y, pw, card_h, "Sinyal BVP", THEME_PRIMARY)
    draw_bvp_chart(canvas, bvp_signal, px + 5, content_y, pw - 10, card_h - 40)
    current_y += card_h + 10
    
    # 6. Card: FFT Spectrum
    card_h = 115
    content_y = draw_card(canvas, px, current_y, pw, card_h, "Spektrum FFT", THEME_SECONDARY)
    draw_fft_chart(canvas, freqs, fft_magnitude, heart_rate,
                   px + 5, content_y, pw - 10, card_h - 50)
    current_y += card_h + 10
    
    # 7. Info section
    draw_info_section(canvas, px, current_y, pw, buffer_progress, fps, evm_enabled, evm_amp)
    current_y += 50
    
    # 8. Instructions
    if show_instructions:
        draw_instructions_small(canvas, px + 5, h - 10)
    
    return canvas


# Legacy compatibility functions
def draw_face_box(frame, face_rect, color=None, label=""):
    return draw_face_corners(frame, face_rect, None)

def draw_roi_box(frame, roi_rect, color=None, label=""):
    return frame

def draw_bvp_waveform(frame, *args, **kwargs):
    return frame

def draw_fft_plot(frame, *args, **kwargs):
    return frame

def draw_heart_rate_display(frame, *args, **kwargs):
    return frame
