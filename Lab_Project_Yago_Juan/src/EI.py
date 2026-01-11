from typing import Optional, Tuple
import cv2
import numpy as np
import math
from collections import deque, Counter

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

def find_largest_quad(frame_bgr: np.ndarray,
                      min_area_ratio: float = 0.06,
                      canny1: int = 50,
                      canny2: int = 150) -> Optional[np.ndarray]:
    """
    Encuentra en TODO el frame el mayor contorno convexo aproximable a 4 puntos.
    Devuelve quad (4,2) float32 o None.
    """
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, canny1, canny2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best_quad = None
    best_area = 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_ratio * frame_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best_quad = approx.reshape(4, 2).astype("float32")

    return best_quad

def warp_quad(frame_bgr: np.ndarray, quad_pts: np.ndarray, out_size: int = 600) -> np.ndarray:
    """Rectifica la hoja a una imagen frontal out_size x out_size."""
    rect = _order_points(quad_pts)
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (out_size, out_size))
    return warped

def draw_text(img, text, pos, scale=1.0):
    # sombra para legibilidad
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    # texto rojo
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA)

class StableLabel:
    """Votación temporal para evitar parpadeos en vídeo."""
    def __init__(self, window_size: int = 12, min_consensus: int = 8):
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
        self.min_consensus = min_consensus

    def update(self, label: str) -> Tuple[str, int, bool]:
        self.window.append(label)
        c = Counter(self.window)
        best_label, count = c.most_common(1)[0]
        stable = (best_label not in ("UNKNOWN", "NO_TARGET")) and (count >= self.min_consensus)
        return best_label, count, stable

    def reset(self):
        self.window.clear()

class PatternDetector:
    """
    Detector robusto dentro de la hoja rectificada:
    - Normaliza iluminación (CLAHE)
    - Recorta un margen interior para evitar bordes/sombras
    - Canny + HoughLinesP para líneas
    - HoughCircles para círculos
    Devuelve (label, score, dbg)
    """

    def __init__(self, inner_margin_ratio: float = 0.10, debug: bool = False):
        self.inner_margin_ratio = inner_margin_ratio
        self.debug = debug
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _crop_inner(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        m = int(min(h, w) * self.inner_margin_ratio)
        if m <= 0:
            return img
        return img[m:h - m, m:w - m]

    def _preprocess(self, warped_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _detect_lines(self, edges: np.ndarray) -> Optional[Tuple[str, float]]:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=60,
            maxLineGap=12
        )
        if lines is None:
            return None

        bins = {"LINE_H": 0.0, "LINE_V": 0.0, "LINE_D1": 0.0, "LINE_D2": 0.0}
        total_len = 0.0

        def add(angle_deg, length):
            a = angle_deg
            while a > 90: a -= 180
            while a < -90: a += 180

            if abs(a) <= 15:
                bins["LINE_H"] += length
            elif abs(abs(a) - 90) <= 15:
                bins["LINE_V"] += length
            elif abs(a - 45) <= 18:
                bins["LINE_D1"] += length
            elif abs(a + 45) <= 18:
                bins["LINE_D2"] += length

        for (x1, y1, x2, y2) in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            total_len += length
            add(angle, length)

        if total_len <= 0:
            return None

        best_label = max(bins, key=bins.get)
        best_len = bins[best_label]
        dominance = best_len / total_len

        if dominance < 0.55:
            return None

        score_len = min(1.0, total_len / 1200.0)
        score = 0.65 * dominance + 0.35 * score_len
        return best_label, float(min(1.0, score))

    def _detect_circle(self, gray: np.ndarray) -> Optional[Tuple[str, float]]:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=140,
            param2=40,
            minRadius=25,
            maxRadius=0
        )
        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(int)
        x, y, r = circles[0]
        score = min(1.0, r / 160.0)
        if score < 0.20:
            return None
        return ("CIRCLE", float(score))

    def detect(self, warped_bgr: np.ndarray) -> Tuple[str, float, Optional[dict]]:
        if warped_bgr is None:
            return ("UNKNOWN", 0.0, None)

        roi = self._crop_inner(warped_bgr)
        gray = self._preprocess(roi)

        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        line_res = self._detect_lines(edges)
        circ_res = self._detect_circle(gray)

        candidates = []
        if line_res is not None:
            candidates.append(line_res)
        if circ_res is not None:
            candidates.append(circ_res)

        if not candidates:
            dbg = {"roi": roi, "gray": gray, "edges": edges} if self.debug else None
            return ("UNKNOWN", 0.0, dbg)

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = candidates[0]

        # Anti "siempre círculo": si la línea es comparable, gana línea
        if best_label == "CIRCLE" and line_res is not None:
            line_label, line_score = line_res
            if line_score >= best_score - 0.10:
                best_label, best_score = line_label, line_score

        dbg = {"roi": roi, "gray": gray, "edges": edges} if self.debug else None
        return (best_label, best_score, dbg)

class SequenceDecoder:
    """
    - Memoriza hasta 4 patrones confirmados.
    - Secuencia esperada: ["LINE_H", "LINE_V", "CIRCLE"]
    - Si el orden falla -> BLOCK y reset (por defecto).
    - Si completa -> ALLOW y reset.
    """

    def __init__(self, expected_sequence, max_len=4, reset_on_block=True):
        if not expected_sequence:
            raise ValueError("expected_sequence no puede ser vacío")
        if len(expected_sequence) > max_len:
            raise ValueError("expected_sequence no puede ser mayor que max_len")

        self.expected = list(expected_sequence)
        self.max_len = max_len
        self.reset_on_block = reset_on_block
        self.buffer = deque(maxlen=max_len)

    def reset(self):
        self.buffer.clear()

    def state(self):
        return list(self.buffer)

    def update(self, pattern_label):
        self.buffer.append(pattern_label)
        k = len(self.buffer)

        if list(self.buffer) != self.expected[:k]:
            if self.reset_on_block:
                self.reset()
            return ("BLOCK", self.state())

        if k == len(self.expected):
            self.reset()
            return ("ALLOW", [])

        return ("IN_PROGRESS", self.state())
