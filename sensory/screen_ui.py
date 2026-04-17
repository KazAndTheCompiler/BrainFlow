"""
Screen/UI Understanding Pipeline

Captures screen → extracts text via OCR → analyzes UI layout via contours
→ classifies UI regions → feeds semantically meaningful data to the brain.

Components:
1. OCR Engine    - Tesseract text extraction with word-level bounding boxes
2. Layout Engine - cv2 contour detection for UI region boundaries
3. UI Classifier - Heuristic classification of region types (button, input, text, image, etc.)
4. Saliency Map  - Identifies most visually important region on screen

One-shot OCR approach:
- Single Tesseract call per analyze() using get_text_from_bbox()
- Word bounding boxes are clustered into region groups
- No per-region re-OCR — words are pre-attributed to regions by spatial overlap

Dependencies:
- opencv-python-headless (layout analysis)
- pytesseract + Tesseract binary (OCR)
- mss or Pillow (screen capture)
"""

import time
import hashlib
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum


try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pytesseract
    try:
        pytesseract.get_tesseract_version()
        HAS_OCR = True
    except Exception:
        HAS_OCR = False
except ImportError:
    HAS_OCR = False


class UIRegionType(Enum):
    BUTTON = "button"
    INPUT = "input"
    TEXT_BLOCK = "text_block"
    CODE = "code"
    NAVIGATION = "navigation"
    ICON = "icon"
    IMAGE = "image"
    TABLE = "table"
    UNKNOWN = "unknown"


@dataclass
class UIRegion:
    region_type: UIRegionType
    bounds: Tuple[int, int, int, int]
    text: str = ""
    confidence: float = 0.0
    is_interactive: bool = False
    label: str = ""
    words: List[Dict] = field(default_factory=list)


@dataclass
class ScreenLayout:
    timestamp: float
    regions: List[UIRegion] = field(default_factory=list)
    full_text: str = ""
    window_title: str = ""
    dominant_type: UIRegionType = UIRegionType.UNKNOWN
    saliency_region: Optional[UIRegion] = None
    text_density_map: np.ndarray = None
    capture_num: int = 0
    screen_hash: str = ""


# --------------------------------------------------------------------------- #
# Layout Engine — contour-based UI region detection
# --------------------------------------------------------------------------- #

class LayoutEngine:
    """
    Detects UI region boundaries using cv2 contour analysis.
    """

    MIN_REGION_AREA = 800
    MAX_REGIONS = 40

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def find_regions(self, screenshot: 'Image.Image') -> List[UIRegion]:
        if not HAS_CV2:
            return []

        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scale = min(1.0, 1280 / max(img.shape[:2]))
        if scale < 1.0:
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
            scale_inv = 1.0 / scale
        else:
            scale_inv = 1.0

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 80, 200)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_REGION_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue

            x = int(x * scale_inv)
            y = int(y * scale_inv)
            w = int(w * scale_inv)
            h = int(h * scale_inv)

            is_interactive = self._detect_interactive(x, y, w, h, screenshot)

            regions.append(UIRegion(
                region_type=UIRegionType.UNKNOWN,
                bounds=(x, y, w, h),
                is_interactive=is_interactive,
                confidence=float(min(area * scale * scale / 5000, 1.0))
            ))

        regions = self._dedupe_overlaps(regions)
        regions.sort(key=lambda r: r.bounds[2] * r.bounds[3], reverse=True)
        return regions[:self.MAX_REGIONS]

    def _detect_interactive(self, x: int, y: int, w: int, h: int,
                           screenshot: 'Image.Image') -> bool:
        crop = screenshot.crop((x, y, min(x + w, screenshot.width),
                                min(y + h, screenshot.height)))
        gray = crop.convert("L")
        arr = np.array(gray)
        edges = cv2.Canny(arr, 80, 200)
        border_pixels = (
            np.concatenate([edges[:2, :].flatten(), edges[-2:, :].flatten()]) +
            np.concatenate([edges[:, :2].flatten(), edges[:, -2:].flatten()])
        )
        return np.sum(border_pixels > 0) / max(len(border_pixels), 1) > 0.08

    def _dedupe_overlaps(self, regions: List[UIRegion]) -> List[UIRegion]:
        filtered = []
        for r in regions:
            is_subregion = False
            for f in filtered:
                fx, fy, fw, fh = f.bounds
                rx, ry, rw, rh = r.bounds
                ox = max(0, min(fx + fw, rx + rw) - max(fx, rx))
                oy = max(0, min(fy + fh, ry + rh) - max(fy, ry))
                if ox * oy > 0.6 * (rw * rh):
                    is_subregion = True
                    break
            if not is_subregion:
                filtered.append(r)
        return filtered


# --------------------------------------------------------------------------- #
# OCR Engine — single-pass word-level extraction with bbox attribution
# --------------------------------------------------------------------------- #

class OCREngine:
    """
    One-shot OCR: gets all word bounding boxes in a single Tesseract call,
    then attributes words to UI regions by spatial overlap.
    """

    def __init__(self):
        self._last_words: List[Dict] = []
        self._last_full_text: str = ""

    def get_words(self, screenshot: 'Image.Image',
                  scale: float = 1.0) -> Tuple[str, List[Dict]]:
        """
        Single Tesseract call — returns (full_text, word_list).
        word_list: [{'text': str, 'x': int, 'y': int, 'w': int, 'h': int}]
        """
        if not HAS_OCR:
            return "", []

        try:
            config = "--psm 6 -c tessedit_pageseg_mode=6"
            data = pytesseract.image_to_data(
                screenshot, output_type=pytesseract.Output.DICT,
                config=config, timeout=5
            )

            words = []
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                if not text:
                    continue
                x = int(data["left"][i] * scale)
                y = int(data["top"][i] * scale)
                w = int(data["width"][i] * scale)
                h = int(data["height"][i] * scale)
                conf = float(data["conf"][i])
                if conf < 0 or w < 2 or h < 2:
                    continue
                words.append({
                    "text": text,
                    "x": x, "y": y,
                    "w": w, "h": h,
                    "conf": conf,
                })

            full_text = " ".join(w["text"] for w in words)
            self._last_words = words
            self._last_full_text = full_text
            return full_text, words

        except Exception:
            return "", []


# --------------------------------------------------------------------------- #
# UI Classifier — heuristic region type classification
# --------------------------------------------------------------------------- #

class UIClassifier:
    """
    Classifies UI regions using:
    - Visual properties (aspect ratio, border density, color variance)
    - Text content (keyword matching for code/button/nav)
    """

    CODE_CHARS = frozenset("{}[]();=>&|!-+*/<>#\\\"'")
    MONOSPACE_RATIO = 0.15

    INTERACTIVE_KW = frozenset({
        "button", "submit", "click", "login", "sign", "register",
        "search", "cancel", "delete", "save", "close", "ok", "apply",
        "reset", "upload", "download", "edit", "add", "create", "submit",
    })

    CODE_KW = frozenset({
        "def ", "function", "class ", "import ", "const ", "let ", "var ",
        "return ", "if (", "for (", "while (", "async ", "await ",
        "from ", "package ", "public ", "private ", "static ", "void ",
        "import ", "require(", "module", "exports",
    })

    NAV_KW = frozenset({
        "menu", "home", "settings", "profile", "account", "help",
        "file", "edit", "view", "window", "tools", "navigation",
    })

    def classify(self, region: UIRegion, screenshot: 'Image.Image',
                words: List[Dict]) -> UIRegion:
        """Classify region type and extract text via word attribution."""
        x, y, w, h = region.bounds
        area = w * h

        region_words = self._words_in_region(words, x, y, w, h)
        region.text = " ".join(w["text"] for w in region_words)
        region.words = region_words

        if len(region_words) > 0:
            region.label = region_words[0]["text"]
            if len(region_words) > 1:
                region.label += f" +{len(region_words)-1}"
        else:
            region.label = region.text[:80].strip() if region.text else ""

        text_lower = region.text.lower()
        text_words = set(text_lower.split())

        if region.is_interactive or text_words & self.INTERACTIVE_KW:
            region.region_type = UIRegionType.BUTTON

        elif self._is_code(region, region_words):
            region.region_type = UIRegionType.CODE

        elif len(region.text) > 80 and "\n" in region.text:
            region.region_type = UIRegionType.TEXT_BLOCK

        elif text_words & self.NAV_KW or self._is_nav_bar(w, h):
            region.region_type = UIRegionType.NAVIGATION

        elif self._is_icon(w, h, len(region_words), region.is_interactive):
            region.region_type = UIRegionType.ICON

        elif self._is_image(w, h, region_words):
            region.region_type = UIRegionType.IMAGE

        elif self._is_table(w, h, region_words):
            region.region_type = UIRegionType.TABLE

        elif self._is_input_field(w, h, region_words, region.is_interactive):
            region.region_type = UIRegionType.INPUT

        elif region.text.strip():
            region.region_type = UIRegionType.TEXT_BLOCK

        return region

    def _words_in_region(self, words: List[Dict],
                          rx: int, ry: int, rw: int, rh: int) -> List[Dict]:
        """Return words whose bounding box overlaps the region."""
        result = []
        for w in words:
            wx, wy, ww, wh = w["x"], w["y"], w["w"], w["h"]
            ox = max(0, min(rx + rw, wx + ww) - max(rx, wx))
            oy = max(0, min(ry + rh, wy + wh) - max(ry, wy))
            if ox * oy > 0:
                result.append(w)
        result.sort(key=lambda w: w["y"] * 10000 + w["x"])
        return result

    def _is_code(self, region: UIRegion, words: List[Dict]) -> bool:
        if not words:
            return False
        code_char_count = sum(1 for w in words for c in w["text"] if c in self.CODE_CHARS)
        total_chars = sum(len(w["text"]) for w in words)
        if total_chars == 0:
            return False
        code_ratio = code_char_count / total_chars
        return code_ratio > self.MONOSPACE_RATIO or (total_chars > 20 and code_ratio > 0.08)

    def _is_nav_bar(self, w: int, h: int) -> bool:
        return h > 0 and (w / h) > 6 and h < 80

    def _is_icon(self, w: int, h: int, word_count: int,
                 is_interactive: bool) -> bool:
        area = w * h
        if area == 0:
            return False
        aspect = w / h if h > 0 else 0
        is_square = 0.4 < aspect < 2.5
        is_small = area < 15000
        has_minimal_text = word_count <= 3
        return is_square and is_small and has_minimal_text and not is_interactive

    def _is_image(self, w: int, h: int, words: List[Dict]) -> bool:
        area = w * h
        if area < 10000:
            return False
        crop_area_pixels = w * h
        aspect = w / h if h > 0 else 0
        is_landscape = 0.5 < aspect < 3.0
        return is_landscape and len(words) == 0

    def _is_table(self, w: int, h: int, words: List[Dict]) -> bool:
        if len(words) < 6 or w == 0 or h == 0:
            return False
        ys = [w["y"] for w in words]
        y_variance = np.var(ys) if len(ys) > 1 else 0
        xs = [w["x"] for w in words]
        x_variance = np.var(xs) if len(xs) > 1 else 0
        return y_variance > 100 and x_variance > 100

    def _is_input_field(self, w: int, h: int, words: List[Dict],
                        is_interactive: bool) -> bool:
        if not is_interactive:
            return False
        has_borders = h > 0 and 20 < w < 800 and h < 80
        no_text = len(words) == 0
        placeholder_like = len(words) == 1 and len(words[0]["text"]) < 30
        return has_borders and (no_text or placeholder_like)


# --------------------------------------------------------------------------- #
# Saliency Engine — most important region
# --------------------------------------------------------------------------- #

class SaliencyEngine:
    """
    Scores regions by visual importance:
    - Text presence (words = attention)
    - Interactive (clickable things draw the eye)
    - Contrast (high contrast regions stand out)
    - Size (smaller = more specific/salient; very large = background)
    - Center bias (regions near screen center score higher)
    """

    def __init__(self):
        self._screen_center: Tuple[float, float] = (0.5, 0.5)

    def find_salient_region(self, screenshot: 'Image.Image',
                            regions: List[UIRegion],
                            screen_w: int = 1920,
                            screen_h: int = 1080) -> Optional[UIRegion]:
        if not regions:
            return None
        self._screen_center = (screen_w / 2, screen_h / 2)
        scored = [(self._salience_score(r), r) for r in regions]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _salience_score(self, region: UIRegion) -> float:
        x, y, w, h = region.bounds
        area = w * h

        text_score = min(len(region.text) / 100, 1.0) * 0.35
        text_score += 0.15 if region.words else 0.0

        interactive_score = 0.2 if region.is_interactive else 0.0

        cx = x + w / 2
        cy = y + h / 2
        dist = np.sqrt((cx - self._screen_center[0]) ** 2 +
                       (cy - self._screen_center[1]) ** 2)
        screen_diag = np.sqrt(self._screen_center[0]**2 + self._screen_center[1]**2)
        center_bias = 1.0 - min(dist / screen_diag, 1.0)
        center_score = center_bias * 0.1

        size_score = 1.0 / (1.0 + np.log1p(area) / 18)

        score = (text_score + interactive_score + center_score) * size_score
        return score


# --------------------------------------------------------------------------- #
# Screen UI Engine — unified pipeline
# --------------------------------------------------------------------------- #

class ScreenUIEngine:
    """
    Unified screen/UI understanding pipeline.

    Usage:
        engine = ScreenUIEngine()
        layout = engine.analyze(screenshot)
        # layout.full_text       - all text on screen
        # layout.regions         - list of UIRegion objects
        # layout.saliency_region - most important region
        # layout.dominant_type   - most common UI type on screen
    """

    def __init__(self):
        self.ocr = OCREngine()
        self.classifier = UIClassifier()
        self.saliency = SaliencyEngine()
        self._layout_engine: Optional[LayoutEngine] = None
        self._last_layout: Optional[ScreenLayout] = None
        self._last_screen_hash = ""
        self._lock = threading.Lock()
        self._text_encoder_cache = None
        self._vision_encoder_cache = None

    def analyze(self, screenshot: 'Image.Image',
                capture_num: int = 0,
                window_title: str = "") -> ScreenLayout:
        """
        Full pipeline: one-shot OCR → layout detection → UI classification → saliency.
        All OCR in a single Tesseract call (word-level with bounding boxes).
        """
        with self._lock:
            t0 = time.time()
            img_w, img_h = screenshot.size

            screen_hash = hashlib.md5(
                screenshot.tobytes()[:10000]
            ).hexdigest()[:12]

            if self._layout_engine is None:
                self._layout_engine = LayoutEngine(img_w, img_h)

            scale = 1.0
            ocr_img = screenshot
            if img_w > 1920:
                scale = 1920 / img_w
                ocr_img = screenshot.resize(
                    (int(img_w * scale), int(img_h * scale)), Image.LANCZOS
                )

            full_text, words = self.ocr.get_words(ocr_img, scale=scale)

            raw_regions = self._layout_engine.find_regions(screenshot)

            classified = []
            for region in raw_regions:
                classified.append(
                    self.classifier.classify(region, screenshot, words)
                )

            saliency_region = self.saliency.find_salient_region(
                screenshot, classified, img_w, img_h
            )

            type_counts: Dict[str, int] = {}
            for r in classified:
                rt = r.region_type.value
                type_counts[rt] = type_counts.get(rt, 0) + 1
            dominant = max(type_counts, key=type_counts.get) if type_counts else "unknown"
            try:
                dominant_type = UIRegionType(dominant)
            except ValueError:
                dominant_type = UIRegionType.UNKNOWN

            text_density_map = self._build_text_density_map(img_w, img_h, classified)

            layout = ScreenLayout(
                timestamp=time.time(),
                regions=classified,
                full_text=full_text,
                window_title=window_title,
                dominant_type=dominant_type,
                saliency_region=saliency_region,
                text_density_map=text_density_map,
                capture_num=capture_num,
                screen_hash=screen_hash,
            )

            self._last_layout = layout
            self._last_screen_hash = screen_hash
            return layout

    def _build_text_density_map(self, img_w: int, img_h: int,
                                regions: List[UIRegion]) -> np.ndarray:
        density = np.zeros((16, 16), dtype=np.float32)
        for region in regions:
            if not region.text:
                continue
            x, y, w, h = region.bounds
            cx = int(np.clip((x + w // 2) / img_w * 16, 0, 15))
            cy = int(np.clip((y + h // 2) / img_h * 16, 0, 15))
            weight = min(len(region.text) / 100, 2.0)
            density[cy, cx] += weight
        if density.max() > 0:
            density /= density.max()
        return density

    def get_text_summary(self, layout: ScreenLayout) -> str:
        parts = []
        if layout.window_title:
            parts.append(f"Window: {layout.window_title}")
        if layout.saliency_region:
            salient = layout.saliency_region
            label = salient.label or salient.text[:60]
            parts.append(f"Focus: [{salient.region_type.value}] {label}")
        type_counts: Dict[str, int] = {}
        for r in layout.regions:
            rt = r.region_type.value
            type_counts[rt] = type_counts.get(rt, 0) + 1
        if type_counts:
            top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:6]
            types_str = ", ".join(f"{c}×{t}" for t, c in top_types)
            parts.append(f"UI: {types_str}")
        if layout.full_text:
            parts.append(f"Text: {len(layout.full_text)} chars, {len(layout.regions)} regions")
        return " | ".join(parts)

    def encode_for_brain(self, layout: ScreenLayout) -> dict:
        from sensory.text import TextEncoder
        from sensory.vision import VisionEncoder

        if self._text_encoder_cache is None:
            self._text_encoder_cache = TextEncoder(feature_dim=256)
        if self._vision_encoder_cache is None:
            self._vision_encoder_cache = VisionEncoder(feature_dim=256)

        summary = self.get_text_summary(layout)
        text_features = self._text_encoder_cache.encode(summary)

        vision_features = np.zeros(256, dtype=np.float32)
        if layout.text_density_map is not None:
            flat = layout.text_density_map.flatten()
            vision_features[:min(len(flat), 256)] = flat[:256]

        if layout.saliency_region:
            x, y, w, h = layout.saliency_region.bounds
            focus_enc = np.zeros(256, dtype=np.float32)
            focus_enc[0] = x / 3840
            focus_enc[1] = y / 2160
            focus_enc[2] = (w * h) / (3840 * 2160)
            if layout.saliency_region.text:
                focus_enc[3] = min(len(layout.saliency_region.text) / 200, 1.0)
            vision_features = vision_features * 0.7 + focus_enc * 0.3

        return {
            "text_features": text_features,
            "vision_features": vision_features,
            "ui_features": self._encode_ui_types(layout),
            "summary": summary,
        }

    def _encode_ui_types(self, layout: ScreenLayout) -> np.ndarray:
        type_order = ["button", "input", "text_block", "code", "navigation", "icon", "image", "table", "unknown"]
        ui_vec = np.zeros(len(type_order), dtype=np.float32)
        for r in layout.regions:
            try:
                idx = type_order.index(r.region_type.value)
                ui_vec[idx] += 1
            except ValueError:
                pass
        if ui_vec.max() > 0:
            ui_vec /= ui_vec.max()
        return ui_vec

    def has_screen_changed(self, screenshot: 'Image.Image') -> bool:
        h = hashlib.md5(screenshot.tobytes()[:10000]).hexdigest()[:12]
        return h != self._last_screen_hash
