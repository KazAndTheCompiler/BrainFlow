"""
Text → Spike Encoding

Converts text input into spike-compatible feature vectors.

Strategy (no external NLP libraries required):
- TF-IDF-inspired word importance weighting (common words suppressed, rare words amplified)
- Structural features (length, word count, punctuation density, caps ratio)
- Concept category detection (code, ui, natural_language) as orthogonal feature groups
- Positional encoding (first/last word bonus)
- Smooth temporal accumulation (recent words weighted higher)
"""

import re
import numpy as np
from collections import Counter
from typing import Optional


STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "whose",
    "if", "then", "else", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "same", "so", "than", "too", "very", "just",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "once",
})


CONCEPT_CATEGORIES = {
    "code": {
        "def ", "function", "class ", "import ", "const ", "let ", "var ",
        "return ", "if (", "for (", "while (", "async ", "await ", "from ",
        "package ", "public ", "private ", "static ", "void ", "require(",
        "module", "exports", "=>", "->", "//", "/*", "*/", "===", "!==",
        "==", "!=", "&&", "||", "++", "--", "+=", "-=", "(", ")",
        "[", "]", "{", "}", ";", ":", ",",
    },
    "ui_interactive": {
        "button", "click", "submit", "login", "sign", "register", "search",
        "cancel", "delete", "save", "close", "ok", "apply", "reset",
        "upload", "download", "edit", "add", "create", "menu", "nav",
        "settings", "profile", "home", "back", "next", "forward",
    },
    "ui_static": {
        "window", "panel", "sidebar", "header", "footer", "toolbar",
        "dialog", "modal", "popup", "dropdown", "tab", "field", "label",
        "table", "row", "column", "cell", "icon", "image", "text",
    },
    "data": {
        "number", "count", "total", "sum", "average", "percent", "%",
        "result", "output", "input", "value", "data", "record", "file",
    },
    "action": {
        "clicked", "typed", "selected", "scrolled", "opened", "closed",
        "running", "loading", "error", "warning", "success", "failed",
        "completed", "started", "stopped", "waiting",
    },
}


class TextEncoder:
    """
    Encodes text into 256-dim spike-compatible feature vectors.

    Feature layout (256 total, 4 quarters of 64):
    - Q0 (0-63):   Word importance vector (TF-IDF weighted)
    - Q1 (64-127): Concept category activations (5 categories)
    - Q2 (128-191): Structural features (length, word stats, punctuation)
    - Q3 (192-255): Positional encoding (start/end emphasis)
    """

    Q_SIZE = 64
    MAX_WORDS = 32
    MAX_CHARS = 256

    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        self._corpus: Counter = Counter()
        self._corpus_size: int = 0
        self._total_docs: int = 0
        self._last_text: str = ""
        self._last_features: Optional[np.ndarray] = None

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9#'+-]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    def _idf(self, word: str) -> float:
        if self._total_docs == 0:
            return 1.0
        doc_freq = max(1, self._corpus.get(word, 0))
        return np.log((self._total_docs + 1) / doc_freq)

    def _detect_concepts(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        concept_vec = np.zeros(len(CONCEPT_CATEGORIES), dtype=np.float32)
        for i, (category, keywords) in enumerate(CONCEPT_CATEGORIES.items()):
            matches = sum(1 for kw in keywords if kw in text_lower)
            concept_vec[i] = min(matches / 3.0, 1.0)
        return concept_vec

    def _structural_features(self, text: str, tokens: list[str]) -> np.ndarray:
        struct = np.zeros(self.Q_SIZE, dtype=np.float32)

        if len(text) == 0:
            return struct

        struct[0] = min(len(text) / 200.0, 1.0)
        struct[1] = min(len(tokens) / 30.0, 1.0)

        alpha = sum(1 for c in text if c.isalpha())
        upper = sum(1 for c in text if c.isupper())
        digit = sum(1 for c in text if c.isdigit())
        punct = sum(1 for c in text if c in ".,!?;:()[]{}")
        space = sum(1 for c in text if c.isspace())

        total = max(len(text), 1)
        struct[2] = alpha / total
        struct[3] = upper / max(alpha, 1) if alpha > 0 else 0.0
        struct[4] = digit / total
        struct[5] = punct / total
        struct[6] = space / total
        struct[7] = sum(1 for c in text if c == "\n") / max(text.count("\n"), 1) if text.count("\n") > 0 else 0.0

        if tokens:
            avg_len = np.mean([len(t) for t in tokens])
            struct[8] = min(avg_len / 8.0, 1.0)
            max_len = max(len(t) for t in tokens)
            struct[9] = min(max_len / 15.0, 1.0)
        else:
            struct[8] = 0.0
            struct[9] = 0.0

        q_marks = text.count("?")
        ex_marks = text.count("!")
        period_cnt = text.count(".")
        struct[10] = min(q_marks / 5.0, 1.0)
        struct[11] = min(ex_marks / 3.0, 1.0)
        struct[12] = min(period_cnt / 10.0, 1.0)

        code_chars = sum(1 for c in text if c in "{}[]();:+-*/<>&|=#")
        struct[13] = min(code_chars / max(len(text), 1) * 5, 1.0)

        word_set = set(tokens)
        stop_ratio = len([w for w in tokens if w in STOP_WORDS]) / max(len(tokens), 1)
        struct[14] = stop_ratio

        struct[15] = 1.0 if text.isupper() else 0.0
        struct[16] = 1.0 if text.islower() else 0.0
        struct[17] = 1.0 if any(c.isdigit() for c in text) else 0.0
        struct[18] = 1.0 if any(c.isalpha() for c in text) else 0.0

        return struct

    def encode(self, text: str) -> np.ndarray:
        if not text or len(text.strip()) == 0:
            return np.array([], dtype=np.float32)

        text = text[:self.MAX_CHARS]
        tokens = self._tokenize(text)

        word_vec = np.zeros(self.Q_SIZE, dtype=np.float32)
        pos_vec = np.zeros(self.Q_SIZE, dtype=np.float32)

        word_scores: dict[str, float] = {}
        for i, word in enumerate(tokens[: self.MAX_WORDS]):
            idf = self._idf(word)
            pos_weight = np.exp(-i * 0.08)
            word_scores[word] = idf * pos_weight

        if word_scores:
            max_score = max(word_scores.values())
            if max_score > 0:
                for i, word in enumerate(tokens[: self.MAX_WORDS]):
                    score = word_scores[word] / max_score
                    idx = min(i, self.Q_SIZE - 1)
                    word_vec[idx] = max(word_vec[idx], score)

        if tokens:
            num_tokens = min(len(tokens), self.MAX_WORDS)
            pos_decay = 1.0 / (1.0 + np.arange(num_tokens) * 0.1)
            pos_vec[:num_tokens] = pos_decay[:num_tokens]

        concept_vec = self._detect_concepts(text)
        while len(concept_vec) < self.Q_SIZE:
            concept_vec = np.append(concept_vec, 0.0)
        concept_vec = concept_vec[: self.Q_SIZE]

        struct_vec = self._structural_features(text, tokens)

        features = np.zeros(self.feature_dim, dtype=np.float32)
        features[0:64] = word_vec
        features[64:128] = concept_vec
        features[128:192] = struct_vec
        features[192:256] = pos_vec

        features = np.clip(features, 0, 1)

        self._last_text = text
        self._last_features = features.copy()
        return features

    def update_corpus(self, texts: list[str]):
        for text in texts:
            tokens = self._tokenize(text)
            for word in tokens:
                self._corpus[word] += 1
            self._total_docs += 1
        self._corpus_size = sum(self._corpus.values())

    def get_last_features(self) -> Optional[np.ndarray]:
        return self._last_features.copy() if self._last_features is not None else None

    def get_text_stats(self, text: str) -> dict:
        tokens = self._tokenize(text)
        return {
            "chars": len(text),
            "tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "stop_word_ratio": len([t for t in tokens if t in STOP_WORDS]) / max(len(tokens), 1),
            "concept_code": sum(1 for kw in CONCEPT_CATEGORIES["code"] if kw in text.lower()) / 3.0,
            "concept_ui": sum(1 for kw in CONCEPT_CATEGORIES["ui_interactive"] if kw in text.lower()) / 3.0,
        }
