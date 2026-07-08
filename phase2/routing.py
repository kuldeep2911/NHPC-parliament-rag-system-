"""
Per-page routing for combined-image PDFs.

A single PDF may mix page kinds:
    digital      real text layer            -> parse_document() (Docling/pdfplumber)
    scanned      little/no text, full image -> ocr_image()      (provider OCR)
    image_based  text baked inside an image -> parse_visual()   (provider VLM)

Classification uses extractable-text length + image-coverage, NOT the file
extension. Each page yields a PageRoute recording the decision and the heuristic
values, which is surfaced in parsed.json (page_routing[]) and in the trace.

The heuristics are computed with pdfplumber (already a dependency). If pdfplumber
is unavailable the whole PDF degrades to a single 'digital' route so the reader
still parses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageRoute:
    page: int
    page_type: str                 # digital | scanned | image_based
    reason: str                    # human-readable decision reason
    text_chars: int = 0
    image_coverage: float = 0.0    # image area / page area (0..1)
    n_images: int = 0
    route: str = "parse_document"  # parse_document | ocr_image | parse_visual

    def to_dict(self):
        return {
            "page": self.page,
            "page_type": self.page_type,
            "route": self.route,
            "reason": self.reason,
            "text_chars": self.text_chars,
            "image_coverage": round(self.image_coverage, 3),
            "n_images": self.n_images,
        }


ROUTE_FOR_TYPE = {
    "digital": "parse_document",
    "scanned": "ocr_image",
    "image_based": "parse_visual",
}


def _page_image_coverage(page) -> tuple[float, int]:
    """Return (image_area_fraction, n_images) for a pdfplumber page."""
    try:
        pw = float(page.width) or 1.0
        ph = float(page.height) or 1.0
        page_area = pw * ph
        images = getattr(page, "images", []) or []
        covered = 0.0
        for im in images:
            w = float(im.get("width") or abs((im.get("x1", 0) - im.get("x0", 0))))
            h = float(im.get("height") or abs((im.get("bottom", 0) - im.get("top", 0))))
            covered += max(0.0, w) * max(0.0, h)
        return min(1.0, covered / page_area), len(images)
    except Exception:
        return 0.0, 0


def classify_pages(path: str, cfg) -> list:
    """
    Return a list[PageRoute], one per PDF page. Decides each page independently:

      - enough real text            -> digital   (parse_document)
      - sparse text + big image      -> image_based (parse_visual / VLM)
      - sparse text + no/small image -> scanned   (ocr_image)
    """
    try:
        import pdfplumber
        import warnings
        warnings.filterwarnings("ignore")
    except Exception:
        return [PageRoute(page=1, page_type="digital", route="parse_document",
                          reason="pdfplumber unavailable; default digital")]

    routes = []
    try:
        with pdfplumber.open(path) as pdf:
            for pi, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                nchars = len(text)
                coverage, nimg = _page_image_coverage(page)

                if nchars >= cfg.scanned_char_threshold_per_page:
                    pt, reason = "digital", (
                        f"text_chars={nchars} >= {cfg.scanned_char_threshold_per_page}")
                elif coverage >= cfg.image_coverage_threshold and nchars < cfg.visual_min_text_chars:
                    pt, reason = "image_based", (
                        f"image_coverage={coverage:.2f} >= {cfg.image_coverage_threshold} "
                        f"with sparse text ({nchars} chars) -> text likely inside image")
                else:
                    pt, reason = "scanned", (
                        f"sparse text ({nchars} chars) and image_coverage={coverage:.2f} "
                        f"< {cfg.image_coverage_threshold} -> scanned page")

                routes.append(PageRoute(
                    page=pi, page_type=pt, route=ROUTE_FOR_TYPE[pt], reason=reason,
                    text_chars=nchars, image_coverage=coverage, n_images=nimg))
    except Exception as e:
        return [PageRoute(page=1, page_type="digital", route="parse_document",
                          reason=f"page classification failed ({type(e).__name__}); default digital")]

    if not routes:
        routes = [PageRoute(page=1, page_type="digital", route="parse_document",
                            reason="no pages read; default digital")]
    return routes


def summarize_routing(routes) -> dict:
    """Compact per-document summary of page routing for flags/report."""
    from collections import Counter
    c = Counter(r.page_type for r in routes)
    return {
        "n_pages": len(routes),
        "digital": c.get("digital", 0),
        "scanned": c.get("scanned", 0),
        "image_based": c.get("image_based", 0),
        "mixed": len([k for k in ("digital", "scanned", "image_based") if c.get(k)]) > 1,
    }
