"""
Measure the ACTUAL RRF score distribution on this corpus, so WIDEN_TAU / WIDEN_DELTA are
set against the real scale instead of an imagined 0-1 one.

    python -m phase4.scripts.measure_rrf

RRF scores are not normalised: with rrf_k=60 a single retriever at rank 1 contributes
weight/(60+1). This script runs a spread of realistic queries -- strong (should score
high), vague (should score low), entity-bearing, and Hindi -- and prints the observed
top_score and #1-#2 gap for each, plus the recommended thresholds.
"""

from __future__ import annotations

import sys

from phase3.db import connect
from phase3.embeddings import get_embedder
from phase4.config import Phase4Config, load_dotenv
from phase4.retrieval import dense, entity, keyword, fuse

# A spread of query shapes. The point is to see where GOOD queries land versus BAD ones,
# because tau/delta must separate them.
QUERIES = [
    ("strong  ", "electricity dues outstanding against Jammu and Kashmir power departments"),
    ("strong  ", "details of hydroelectric projects under construction in Himachal Pradesh"),
    ("strong  ", "CSR expenditure by NHPC in the last five years"),
    ("entity  ", "Chamera power station generation"),
    ("entity  ", "Subansiri Lower project status"),
    ("medium  ", "renewable energy capacity addition"),
    ("vague   ", "details thereof"),
    ("vague   ", "the steps taken by the Government"),
    ("nonsense", "zxcv qwerty asdf nothing relevant here"),
    ("hindi   ", "जम्मू और कश्मीर में बिजली के बकाया राशि का विवरण"),
    ("hindi   ", "जलविद्युत परियोजना की जानकारी"),
]


def main():
    load_dotenv()
    cfg = Phase4Config()
    errs = cfg.validate_phase4()
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1

    embedder = get_embedder(cfg)

    max_single = cfg.rrf_weight_dense / (cfg.rrf_k + 1)
    max_all = ((cfg.rrf_weight_dense + cfg.rrf_weight_keyword + cfg.rrf_weight_entity)
               / (cfg.rrf_k + 1))
    print("=" * 78)
    print("RRF SCALE (rrf_k=%d, weights dense=%.2f keyword=%.2f entity=%.2f)"
          % (cfg.rrf_k, cfg.rrf_weight_dense, cfg.rrf_weight_keyword, cfg.rrf_weight_entity))
    print("=" * 78)
    print(f"  a lone DENSE hit at rank 1        : {max_single:.5f}")
    print(f"  rank 1 in ALL THREE lists (max)   : {max_all:.5f}")
    print(f"  a lone dense hit at rank 10       : {cfg.rrf_weight_dense / (cfg.rrf_k + 10):.5f}")
    print()

    rows = []
    with connect(cfg) as conn:
        vocab = entity.load_vocabulary(conn)
        print(f"entity vocabulary: {len(vocab)} distinct entities\n")
        print(f"{'kind':9} {'top_score':>10} {'gap':>9} {'elig':>5} {'fired':>6} "
              f"{'cands':>6}  query")
        print("-" * 78)
        for kind, q in QUERIES:
            qvec = embedder.embed_queries([q])[0]      # QUERY mode
            ents = entity.extract_entities(q, vocab)

            lists = {
                "dense": dense.search(conn, qvec, cfg.dense_top_n),
                "keyword": keyword.search(conn, q, cfg.keyword_top_n),
                "entity": entity.search(conn, ents, cfg.entity_top_n),
            }
            eligible = {"dense", "keyword"} | ({"entity"} if ents else set())
            fired = {r for r in eligible if lists.get(r)}

            fused, stats = fuse.fuse(lists, cfg, eligible, fired)
            rows.append((kind, stats["top_score"], stats["score_gap"]))
            print(f"{kind} {stats['top_score']:>10.5f} {stats['score_gap']:>9.5f} "
                  f"{stats['n_eligible']:>5} {stats['n_fired']:>6} "
                  f"{stats['n_candidates']:>6}  {q[:34]}")

    good = [t for k, t, _ in rows if k.strip() in ("strong", "entity")]
    bad = [t for k, t, _ in rows if k.strip() in ("vague", "nonsense")]
    gaps_good = [g for k, _, g in rows if k.strip() in ("strong", "entity")]

    print()
    print("=" * 78)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 78)
    if good and bad:
        print(f"  strong/entity top_score : min={min(good):.5f}  max={max(good):.5f}")
        print(f"  vague/nonsense top_score: min={min(bad):.5f}  max={max(bad):.5f}")
        print(f"  strong/entity gap       : min={min(gaps_good):.5f}")
        print()
        print(f"  WIDEN_TAU   (current {cfg.widen_tau:.5f}) — widen when top_score is below this")
        print(f"  WIDEN_DELTA (current {cfg.widen_delta:.5f}) — widen when the #1-#2 gap is below this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
