from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


RRF_K = 60                  
RERANK_RRF_K = 10
CANDIDATE_DEPTH = 1000      


@dataclass
class RetrievalConfig:
    depth: int = CANDIDATE_DEPTH
    min_score: float = 0.0

    # near-duplicate suppression on the fragment list
    dedupe_threshold: float = 0.45
    dedupe_window: int = 200

    phenomenon_boost: float = 0.20       # fraction of the RRF pool's score span
    phenomenon_mode: str = "add"         # add | multiply -- add is span-scaled
    phenomenon_boost_doc: float = 0.30   # ADDITIVE, as a FRACTION of the score span

    bm25_weight: float = 0.5
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    graph_weight: float = 0.5   # 0 disables the graph channel; see bm25_weight
    graph_neighbour: float = 0.4
    rm3_terms: int = 0          # 0 disables pseudo-relevance feedback
    rm3_feedback: int = 10
    rm3_original_weight: float = 0.6
    mmr_lambda: float = 1.0     # 1.0 disables diversification

    doc_score: str = "rankdecay"     # rankdecay | combsum | cosine
    doc_decay: float = 0.85          # per-chunk decay for rankdecay, see below
    doc_pool: int = 30
    doc_agg: str = "max"
    doc_top_m: int = 3
    doc_hit_bonus: float = 0.02
    doc_hit_cap: int = 3

    # optional cross-encoder rerank (see rerank() for the compliance note)
    reranker: str = "BAAI/bge-reranker-v2-m3"
    rerank_depth: int = 150
    rerank_blend: float = 0.5
    rerank_context: bool = True

    weights: list[float] = field(default_factory=list)


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    """Every knob that changes ranking. Imported by evaluar.py verbatim."""
    d = RetrievalConfig()
    parser.add_argument("--index-dir", type=Path, default=Path("base_vectorial"),
                        help="Folder CONTAINING the encoder_*/ subfolders, "
                             "each with index.faiss + metadata.jsonl. "
                             "Resolved against the current directory first, "
                             "then against this script's own directory.")
    parser.add_argument("--depth", type=int, default=d.depth)
    parser.add_argument("--min-score", type=float, default=d.min_score)
    parser.add_argument("--dedupe-threshold", type=float, default=d.dedupe_threshold,
                        help="Shingle containment above which a fragment is a "
                             "repeat of one already selected. 0 disables. "
                             "Measured: adjacent windows ~0.31, repeated "
                             "boilerplate ~0.95. Below 0.31 you start "
                             "suppressing neighbours; sweep it.")
    parser.add_argument("--phenomenon-boost", type=float, default=d.phenomenon_boost,
                        help="Bonus for chunks whose `fenomeno` matches the "
                             "query-id range, in the space set by "
                             "--phenomenon-mode.")
    parser.add_argument("--phenomenon-mode", choices=["add", "multiply"],
                        default=d.phenomenon_mode,
                        help="add (default) = span-scaled, same unit as "
                             "--phenomenon-boost-doc. multiply = old "
                             "behaviour, kept for ablation only -- see "
                             "apply_phenomenon_boost().")
    parser.add_argument("--phenomenon-boost-doc", type=float,
                        default=d.phenomenon_boost_doc,
                        help="Bonus in cosine/CombSUM space, expressed as a "
                             "FRACTION OF THE POOL'S SCORE SPAN so that one "
                             "value means the same thing in either space. "
                             "0.30 = 30%% of the range between the best and "
                             "worst candidate.")
    parser.add_argument("--graph-weight", type=float, default=d.graph_weight,
                        help="weight of the knowledge-graph channel in the "
                             "fusion (8.5). 0 disables it. Loaded "
                             "automatically from base_vectorial/grafo/"
                             "grafo.graphml when present.")
    parser.add_argument("--graph-neighbour", type=float, default=d.graph_neighbour,
                        help="discount applied to chunks reached through a "
                             "first-order neighbour rather than the entity "
                             "itself")
    parser.add_argument("--bm25-weight", type=float, default=d.bm25_weight,
                        help="Weight of the lexical channel. 0 disables it "
                             "and skips the index build entirely.")
    parser.add_argument("--rm3-terms", type=int, default=d.rm3_terms,
                        help="RM3 pseudo-relevance feedback: expansion terms "
                             "taken from the lexical channel's own top hits. "
                             "0 disables. No model involved, so 8.3's ban on "
                             "decoder-based query expansion does not apply.")
    parser.add_argument("--rm3-feedback", type=int, default=d.rm3_feedback)
    parser.add_argument("--mmr-lambda", type=float, default=d.mmr_lambda,
                        help="Maximal Marginal Relevance on the fragment "
                             "list. 1.0 = off. Lower trades relevance for "
                             "coverage, which is what nDCG@10 rewards when a "
                             "query has several relevant passages.")
    parser.add_argument("--doc-score", choices=["rankdecay", "combsum", "cosine"],
                        default=d.doc_score,
                        help="Score space documents are aggregated over (8.6). "
                             "rankdecay (default, see RetrievalConfig for why) "
                             "ignores --doc-pool/--doc-agg/--doc-hit-* entirely "
                             "and uses --doc-decay instead; combsum/cosine are "
                             "kept for ablation against the old behaviour and "
                             "use the doc-pool/doc-agg/hit-bonus knobs below.")
    parser.add_argument("--doc-decay", type=float, default=d.doc_decay,
                        help="rankdecay only: weight of a document's i-th "
                             "ranked chunk is doc_decay**i. Lower rewards "
                             "breadth less; 1.0 makes every chunk count "
                             "equally (closer to old doc_agg='sum').")
    parser.add_argument("--doc-agg", choices=["max", "sum", "mean", "rrf"],
                        default=d.doc_agg,
                        help="how a document's chunk scores become one score "
                             "(8.6). max is pure best-chunk and makes "
                             "--doc-pool inert; sum/mean/rrf reward a document "
                             "with several good passages.")
    parser.add_argument("--doc-top-m", type=int, default=d.doc_top_m,
                        help="chunks per document combined by sum/mean/rrf")
    parser.add_argument("--doc-pool", type=int, default=d.doc_pool,
                        help="k_chunk: chunks aggregated into document scores.")
    parser.add_argument("--doc-hit-bonus", type=float, default=d.doc_hit_bonus,
                        help="Per-extra-chunk bonus in the document score. The "
                             "old 0.05 x 5 = 25%% dwarfed the ~11%% spread of "
                             "the cosine scores it multiplied, which made "
                             "document ranking a chunk-count contest.")
    parser.add_argument("--doc-hit-cap", type=int, default=d.doc_hit_cap)
    parser.add_argument("--doc-encoder", default="",
                        help="Substring of the encoder that leads the store "
                             "list, e.g. 'e5'. Default: alphabetical.")
    parser.add_argument("--reranker", default=d.reranker,
                        help="Optional cross-encoder, e.g. "
                             "'BAAI/bge-reranker-v2-m3'. Empty = off. Read the "
                             "compliance note in rerank() before enabling.")
    parser.add_argument("--rerank-depth", type=int, default=d.rerank_depth)
    parser.add_argument("--no-rerank-context", action="store_false",
                        dest="rerank_context", default=d.rerank_context,
                        help="score the cross-encoder on `texto` alone "
                             "instead of `contexto. texto`")
    parser.add_argument("--rerank-blend", type=float, default=d.rerank_blend,
                        help="1.0 = cross-encoder order replaces the "
                             "retrievers'. 0.5 = equal RRF vote. Replacing "
                             "outright demotes chunks every retriever agreed "
                             "on; see rerank().")


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_input(path: Path) -> Path:

    if path.exists():
        return path
    beside = SCRIPT_DIR / path
    if beside.exists():
        return beside
    raise SystemExit(
        f"Not found: {path}\n"
        f"  looked in {path.resolve()}\n"
        f"       and {beside}\n"
        f"  --index-dir must point at the folder CONTAINING the encoder_*/ "
        f"subfolders,\n  e.g. entrega/base_vectorial from the repo root, or "
        f"base_vectorial from inside entrega/.")


def config_from_args(args) -> RetrievalConfig:
    cfg = RetrievalConfig()
    for name in vars(cfg):
        if hasattr(args, name):
            setattr(cfg, name, getattr(args, name))
    return cfg


_ABBREV = (r"Sr|Sra|Srta|Dr|Dra|Ing|Lic|Mg|Prof|Ph\.D|EE\.UU|EEUU|etc|vs|cf|"
           r"p\.ej|aprox|núm|No|Nro|Art|Fig|Tab|Cap|Vol|ed|eds|al|Mr|Mrs|Ms|"
           r"St|Jr|Inc|Ltd|Co|U\.S|U\.K|e\.g|i\.e|approx|Ref|Eq")

_PROTECT = [
    (re.compile(r"\b(?:[A-ZÁÉÍÓÚÑ]{1,2}\.){2,}"),
     lambda m: m.group(0).replace(".", "@@")),
    (re.compile(rf"\b({_ABBREV})\.", re.IGNORECASE), r"\1@@"),
    (re.compile(r"\b(\d+)\.(\d)"), r"\1@@\2"),
    (re.compile(r"\b([A-ZÁÉÍÓÚÑ])\.(?=\s*[A-ZÁÉÍÓÚÑ])"), r"\1@@"),
]

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'»)\]]*\s+")
_SEGMENTER: object | None | bool = None


def _get_segmenter():
    """pysbd if installed and not disabled; built once per process."""
    global _SEGMENTER
    if _SEGMENTER is None:
        if os.environ.get("CODEFEST_NO_PYSBD"):
            _SEGMENTER = False
        else:
            try:
                import pysbd
                _SEGMENTER = pysbd.Segmenter(language="es", clean=False)
            except Exception:
                _SEGMENTER = False
    return _SEGMENTER or None


def split_sentences(text: str) -> list[str]:

    segmenter = _get_segmenter()
    if segmenter is not None:
        try:
            out = []
            for paragraph in text.split("\n\n"):
                if paragraph.strip():
                    out += [s.strip() for s in segmenter.segment(paragraph) if s.strip()]
            return out
        except Exception:
            pass
    out = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        masked = paragraph
        for pattern, replacement in _PROTECT:
            masked = pattern.sub(replacement, masked)
        for sentence in _SENTENCE_END.split(masked):
            sentence = sentence.replace("@@", ".").strip()
            if sentence:
                out.append(sentence)
    return out


def split_to_250_words(text: str, limit: int = 250) -> list[str]:

    if len(text.split()) <= limit:
        return [text]

    parts, current, n_words = [], [], 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if current and n_words + words > limit:
            parts.append(" ".join(current))
            current, n_words = [], 0
        current.append(sentence)
        n_words += words
    if current:
        parts.append(" ".join(current))

    # A single sentence over 250 words is pathological; hard-cut it rather
    # than emit an illegal fragment. 9.3.2 discards oversized fragments.
    final = []
    for part in parts:
        words = part.split()
        if len(words) <= limit:
            final.append(part)
        else:
            for i in range(0, len(words), limit):
                final.append(" ".join(words[i:i + limit]))
    return final



_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """
    PDF extraction and OCR disagree about accents constantly, and Spanish
    queries are typed both ways. "informacion" and "información" are the same
    token for retrieval purposes.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")

_LINE_SEPARATORS = re.compile("[\u0085\u2028\u2029]")


def sanitize_text(text: str) -> str:
    """Replace the invisible separators with a plain space."""
    return _LINE_SEPARATORS.sub(" ", text)


def read_jsonl(path: Path) -> list[dict]:
    """
    Read JSON Lines safely. Splits on "\n" ONLY -- see _LINE_SEPARATORS.
    9.3 defines the newline as the sole delimiter, so this is also the
    literal reading of the spec.
    """
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]

_STOPWORDS = set("""
de la que el en y a los del se las por un para con no una su al lo como mas
pero sus le ya o este si porque esta entre cuando muy sin sobre tambien me
hasta hay donde quien desde todo nos durante todos uno les ni contra otros
ese eso ante ellos e esto mi antes algunos qué unos yo otro otras otra él
tanto esa estos mucho quienes nada muchos cual sea poco ella estar haber
estas estaba estamos algunas algo nosotros
son ser sido siendo esta estan estos estas fue fueron era eran han ha hemos
cuales cuál cual como cuando donde cuanto cuantos quienes cuyo cuya cuyos
que qué segun asi tal tales cada mismo misma mismos mismas dentro fuera
puede pueden podria podrian debe deben hace hacen tiene tienen
the of and to in a is for on that with as by at from or an be are this it
was were which has have had not but their its can will more other such
o e do da em para com no na os as um uma dos das ao pelo pela por mais como
mas ou se que nao ser sao foi ate entre sobre
""".split())


def tokenize(text: str) -> list[str]:
    """Accent-free lowercase word tokens, stopwords and 1-char tokens removed."""
    lowered = strip_accents(text.lower())
    words = _WS.sub(" ", _PUNCT.sub(" ", lowered)).split()
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


def shingles(text: str, n: int = 8) -> set[tuple]:
    """Accent- and punctuation-insensitive word n-grams, for dedupe only."""
    lowered = strip_accents(text.lower())
    words = _WS.sub(" ", _PUNCT.sub(" ", lowered)).strip().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class VectorStore:
    def __init__(self, folder: Path):
        import faiss
        from sentence_transformers import SentenceTransformer

        config = json.loads((folder / "encoder.json").read_text(encoding="utf-8"))
        self.folder = folder
        self.name = config["model"]
        self.query_prefix = config.get("query_prefix", "")
        # Some encoders ship a custom architecture whose modelling code lives
        # in the HF repo rather than in transformers. build_index.py records
        # the flag in encoder.json so it cannot drift between index time and
        # query time.
        self.model = SentenceTransformer(
            self.name, trust_remote_code=bool(config.get("trust_remote_code")))
        self.index = faiss.read_index(str(folder / "index.faiss"))
        # line i of metadata.jsonl <-> FAISS internal id i (1.4)
        with (folder / "metadata.jsonl").open(encoding="utf-8") as fh:
            self.metadata = [json.loads(line) for line in fh if line.strip()]
        if self.index.ntotal != len(self.metadata):
            raise SystemExit(
                f"{folder}: {self.index.ntotal} vectors vs "
                f"{len(self.metadata)} metadata lines. The index and the "
                f"metadata store are not in the same order; rebuild.")

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        # Same encoder, same prefix, same normalisation as indexing time (8.1)
        vector = self.model.encode([self.query_prefix + query],
                                   convert_to_numpy=True,
                                   normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vector, min(k, self.index.ntotal))
        return [(self.metadata[i], float(s))
                for i, s in zip(ids[0], scores[0]) if i != -1]


def load_stores(index_dir: Path, doc_encoder: str = "",
                doc_score: str = "") -> list[VectorStore]:
    """
    Load every encoder_* folder. The FIRST store leads: its cosine scores are
    what `--doc-score cosine` aggregates over, and its metadata is what the
    lexical index is built from. `doc_encoder` (a substring of the model
    name) moves a chosen encoder to the front, because alphabetical folder
    order is an arbitrary basis for a scored decision.
    """
    folders = sorted(p for p in index_dir.iterdir()
                     if p.is_dir() and (p / "index.faiss").exists())
    if not folders:
        raise SystemExit(f"No encoder_*/index.faiss found under {index_dir}")

    stores = [VectorStore(f) for f in folders]

    if doc_encoder:
        stores.sort(key=lambda s: doc_encoder.lower() not in s.name.lower())
        if doc_encoder.lower() not in stores[0].name.lower():
            print(f"  WARNING: no index matches --doc-encoder {doc_encoder!r}; "
                  f"leading with {stores[0].name}")
        if doc_score in ("combsum", "rankdecay"):
            print(f"  WARNING: --doc-encoder {doc_encoder!r} has NO EFFECT "
                  f"under --doc-score {doc_score}. combsum adds normalised "
                  f"scores from every channel and rankdecay aggregates from "
                  f"the fused fragment ranking directly -- in both cases "
                  f"addition/fusion is commutative, so which encoder leads "
                  f"is irrelevant; stores[0] is only consulted by "
                  f"--doc-score cosine. Results here will be identical to "
                  f"any other --doc-encoder value.")

    sizes = {s.index.ntotal for s in stores}
    if len(sizes) > 1:
        print(f"  WARNING: indexes hold different chunk counts {sorted(sizes)}. "
              "Fusion keys on chunk_id and assumes one shared chunk set. "
              "Rebuild every encoder in a single build_index run.")
    return stores

class LexicalIndex:
    """
    BM25 restricted to the vocabulary of the query set.

    A full inverted index over ~133k chunks is ~20M postings, which in pure
    Python is gigabytes. But the query set is known before retrieval starts:
    50 questions carry a few hundred distinct content words between them.
    Only those terms can ever contribute to a score, so only those get
    postings. The build is one pass over the metadata and a few hundred MB.

    The text indexed is `contexto` + `texto`, matching what build_index.py
    hands the encoder, so the two channels see the same document.
    """

    def __init__(self, metadata, postings, doc_len, k1: float, b: float):
        self.metadata = metadata
        self.postings = postings
        self.doc_len = doc_len
        self.n_docs = len(doc_len)
        self.avgdl = (sum(doc_len) / self.n_docs) if self.n_docs else 1.0
        self.k1, self.b = k1, b
        self.idf = {}
        for term, plist in postings.items():
            df = len(plist) // 2
            self.idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)) if df else 0.0

    @classmethod
    def build(cls, metadata: list[dict], queries: list[str],
              k1: float = 1.2, b: float = 0.75) -> "LexicalIndex":
        vocab: set[str] = set()
        for query in queries:
            vocab.update(tokenize(query))

        postings: dict[str, array] = {term: array("i") for term in vocab}
        doc_len = array("i")

        for meta in metadata:
            text = meta.get("texto", "")
            context = meta.get("contexto", "")
            tokens = tokenize(f"{context} {text}" if context else text)
            doc_len.append(len(tokens))
            if not tokens:
                continue
            for term, tf in Counter(t for t in tokens if t in vocab).items():
                postings[term].extend((len(doc_len) - 1, tf))

        return cls(metadata, postings, doc_len, k1, b)

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl

        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for i in range(0, len(plist), 2):
                index, tf = plist[i], plist[i + 1]
                norm = 1.0 - b + b * (self.doc_len[index] / avgdl)
                scores[index] += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)

        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.metadata[i], float(s)) for i, s in top]

class GraphIndex:


    def __init__(self, entities: dict, edges: list):
        self.entities = entities        
        self.neighbours: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for a, b, weight in edges:
            self.neighbours[a].append((b, weight))
            self.neighbours[b].append((a, weight))
      
        self.by_length = sorted(entities, key=len, reverse=True)

    @classmethod
    def load(cls, path: Path) -> "GraphIndex | None":
        import xml.etree.ElementTree as ET

        if not path.exists():
            return None
        namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
        root = ET.parse(path).getroot()

        names = {k.get("id"): k.get("attr.name")
                 for k in root.findall("g:key", namespace)}

        def data(element) -> dict:
            return {names.get(d.get("key"), d.get("key")): (d.text or "")
                    for d in element.findall("g:data", namespace)}

        graph = root.find("g:graph", namespace)
        if graph is None:
            return None

        entities = {}
        for node in graph.findall("g:node", namespace):
            fields = data(node)
            entities[node.get("id")] = {
                "tipo": fields.get("tipo", "MISC"),
                "chunks": fields.get("chunks", "").split()}

        edges = []
        for edge in graph.findall("g:edge", namespace):
            fields = data(edge)
            try:
                weight = int(fields.get("peso", 1))
            except ValueError:
                weight = 1
            edges.append((edge.get("source"), edge.get("target"), weight))

        return cls(entities, edges)

    def link(self, query: str) -> list[str]:
        """
        Entities named in the query.

        Gazetteer matching against the graph's own vocabulary rather than a
        second NER pass: the graph already contains every entity the NER
        found in the corpus, an entity absent from it has no chunks to
        contribute, and this keeps entrega/ free of a transformers
        dependency. Matching is on the same normalisation used at build time.
        """
        text = f" {strip_accents(query.lower())} "
        text = _WS.sub(" ", _PUNCT.sub(" ", text))
        found, consumed = [], []
        for name in self.by_length:
            padded = f" {name} "
            if padded in text and not any(name in c for c in consumed):
                found.append(name)
                consumed.append(name)
        return found

    def search(self, query: str, k: int, metadata_by_chunk: dict,
               neighbour_weight: float = 0.4) -> list[tuple[dict, float]]:
        seeds = self.link(query)
        if not seeds:
            return []

        scores: dict[str, float] = defaultdict(float)
        for seed in seeds:
            for chunk_id in self.entities.get(seed, {}).get("chunks", ()):
                scores[chunk_id] += 1.0

            for other, weight in self.neighbours.get(seed, ()):
                bonus = neighbour_weight * min(weight, 10) / 10.0
                for chunk_id in self.entities.get(other, {}).get("chunks", ()):
                    scores[chunk_id] += bonus

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(metadata_by_chunk[c], s) for c, s in ranked
                if c in metadata_by_chunk]


def fuse_rrf(rankings: list[list[tuple[dict, float]]],
             weights: list[float] | None = None) -> list[tuple[dict, float]]:
    """
    Weighted Reciprocal Rank Fusion (8.4). Rank-based, so it needs no score
    calibration between a cosine channel and a BM25 channel.
    """
    weights = weights or [1.0] * len(rankings)
    points: dict[str, float] = defaultdict(float)
    registry: dict[str, dict] = {}

    for ranking, weight in zip(rankings, weights):
        if weight <= 0:
            continue
        for rank, (meta, _score) in enumerate(ranking, start=1):
            key = meta["chunk_id"]
            points[key] += weight / (RRF_K + rank)
            registry[key] = meta

    ordered = sorted(points.items(), key=lambda kv: -kv[1])
    return [(registry[key], score) for key, score in ordered]


def fuse_combsum(rankings: list[list[tuple[dict, float]]],
                 weights: list[float] | None = None) -> list[tuple[dict, float]]:
    """
    Min-max normalised CombSUM (8.4, equation 5).

    Used for the DOCUMENT level only. RRF compresses everything into
    [1/(60+k), 1/61], a range too narrow for 8.6's aggregation to
    discriminate: max pooling over near-identical numbers cannot tell a
    strong document from a mediocre one. CombSUM over normalised scores keeps
    the spread that the aggregation step needs.

    A chunk absent from a channel's candidate list scores 0 in that channel,
    which is what 8.4 prescribes.
    """
    weights = weights or [1.0] * len(rankings)
    totals: dict[str, float] = defaultdict(float)
    registry: dict[str, dict] = {}

    for ranking, weight in zip(rankings, weights):
        if weight <= 0 or not ranking:
            continue
        values = [s for _m, s in ranking]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        for meta, score in ranking:
            key = meta["chunk_id"]
            totals[key] += weight * (score - low) / span
            registry[key] = meta

    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    return [(registry[key], score) for key, score in ordered]


# =====================================================================
# post-filters 
# =====================================================================

def expected_phenomenon(query_id: str) -> int:
    """q001-q016 -> 1, q017-q032 -> 2, q033-q050 -> 3."""
    match = re.search(r"(\d+)", query_id or "")
    if not match:
        return 0
    n = int(match.group(1))
    return 1 if n <= 16 else 2 if n <= 32 else 3


def apply_phenomenon_boost(candidates: list[tuple[dict, float]], phenomenon: int,
                           boost: float, mode: str = "multiply"
                           ) -> list[tuple[dict, float]]:
    """
    Reward chunks whose `fenomeno` matches the query-id range, then re-sort.

    A bonus rather than a filter: the id->phenomenon mapping is inferred from
    the question extract, not published, so if it is wrong somewhere this
    costs a few rank positions instead of deleting the answer from the pool.

    MODE MATTERS, and getting it wrong is invisible. In RRF space scores sit
    around 0.016 and multiplying by 1.08 is worth roughly five rank
    positions -- a nudge. In cosine space the top scores sit around 0.88 and
    the same 8% multiplier adds ~0.07, which reorders the whole head of the
    list. Multiplying is therefore not portable between the two.

    mode="add" IS A FRACTION OF THE POOL'S SCORE SPAN, NOT AN ABSOLUTE.
        This was an absolute number once, and it silently did nothing. The
        reasoning behind 0.03 was "cosine runs 0.78-0.88, so this is 30% of
        the range" -- but that is the range of the TOP of the list, not of
        the 1000-deep pool actually being reordered, whose span is nearer
        1.0. Worse, switching --doc-score to combsum with three channels
        stretches the span to ~3.0, so the same constant became 1% of the
        range and the phenomenon filter stopped existing. The visible
        symptom was Fenomeno 3 documents taking every document slot on
        Fenomeno 1 queries.

        Scaling by the observed span makes one number mean the same thing in
        cosine space, in CombSUM space, and in whatever space comes next.
    """
    if not phenomenon or boost <= 0:
        return candidates

    if mode == "add":
        values = [s for _m, s in candidates]
        span = (max(values) - min(values)) if values else 0.0
        bonus = boost * span
        rescored = [(meta, score + (bonus if meta.get("fenomeno") == phenomenon else 0.0))
                    for meta, score in candidates]
    else:
        factor = 1.0 + boost
        rescored = [(meta, score * (factor if meta.get("fenomeno") == phenomenon else 1.0))
                    for meta, score in candidates]
    return sorted(rescored, key=lambda pair: -pair[1])


def deduplicate(candidates: list[tuple[dict, float]], threshold: float,
                n: int = 8, window: int = 200) -> list[tuple[dict, float]]:
    """
    Drop a candidate whose text substantially repeats one already kept.

    Chunks overlap by ~350 characters by design, so when a passage is
    relevant both the chunk containing it and its neighbour score highly. The
    second adds nothing a grader can reward -- same sentences at a different
    offset -- while occupying one of only ten slots. NDCG@10 discounts by
    log2(i+1), so a wasted slot at rank 2 costs far more than one at rank 9,
    and dropping a duplicate promotes everything below it.

    THRESHOLD CALIBRATION -- THERE ARE TWO REGIMES AND ONE NUMBER.
    Containment is measured against the SHORTER of the two shingle sets.
    Measured on this corpus's chunk geometry:

        adjacent windows (1000 chars, 350 overlap)  containment ~0.31
        repeated boilerplate / re-published text     containment ~0.95

    The old default of 0.6 sits between them, so it caught the boilerplate
    and never once caught the overlapping neighbour the docstring claimed it
    was for. That is not necessarily wrong -- an adjacent window still
    carries 65% new text, and the real grader may award it partial
    relevance -- but it means the knob was never doing the job it was
    documented as doing, and nobody had checked.

    0.45 is the default here: still safely above the 0.31 neighbour case,
    with more margin against lightly-edited republications. Going below 0.31
    turns on neighbour suppression, which is a real behaviour change and
    worth an actual sweep: {0.28, 0.35, 0.45, 0.60} against nDCG@10. Watch
    the metric, not the suppression count -- a big count only proves the
    filter is firing, not that it is helping.
    """
    if threshold <= 0:
        return candidates

    head, tail = candidates[:window], candidates[window:]
    kept: list[tuple[dict, float]] = []
    kept_shingles: list[set] = []

    for meta, score in head:
        mine = shingles(meta.get("texto", ""), n)
        if not mine:
            continue
        duplicate = any(
            theirs and len(mine & theirs) / min(len(mine), len(theirs)) >= threshold
            for theirs in kept_shingles)
        if not duplicate:
            kept.append((meta, score))
            kept_shingles.append(mine)

    return kept + tail


def rerank(candidates: list[tuple[dict, float]], query: str, model_name: str,
           depth: int, blend: float = 0.5, use_context: bool = True,
           _cache: dict = {}) -> list[tuple[dict, float]]:
    """
    Optional cross-encoder rerank of the top `depth` fragments.
    """
    if not model_name or not candidates:
        return candidates

    if model_name not in _cache:
        from sentence_transformers import CrossEncoder
        _cache[model_name] = CrossEncoder(model_name, max_length=512)
    model = _cache[model_name]

    head, tail = candidates[:depth], candidates[depth:]
    # The heading trail goes to the cross-encoder too. The dense channel
    # embeds `contexto + texto`, so scoring the reranker on `texto` alone
    # judges the passage stripped of the one signal that says what it is
    # about: "estas afectaciones se han dado cuando..." reads very differently
    # with "4.5 Paz ambiental" in front of it. Truncation is the encoder's
    # job -- max_length=512 cuts the tail, and the context is worth more per
    # token than the last sentence it displaces.
    pairs = [(query, (f"{meta['contexto']}. {meta.get('texto', '')}"
                      if use_context and meta.get("contexto")
                      else meta.get("texto", "")))
             for meta, _ in head]
    scores = model.predict(pairs, show_progress_bar=False)

    if blend >= 1.0:
        reordered = sorted(zip(head, scores), key=lambda pair: -float(pair[1]))
        return [(meta, float(score)) for (meta, _old), score in reordered] + tail

    # RRF between the retrievers' order (0..n-1, already sorted) and the
    # cross-encoder's order, so a unanimous rank-1 needs a strong contrary
    # signal to be displaced rather than a marginally higher logit.
    by_cross = sorted(range(len(head)), key=lambda i: -float(scores[i]))
    points = {i: (1.0 - blend) / (RERANK_RRF_K + rank)
              for rank, i in enumerate(range(len(head)), 1)}
    for rank, i in enumerate(by_cross, 1):
        points[i] += blend / (RERANK_RRF_K + rank)

    order = sorted(points.items(), key=lambda kv: -kv[1])
    return [(head[i][0], score) for i, score in order] + tail



# =====================================================================
# query expansion and diversification
# =====================================================================

def expand_query_rm3(lexical: "LexicalIndex", query: str,
                     feedback_docs: int = 10, terms: int = 10,
                     original_weight: float = 0.6) -> str:

    top = lexical.search(query, feedback_docs)
    if not top:
        return query

    counts: Counter = Counter()
    for meta, _score in top:
        text = meta.get("texto", "")
        context = meta.get("contexto", "")
        counts.update(set(tokenize(f"{context} {text}" if context else text)))

    original = set(tokenize(query))
    # Weight by how concentrated a term is in the feedback set relative to
    # the corpus: idf is what stops "gobierno" and "territorio" winning.
    scored = [(counts[t] * lexical.idf.get(t, 0.0) or counts[t], t)
              for t in counts if t not in original]
    if not scored:
        return query
    for term in [t for _w, t in sorted(scored, reverse=True)[:terms]]:
        lexical.idf.setdefault(term, 0.0)

    picked = [t for _w, t in sorted(scored, reverse=True)[:terms]]
    repeats = max(1, int(round((1 - original_weight) / max(original_weight, 0.01) * 2)))
    return query + " " + " ".join(picked * repeats)


def diversify_mmr(candidates: list[tuple[dict, float]], lam: float = 0.7,
                  n: int = 10, window: int = 60) -> list[tuple[dict, float]]:
    """
    Maximal Marginal Relevance over the top candidates.
    """
    if lam >= 1.0 or not candidates:
        return candidates

    head, tail = candidates[:window], candidates[window:]
    shingle_cache = [shingles(m.get("texto", ""), 8) for m, _s in head]
    scores = [s for _m, s in head]
    high = max(scores) or 1.0
    low = min(scores)
    span = (high - low) or 1.0

    chosen: list[int] = []
    remaining = set(range(len(head)))
    while remaining and len(chosen) < n:
        best_index, best_value = None, -1e9
        for i in sorted(remaining):
            relevance = (scores[i] - low) / span
            mine = shingle_cache[i]
            novelty = 0.0
            for j in chosen:
                theirs = shingle_cache[j]
                if mine and theirs:
                    novelty = max(novelty, len(mine & theirs)
                                  / min(len(mine), len(theirs)))
            value = lam * relevance - (1 - lam) * novelty
            if value > best_value:
                best_index, best_value = i, value
        chosen.append(best_index)
        remaining.discard(best_index)

    ordered = [head[i] for i in chosen]
    ordered += [head[i] for i in range(len(head)) if i not in chosen]
    return ordered + tail



def aggregate_documents(ranking: list[tuple[dict, float]], n: int | None = 3,
                        pool: int = 30, hit_bonus: float = 0.02,
                        hit_cap: int = 3, mode: str = "max",
                        top_m: int = 3) -> list[str]:
    """
    Max pooling plus a bonus for repeated evidence (8.6). Pure arithmetic
    over scores, no generative step.

    MUST be fed a score list with real spread -- cosine or normalised
    CombSUM, never RRF. See the module docstring.
    """
    per_doc: dict[str, list[float]] = defaultdict(list)
    for meta, score in ranking[:pool]:
        per_doc[meta["doc_id"]].append(score)

    if mode == "max":

        base = {d: max(v) for d, v in per_doc.items()}
    elif mode == "sum":

        base = {d: sum(sorted(v, reverse=True)[:top_m]) for d, v in per_doc.items()}
    elif mode == "mean":

        base = {d: sum(sorted(v, reverse=True)[:top_m]) / top_m
                for d, v in per_doc.items()}
    else:                                    # rrf
        position = {}
        for rank, (meta, _s) in enumerate(ranking[:pool], 1):
            position.setdefault(meta["doc_id"], []).append(rank)
        base = {d: sum(1.0 / (RRF_K + r) for r in ranks[:top_m])
                for d, ranks in position.items()}

    hits = {d: len(v) for d, v in per_doc.items()}
    aggregated = {d: base[d] * (1.0 + hit_bonus * min(hits[d] - 1, hit_cap))
                  for d in base}
    ordered = [d for d, _ in sorted(aggregated.items(), key=lambda kv: -kv[1])]

    if n is None:                       # full ranking, for diagnostics
        return ordered


    if len(ordered) < n:
        for meta, _score in ranking:
            if meta["doc_id"] not in ordered:
                ordered.append(meta["doc_id"])
            if len(ordered) >= n:
                break
    return ordered[:n]


def aggregate_documents_rankdecay(ranking: list[tuple[dict, float]],
                                  n: int | None = 3, pool: int = 30,
                                  decay: float = 0.85,
                                  rrf_k: int = RRF_K) -> list[str]:
    
    per_doc: dict[str, list[int]] = defaultdict(list)
    for rank, (meta, _score) in enumerate(ranking[:pool]):
        per_doc[meta["doc_id"]].append(rank)

    scores: dict[str, float] = {}
    for doc_id, ranks in per_doc.items():
        scores[doc_id] = sum(
            (1.0 / (rrf_k + rank + 1)) * (decay ** i)
            for i, rank in enumerate(sorted(ranks)))

    ordered = [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    if n is None:                       # full ranking, for diagnostics
        return ordered

    # 9.3.2: exactly 3 documents or the line is discarded.
    if len(ordered) < n:
        for meta, _score in ranking:
            if meta["doc_id"] not in ordered:
                ordered.append(meta["doc_id"])
            if len(ordered) >= n:
                break
    return ordered[:n]


def build_fragments(candidates: list[tuple[dict, float]], n: int = 10) -> list[dict]:
    """
    Top-n fragments. 9.2.1: a chunk over 250 words is split into complete
    sub-fragments that keep the original chunk_id and each take their own
    rank.
    """
    fragments: list[dict] = []
    for meta, _score in candidates:
        for piece in split_to_250_words(meta["texto"], 250):
            piece = sanitize_text(piece)
            if not piece.strip():
                continue
            fragments.append({"rank": len(fragments) + 1,
                              "chunk_id": meta["chunk_id"],
                              "doc_id": meta["doc_id"],
                              "text": piece})
            if len(fragments) == n:
                return fragments
    return fragments


# =====================================================================
# the single ranking path
# =====================================================================

@dataclass
class Retrieved:
    candidates: list[tuple[dict, float]]   
    unique: list[tuple[dict, float]]   
    doc_ranking: list[tuple[dict, float]]  
    channels: list[tuple[str, list[tuple[dict, float]]]] 

def retrieve(stores: list[VectorStore], query: str, query_id: str = "",
             cfg: RetrievalConfig | None = None,
             lexical: LexicalIndex | None = None,
             graph: "GraphIndex | None" = None) -> Retrieved:
    """
    Query -> Retrieved. THE ONLY ranking path in this project.

    main() calls it and so does evaluar.py. That is not tidiness: the
    post-filters were added to main()'s loop first, and evaluar.py went on
    ranking with a bare `rankings[0]` for a while afterwards. Every number it
    printed described a pipeline that no longer existed. Measuring a
    different system than you ship is worse than not measuring, because it
    looks like data.
    """
    cfg = cfg or RetrievalConfig()

    channels: list[tuple[str, list[tuple[dict, float]]]] = [
        (store.name, store.search(query, cfg.depth)) for store in stores]
    if lexical is not None and cfg.bm25_weight > 0:
        # Expansion applies to the LEXICAL channel only. The dense encoders
        # were trained on natural questions and a query padded with repeated
        # keywords is off-distribution for them; BM25 is a bag of words and
        # has no distribution to leave.
        lexical_query = query
        if cfg.rm3_terms > 0:
            lexical_query = expand_query_rm3(
                lexical, query, cfg.rm3_feedback, cfg.rm3_terms,
                cfg.rm3_original_weight)
        channels.append(("bm25", lexical.search(lexical_query, cfg.depth)))

    if graph is not None and cfg.graph_weight > 0:
        by_chunk = {m["chunk_id"]: m for m in stores[0].metadata}
        hits = graph.search(query, cfg.depth, by_chunk, cfg.graph_neighbour)
        if hits:
            channels.append(("grafo", hits))

    if cfg.min_score > 0:
        # keep the unfiltered ranking if the filter would empty it
        channels = [(name, [(m, s) for m, s in r if s >= cfg.min_score] or r)
                    for name, r in channels]

    weights = [1.0] * len(stores)
    if lexical is not None and cfg.bm25_weight > 0:
        weights.append(cfg.bm25_weight)
    if any(name == "grafo" for name, _r in channels):
        weights.append(cfg.graph_weight)

    rankings = [r for _n, r in channels]
    phenomenon = expected_phenomenon(query_id)

    # ---- fragments: rank space, robust to incomparable score scales (8.4)
    candidates = fuse_rrf(rankings, weights) if len(rankings) > 1 else rankings[0]
    candidates = apply_phenomenon_boost(
        candidates, phenomenon, cfg.phenomenon_boost, mode=cfg.phenomenon_mode)

    if cfg.reranker:
        candidates = rerank(candidates, query, cfg.reranker,
                            cfg.rerank_depth, cfg.rerank_blend,
                            cfg.rerank_context)

    if cfg.doc_score == "rankdecay":
        doc_ranking = candidates
    elif cfg.doc_score == "cosine":
        doc_ranking = apply_phenomenon_boost(
            rankings[0], phenomenon, cfg.phenomenon_boost_doc, mode="add")
    else:
        doc_ranking = apply_phenomenon_boost(
            fuse_combsum(rankings, weights), phenomenon,
            cfg.phenomenon_boost_doc, mode="add")

    unique = deduplicate(candidates, cfg.dedupe_threshold,
                         window=cfg.dedupe_window)
    if cfg.mmr_lambda < 1.0:
        unique = diversify_mmr(unique, cfg.mmr_lambda)
    return Retrieved(candidates, unique, doc_ranking, channels)


# =====================================================================
# queries, validation, entry point
# =====================================================================

def read_queries(path: Path) -> list[tuple[str, str]]:
    """Accepts .jsonl ({query_id, query|text}) or .json (dict or list)."""
    if path.suffix == ".jsonl":
        objects = read_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        objects = ([{"query_id": k, "query": v} for k, v in data.items()]
                   if isinstance(data, dict) else data)

    pairs = []
    for obj in objects:
        query_id = obj.get("query_id") or obj.get("id")
        text = (obj.get("query") or obj.get("text")
                or obj.get("consulta") or obj.get("pregunta"))
        if query_id and text:
            pairs.append((query_id, text))
    return sorted(pairs, key=lambda pair: pair[0])       # q001..q050 (10.3)


def validate(path: Path) -> bool:
    """Strict schema check (9.3.2). Failing here means failing the submission."""

    raw = path.read_text(encoding="utf-8")
    lines = [l for l in raw.split("\n") if l.strip()]
    errors = []

    stray = len(_LINE_SEPARATORS.findall(raw))
    if stray:
        errors.append(f"{stray} raw U+0085/U+2028/U+2029 in the output. A "
                      f"grader using splitlines() will see a malformed file "
                      f"(9.3.2 discards those). Fragment text is not sanitized.")

    if len(lines) != 50:
        errors.append(f"expected 50 lines, found {len(lines)}")

    previous = ""
    for n, line in enumerate(lines, 1):
        obj = json.loads(line)
        query_id = obj.get("query_id", "")
        if not query_id:
            errors.append(f"line {n}: missing query_id")
        if query_id <= previous:
            errors.append(f"line {n}: {query_id} out of order (10.3 wants q001..q050)")
        previous = query_id

        if len(obj.get("documents", [])) != 3:
            errors.append(f"line {n}: documents != 3")
        if len({d["doc_id"] for d in obj.get("documents", [])}) != len(obj.get("documents", [])):
            errors.append(f"line {n}: duplicate doc_id in documents")
        if len(obj.get("fragments", [])) != 10:
            errors.append(f"line {n}: fragments != 10")
        for fragment in obj.get("fragments", []):
            for key in ("rank", "chunk_id", "doc_id", "text"):
                if key not in fragment:
                    errors.append(f"line {n}: fragment missing {key}")
            if len(fragment.get("text", "").split()) > 250:
                errors.append(f"line {n} rank {fragment.get('rank')}: over 250 words")

    print("VALIDATION: OK" if not errors else "VALIDATION:\n  " + "\n  ".join(errors))
    return not errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CODEFEST AD ASTRA 2026 - retrieval and resultados.jsonl")
    add_retrieval_args(parser)
    parser.add_argument("--queries", type=Path, default=Path("consultas_50.jsonl"),
                        help="Ships alongside this script so the jury can "
                             "reproduce resultados.jsonl (1.4).")
    parser.add_argument("--out", type=Path, default=Path("resultados.jsonl"))
    args = parser.parse_args()
    cfg = config_from_args(args)

    index_dir = resolve_input(args.index_dir)
    queries_path = resolve_input(args.queries)
    if args.out == Path("resultados.jsonl") and not Path.cwd().samefile(SCRIPT_DIR):
        args.out = SCRIPT_DIR / args.out      # write next to the index, not the cwd

    stores = load_stores(index_dir, args.doc_encoder, cfg.doc_score)
    queries = read_queries(queries_path)
    print(f"Encoders : {[s.name for s in stores]}")
    print(f"Index    : {index_dir.resolve()}")
    print(f"Queries  : {len(queries)} from {queries_path}")
    print(f"Output   : {args.out.resolve()}")
    print(f"Documents: {cfg.doc_score} space, pool={cfg.doc_pool}, "
          f"hit bonus +{cfg.doc_hit_bonus:.0%} x{cfg.doc_hit_cap}")
    print(f"Fragments: RRF k={RRF_K}, dedupe>={cfg.dedupe_threshold}, "
          f"phenomenon {cfg.phenomenon_mode} {cfg.phenomenon_boost}"
          + (f", reranker={cfg.reranker}" if cfg.reranker else ""))

    lexical = None
    if cfg.bm25_weight > 0:
        print(f"Lexical  : building BM25 over {len(stores[0].metadata)} chunks ...")
        lexical = LexicalIndex.build(stores[0].metadata, [q for _i, q in queries],
                                     cfg.bm25_k1, cfg.bm25_b)
        print(f"           {len(lexical.postings)} query terms, "
              f"avgdl={lexical.avgdl:.0f}")

    graph = None
    if cfg.graph_weight > 0:
        graph = GraphIndex.load(index_dir / "grafo" / "grafo.graphml")
        if graph is not None:
            print(f"Grafo    : {len(graph.entities)} entities, "
                  f"{sum(len(v) for v in graph.neighbours.values()) // 2} "
                  f"relations (bonus, section 8.5)")
        else:
            print(f"Grafo    : none at {index_dir / 'grafo' / 'grafo.graphml'} "
                  f"-- bonus component not built")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dropped_total = 0

    with args.out.open("w", encoding="utf-8") as fh:
        for query_id, text in queries:
            result = retrieve(stores, text, query_id, cfg, lexical, graph)
            dropped_total += len(result.candidates) - len(result.unique)

            fragments = build_fragments(result.unique, 10)
            if len(fragments) < 10:
                # Never ship a short list: 9.3.2 discards the line.
                fragments = build_fragments(result.candidates, 10)

            documents = (
                aggregate_documents_rankdecay(
                    result.doc_ranking, 3, cfg.doc_pool, cfg.doc_decay)
                if cfg.doc_score == "rankdecay" else
                aggregate_documents(
                    result.doc_ranking, 3, cfg.doc_pool,
                    cfg.doc_hit_bonus, cfg.doc_hit_cap,
                    cfg.doc_agg, cfg.doc_top_m))

            fh.write(json.dumps({
                "query_id": query_id,
                "documents": [{"rank": i + 1, "doc_id": d}
                              for i, d in enumerate(documents)],
                "fragments": fragments,
            }, ensure_ascii=False) + "\n")
            print(f"  {query_id} ok")

    print(f"\n  {dropped_total} near-duplicate candidates suppressed in total")
    validate(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()