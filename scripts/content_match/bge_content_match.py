import argparse
import json
import math
import re

import numpy as np
import pandas as pd

SEED = 20261020
NULL_N = 20

VALID = {
    "InfoGive": ["Drugs", "SchedulingAppt", "carecoordination", "Diagnostics", "Generalinformation", "Symptoms", "Instruction", "Prognosis"],
    "InfoSeek": ["Drugs", "Diagnostics", "Generalinformation", "SchedulingAppt", "carecoordination", "Symptoms", "Prognosis"],
    "InfoGiveSDOH": ["HealthCareAccessAndQuality", "EconomicStability", "NeighborhoodAndBuiltEnvironment"],
    "InfoSeekSDOH": ["HealthCareAccessAndQuality", "EconomicStability", "NeighborhoodAndBuiltEnvironment"],
    "PartnershipPatient": ["activeParticipation/involvement", "signoff", "salutation", "Appreciation/Gratitude", "expressOpinions", "Alignment", "connection", "statePreferences", "build trust"],
    "PartnershipProvider": ["salutation", "inviteCollaboration", "signoff", "maintainCommunication", "connection", "Alignment", "checkingUnderstanding/clarification", "Appreciation/Gratitude", "build trust"],
    "Socioemotional/Empathy": ["ExpressConcern/unease", "AcknowledgeSituation", "Reassurance", "PositiveRemarks", "Sadness/fear"],
    "SharedDecisionPatient": ["SeekingApproval", "ExploreOptions", "ApprovalofDecision/Reinforcement"],
    "SharedDecisionProvider": ["ShareOptions", "MakeDecision", "Approval/Reinforcement", "EncourageQuestions", "SummarizeConfirmUnderstanding"],
}
VP = {(c, s) for c, ss in VALID.items() for s in ss}
SDM = {("SharedDecisionProvider", None), ("SharedDecisionPatient", None)}
NEEDS = [
    ("Any information request", {("InfoSeek", None)}, {("InfoGive", None)}, 1129, 986),
    ("Medication question", {("InfoSeek", "Drugs")}, {("InfoGive", "Drugs")}, 367, 219),
    ("Test or result question", {("InfoSeek", "Diagnostics")}, {("InfoGive", "Diagnostics")}, 423, 294),
    ("Scheduling question", {("InfoSeek", "SchedulingAppt")}, {("InfoGive", "SchedulingAppt")}, 217, 124),
    ("Care-coordination question", {("InfoSeek", "carecoordination")}, {("InfoGive", "carecoordination")}, 136, 88),
    ("Symptom question", {("InfoSeek", "Symptoms")}, {("InfoGive", "Symptoms")}, 63, 26),
    ("Concern, sadness or fear",
     {("Socioemotional/Empathy", "ExpressConcern/unease"), ("Socioemotional/Empathy", "Sadness/fear")},
     {("Socioemotional/Empathy", "AcknowledgeSituation"), ("Socioemotional/Empathy", "Reassurance"), ("Socioemotional/Empathy", "PositiveRemarks")},
     337, 112),
    ("Social need", {("InfoSeekSDOH", None), ("InfoGiveSDOH", None)}, {("InfoGiveSDOH", None)}, 765, 250),
    ("Decision request",
     {("SharedDecisionPatient", "SeekingApproval"), ("SharedDecisionPatient", "ExploreOptions")},
     SDM, 331, 100),
]
BANNER = re.compile(r"-{2,}\s*(?:Original\s+)?Message\s*-{2,}", re.I)
TO_RE = re.compile(r"(?i)\bTo:\s*(.*?)\s+Re:\s*")
PH = re.compile(r"^(\W|Person\d+|Org\d+|Dept|MM/DD/YYYY|\s)*$")
SENT = re.compile(r"(?<=[.!?])\s+")


def sq(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def has_label(labels, targets):
    for code, sub in targets:
        if sub is None:
            if any(c == code for c, _ in labels):
                return True
        elif (code, sub) in labels:
            return True
    return False


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def keep_ann(code, sub, span):
    if (code, sub) not in VP:
        return False
    if PH.match(re.sub(r"Dept:?", "", span or "")):
        return False
    return bool(sq(span))


def sentence_for(body, span):
    target = sq(span)
    parts = SENT.split(body.strip()) if body.strip() else []
    for part in parts:
        if target and target in sq(part):
            return re.sub(r"\s+", " ", part).strip()
    return re.sub(r"\s+", " ", span).strip()


def load_pairs(jsonl_path):
    pairs = []
    n_quote = 0
    for line in open(jsonl_path, encoding="utf-8"):
        j = json.loads(line)
        q = j["query"]
        if re.search(r"TO_PAT_YN:\s*(\w)", q).group(1) != "Y":
            continue
        msg = re.search(r"Message:\s*(.*?)\n\nOUTPUT:", q, re.S).group(1)
        hits = list(BANNER.finditer(msg))
        if not hits:
            continue
        n_quote += 1
        authored = msg[: hits[0].start()]
        block = msg[hits[0].end(): hits[1].start() if len(hits) > 1 else len(msg)]
        m = TO_RE.search(block)
        body = block[m.end():] if m else block
        try:
            res = json.loads(j["final_out"]).get("results") or []
        except Exception:
            res = []
        auth_sq = sq(authored)
        body_sq = sq(body)
        cue_spans = []
        reply_spans = []
        for r in res:
            code, sub, span = r.get("Code"), r.get("Sub-code"), r.get("Span") or ""
            if not keep_ann(code, sub, span):
                continue
            s = sq(span)
            if s in body_sq and s not in auth_sq:
                cue_spans.append((code, sub, span))
            if s in auth_sq:
                reply_spans.append((code, sub, span))
        pairs.append(dict(row=j["row"], cue_spans=cue_spans, reply_spans=reply_spans, body=body))
    print("quoted clinical replies", n_quote, "parsed", len(pairs), flush=True)
    return pairs


def build_items(pairs):
    items = {name: [] for name, *_ in NEEDS}
    pools = {name: [] for name, *_ in NEEDS}
    for p in pairs:
        cue_labels = {(c, s) for c, s, _ in p["cue_spans"]}
        reply_labels = {(c, s) for c, s, _ in p["reply_spans"]}
        for name, cues, resp, *_ in NEEDS:
            resp_spans = [span for c, s, span in p["reply_spans"] if has_label({(c, s)}, resp)]
            if resp_spans:
                pools[name].append((p["row"], resp_spans))
            if not has_label(cue_labels, cues):
                continue
            cue_texts = []
            seen = set()
            for c, s, span in p["cue_spans"]:
                if not has_label({(c, s)}, cues):
                    continue
                sent = sentence_for(p["body"], span)
                if sent not in seen:
                    seen.add(sent)
                    cue_texts.append(sent)
            items[name].append(dict(
                row=p["row"],
                cue_texts=cue_texts,
                reply_texts=list(dict.fromkeys(resp_spans)),
                label_match=has_label(reply_labels, resp),
            ))
    for name, _, _, target_n, target_m in NEEDS:
        matched = sum(x["label_match"] for x in items[name])
        print(f"label {name}: cues {len(items[name])} table {target_n}; matches {matched} table {target_m}", flush=True)
    return items, pools


def _load_bge():
    import sys
    import types
    import torch

    guard = "transformers.dependency_versions_check"
    if guard not in sys.modules:
        stub = types.ModuleType(guard)

        def dep_version_check(pkg, hint=None):
            return None

        stub.dep_version_check = dep_version_check
        stub.require_version_core = dep_version_check
        sys.modules[guard] = stub
    from transformers import AutoModel, AutoTokenizer
    import transformers.tokenization_utils_base as tub

    tub.list_repo_templates = lambda *args, **kwargs: []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=dtype)
    model = model.to(device).eval()
    return tok, model, device


def embed_all(texts):
    import torch

    tok, model, device = _load_bge()
    print("device", device, "texts", len(texts), flush=True)
    batch = 16 if device == "cuda" else 8
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        batch_tok = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch_tok = {k: v.to(device) for k, v in batch_tok.items()}
        with torch.no_grad():
            hidden = model(**batch_tok).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
        out[start:start + len(chunk)] = hidden.cpu().numpy()
        if start % (batch * 20) == 0:
            print("embedded", start + len(chunk), "/", len(texts), flush=True)
    return out


def max_sim(cue_ids, reply_ids, emb):
    if not cue_ids or not reply_ids:
        return np.nan
    a = emb[cue_ids]
    b = emb[reply_ids]
    return float(np.max(a @ b.T))


def score(items, pools, emb, index):
    rng = np.random.default_rng(SEED)
    rows = []
    for name, _, _, target_n, target_m in NEEDS:
        group = items[name]
        pool = pools[name]
        pool_rows = np.array([r for r, _ in pool])
        observed = []
        nulls = []
        content = {90: 0, 95: 0, 99: 0}
        obs_by_i = []
        null_by_i = []
        for item in group:
            cue_ids = [index[t] for t in item["cue_texts"]]
            reply_ids = [index[t] for t in item["reply_texts"]]
            sim = max_sim(cue_ids, reply_ids, emb) if item["label_match"] else np.nan
            obs_by_i.append(sim)
            choices = np.flatnonzero(pool_rows != item["row"])
            take = rng.choice(choices, size=min(NULL_N, len(choices)), replace=False)
            these = []
            for j in take:
                ids = [index[t] for t in pool[j][1]]
                these.append(max_sim(cue_ids, ids, emb))
            null_by_i.append(these)
            nulls.extend(these)
            if np.isfinite(sim):
                observed.append(sim)
        nulls = np.asarray(nulls, dtype=float)
        thresholds = {p: float(np.percentile(nulls, p)) if len(nulls) else np.nan for p in (90, 95, 99)}
        for sim in obs_by_i:
            for p, thr in thresholds.items():
                if np.isfinite(sim) and sim > thr:
                    content[p] += 1
        n = len(group)
        k = content[95]
        lo, hi = wilson(k, n)
        y = np.array([1] * len(observed) + [0] * len(nulls))
        scores = np.concatenate([np.asarray(observed, dtype=float), nulls]) if observed else np.array([])
        if len(observed) and len(nulls) and len(np.unique(y)) == 2:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, scores))
        else:
            auc = np.nan
        label_n = sum(x["label_match"] for x in group)
        rows.append(dict(
            need=name,
            pairs_with_cue=n,
            table_pairs=target_n,
            label_match_n=label_n,
            table_label_match_n=target_m,
            label_match_pct=round(100 * label_n / n, 1) if n else np.nan,
            threshold_90=round(thresholds[90], 3),
            threshold_95=round(thresholds[95], 3),
            threshold_99=round(thresholds[99], 3),
            content_n_95=k,
            content_pct_95=round(100 * k / n, 1) if n else np.nan,
            content_lo_95=round(100 * lo, 1) if n else np.nan,
            content_hi_95=round(100 * hi, 1) if n else np.nan,
            content_pct_90=round(100 * content[90] / n, 1) if n else np.nan,
            content_pct_99=round(100 * content[99] / n, 1) if n else np.nan,
            auc=round(auc, 3) if np.isfinite(auc) else np.nan,
            n_observed=len(observed),
            n_null=len(nulls),
        ))
        print(
            f"{name}: content {k}/{n} ({rows[-1]['content_pct_95']}%) "
            f"thr95 {rows[-1]['threshold_95']} auc {rows[-1]['auc']}",
            flush=True,
        )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="BGE content match for quoted reply pairs")
    p.add_argument("--predictions-jsonl", type=str, required=True)
    p.add_argument("--output-csv", type=str, default="content_match_summary.csv")
    args = p.parse_args()
    pairs = load_pairs(args.predictions_jsonl)
    items, pools = build_items(pairs)
    for pair in pairs:
        pair["body"] = ""
    texts = []
    seen = {}

    def add(t):
        if t not in seen:
            seen[t] = len(texts)
            texts.append(t)

    for group in items.values():
        for item in group:
            for t in item["cue_texts"]:
                add(t)
            for t in item["reply_texts"]:
                add(t)
    for group in pools.values():
        for _, spans in group:
            for t in spans:
                add(t)
    print("unique strings", len(texts), flush=True)
    emb = embed_all(texts)
    table = score(items, pools, emb, seen)
    table.to_csv(args.output_csv, index=False)
    print("wrote", args.output_csv, flush=True)
    print("CONTENT_MATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
