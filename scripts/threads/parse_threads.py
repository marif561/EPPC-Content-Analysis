#!/usr/bin/env python3
import argparse
import os
import re

import gender_guesser.detector as gg
import numpy as np
import pandas as pd

HDR = re.compile(
    r"^\s*From:\s*(.*?)\s*Sent:\s*(.*?)\s*To:\s*(.*?)\s*(?:Cc:.*?)?Subject:",
    re.S,
)
CRED = re.compile(
    r",\s*(MD|DO|APRN|PA-?C?|RN|NP|PhD|PharmD|LCSW|MSW|CNS|BSN|MSN|LPN|CMA|RD|DNP|FNP|ANP|OCN|MPH|MS|MA)\b"
    r"|^\s*Nurse\b|\bpool\b|\bPOOL\b|\bteam\b|\bTeam\b|\bclinic\b|\bClinic\b|\bOffice\b|\bnurses?\b"
    r"|\bSmilow\b|\bYNHH\b|\bYale\b|\bCenter\b|\bProgram\b",
    re.I,
)


def clean(n):
    return re.sub(r"\s+", " ", str(n)).strip() if isinstance(n, str) else None


def main():
    ap = argparse.ArgumentParser(description="Parse portal thread export")
    ap.add_argument("--thread-export", required=True)
    ap.add_argument("--output-dir", default="work/intermediate")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.thread_export, encoding_errors="replace", low_memory=False)
    df["pos"] = df.groupby("MESSAGE_id").cumcount()
    df["n_in_thread"] = df.groupby("MESSAGE_id").MESSAGE_id.transform("size")

    rows = []
    for r in df.itertuples():
        s = r.splited_in_order if isinstance(r.splited_in_order, str) else ""
        m = HDR.match(s)
        frm = to = sent = None
        if m:
            frm, sent, to = clean(m.group(1)), clean(m.group(2)), clean(m.group(3))
        body = r.splited_in_order_removed_head if isinstance(r.splited_in_order_removed_head, str) else ""
        rows.append(
            dict(
                thread_id=r.MESSAGE_id,
                pos=r.pos,
                n_in_thread=r.n_in_thread,
                to_pat_thread=r.TO_PAT_YN,
                subject=r.MSG_SUBJECT,
                frm=frm,
                to=to,
                sent=sent,
                body=body.strip(),
                has_header=bool(m),
            )
        )
    M = pd.DataFrame(rows)

    names = pd.concat([M.frm, M.to]).dropna()
    is_clin = {n: bool(CRED.search(n)) for n in names.unique()}

    def side(name):
        if name is None:
            return None
        return "provider" if is_clin.get(name, False) else "patient"

    def direction(r):
        if r.has_header:
            sf, st = side(r.frm), side(r.to)
            if sf == "provider" and st != "provider":
                return "Provider→Patient"
            if sf == "patient" and st == "provider":
                return "Patient→Provider"
            if sf == "provider" and st == "provider":
                return "Provider→Provider"
            return "Unresolved"
        return "Provider→Patient" if r.to_pat_thread == "Y" else "Patient→Provider"

    M["direction"] = M.apply(direction, axis=1)
    M["author_role"] = M.direction.map(
        {
            "Provider→Patient": "Provider",
            "Patient→Provider": "Patient",
            "Provider→Provider": "Provider",
            "Unresolved": "Unresolved",
        }
    )

    det = gg.Detector(case_sensitive=False)

    def first_name(n):
        n = re.sub(r",.*$", "", n)
        n = re.sub(r"^\s*(Nurse|Dr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+", "", n, flags=re.I)
        toks = [t for t in re.split(r"\s+", n) if re.match(r"^[A-Za-z][A-Za-z'-]+$", t)]
        return toks[0] if toks else None

    def sex_of(n):
        if not isinstance(n, str) or re.search(
            r"pool|team|clinic|office|center|program|smilow|ynhh|yale|nurses\b", n, re.I
        ):
            return "U"
        fn = first_name(n)
        if not fn:
            return "U"
        g = det.get_gender(fn)
        return {"male": "M", "mostly_male": "M", "female": "F", "mostly_female": "F"}.get(g, "U")

    def prov_role(n):
        if not isinstance(n, str):
            return "Unknown"
        if re.search(r"\b(MD|DO)\b", n):
            return "Physician"
        if re.search(r"\b(APRN|NP|FNP|ANP|DNP|PA-?C?)\b", n):
            return "APP (APRN/PA)"
        if re.search(r"^\s*Nurse\b|\bRN\b|\bBSN\b|\bMSN\b|\bLPN\b|\bOCN\b", n, re.I):
            return "Nurse"
        if re.search(r"pool|team|clinic|office|center|program|nurses\b", n, re.I):
            return "Pool/team"
        return "Other clinical staff"

    th = []
    for tid, g in M.groupby("thread_id"):
        pats, provs, firstprov = [], [], None
        for r in g.itertuples():
            for nm in (r.frm, r.to):
                if nm is None:
                    continue
                (provs if is_clin.get(nm) else pats).append(nm)
            if firstprov is None and r.direction == "Patient→Provider" and r.to and is_clin.get(r.to):
                firstprov = r.to
        patient = pd.Series(pats).mode().iloc[0] if pats else None
        if provs:
            vc = pd.Series(provs).value_counts()
            top = vc[vc == vc.max()].index.tolist()
            provider = top[0] if len(top) == 1 else (firstprov if firstprov in top else top[0])
        else:
            provider = None
        th.append(
            dict(
                thread_id=tid,
                patient_name=patient,
                provider_name=provider,
                n_clinicians=len(set(provs)),
                addressed_provider=firstprov,
            )
        )
    T = pd.DataFrame(th)
    T["patient_sex"] = T.patient_name.map(sex_of)
    T["provider_sex"] = T.provider_name.map(sex_of)
    T["provider_role"] = T.provider_name.map(prov_role)
    pmap = {n: f"PT{i+1:04d}" for i, n in enumerate(sorted(T.patient_name.dropna().unique()))}
    dmap = {n: f"PR{i+1:04d}" for i, n in enumerate(sorted(T.provider_name.dropna().unique()))}
    T["patient_id"] = T.patient_name.map(pmap).fillna("PT_unknown")
    T["provider_id"] = T.provider_name.map(dmap).fillna("PR_unknown")
    T["dyad"] = T.patient_sex + T.provider_sex
    T["dyad4"] = T.dyad.where(T.dyad.isin(["MM", "MF", "FM", "FF"]), "Unknown")

    def parse_ts(s):
        if not isinstance(s, str):
            return pd.NaT
        s = re.sub(r"\s+(EDT|EST)$", "", s).replace(",", "")
        return pd.to_datetime(s, errors="coerce", format="mixed")

    M["ts"] = M.sent.map(parse_ts)
    M = M.merge(
        T[["thread_id", "patient_id", "provider_id", "patient_sex", "provider_sex", "dyad4", "provider_role"]],
        on="thread_id",
    )
    M["author_id"] = np.where(M.author_role == "Patient", M.patient_id, M.provider_id)
    M["n_words"] = M.body.str.split().str.len()
    M.drop(columns=["frm", "to"]).to_pickle(os.path.join(args.output_dir, "messages.pkl"))
    T.drop(columns=["patient_name", "provider_name", "addressed_provider"]).to_pickle(
        os.path.join(args.output_dir, "threads.pkl")
    )

    print("threads", T.thread_id.nunique())
    print("messages", len(M))
    print("dyad4", T.dyad4.value_counts().to_dict())
    print("ts parsed", int(M.ts.notna().sum()), "of", int(M.has_header.sum()))


if __name__ == "__main__":
    main()
