#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser(description="Adjusted thread length and reply-time models")
    ap.add_argument("--threads", required=True)
    args = ap.parse_args()

    T = pd.read_csv(args.threads)
    dy = T[T.dyad4.isin(["MM", "MF", "FM", "FF"])].copy()
    K = dy.copy()
    K["ps"] = K.patient_sex
    K["vs"] = K.provider_sex
    K["role"] = K.provider_role.replace({"Other clinical staff": "Nurse", "Pool/team": "Nurse"})
    K["role"] = K.role.replace({"Nurse": "Nurse/APP", "APP (APRN/PA)": "Nurse/APP"})
    K = K[K.role.isin(["Physician", "Nurse/APP"]) & K.ps.isin(["M", "F"]) & K.vs.isin(["M", "F"])]
    print("model_n", len(K))

    for c in ["n_msgs", "n_ann"]:
        d = K.dropna(subset=[c])
        m = smf.glm(
            f'{c} ~ C(ps, Treatment("M")) + C(vs, Treatment("M")) + C(role, Treatment("Physician"))',
            d,
            family=sm.families.Poisson(),
        ).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.patient_id)[0]})
        print(c, "n", len(d))
        for k in m.params.index:
            if k == "Intercept":
                continue
            lo, hi = np.exp(m.conf_int().loc[k])
            print(f"  {np.exp(m.params[k]):.2f} ({lo:.2f}-{hi:.2f}) {k}")

    d = K.dropna(subset=["latency_h"]).copy()
    d["ll"] = np.log1p(d.latency_h)
    m = smf.ols(
        'll ~ C(ps, Treatment("M")) + C(vs, Treatment("M")) + C(role, Treatment("Physician"))',
        d,
    ).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.patient_id)[0]})
    print("latency_n", len(d))
    for k in m.params.index:
        if k == "Intercept":
            continue
        lo, hi = np.exp(m.conf_int().loc[k])
        print(f"  {np.exp(m.params[k]):.2f} ({lo:.2f}-{hi:.2f}) {k}")


if __name__ == "__main__":
    main()
