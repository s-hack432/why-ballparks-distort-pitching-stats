"""Step 4 (optional check): logistic versions of the binary-outcome models.

Refits the strikeout, walk + HBP, ball-in-play and home-run models as penalized logistic
regressions with the same design as Step 4, then puts the park effects back on the
probability scale and compares them with the linear-probability estimates.

* Penalty translation. The linear model's ridge penalty lambda corresponds to a normal prior
  with variance p(1-p)/lambda on the probability scale, i.e. a prior precision of
  lambda * p(1-p) on the logit scale. scikit-learn's L2 logistic regression has one penalty
  for all columns, so columns are rescaled to give each block its own effective precision:
  parks, season, platoon and home are scaled up by 100 (effectively unpenalized); batter columns
  by sqrt(lambda_p / lambda_b).
* Probability scale. A park's effect = mean over plate appearances of
  [P(y | park p) - P(y | average park)], approximated at the league rate:
  p(1-p) x (logit coefficient - its PA-weighted mean).

Output: outputs/tables/logistic_vs_linear.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from . import ridge
from .common import get_logger, log_df
from .config import DATA_PROCESSED, OUT_TABLES
from .park_model import FITS, OUTCOMES, add_literal_columns, samples

log = get_logger("logistic_check", "step4b_logistic_check.log")


def main():
    pa = add_literal_columns(pd.read_parquet(DATA_PROCESSED / "pa.parquet"))
    eff = np.load(DATA_PROCESSED / "park_effects_pooled.npz")
    outcomes = list(eff["outcomes"])
    pens = pd.read_csv(OUT_TABLES / "model_penalties.csv")
    pens = pens[pens["fit"] == "pooled"].set_index("outcome")
    data = samples(pa, FITS["pooled"])
    rows = []
    for key in ("k", "bbhbp", "bip", "hr"):
        sample = OUTCOMES[key][0]
        d = data[sample]
        des = ridge.build_design(d, ridge.standard_blocks(True, "park_key"))
        y = d[OUTCOMES[key][1]].to_numpy().astype(int)
        p = y.mean()
        lp, lb = pens.loc[key, "lambda_pitcher"], pens.loc[key, "lambda_batter"]
        prec_p = lp * p * (1 - p)                       # logit-scale prior precision, pitchers
        scale = np.ones(des.n_cols)
        for b in ("park", "season", "same_hand", "home"):
            if b in des.slices:
                scale[des.slices[b]] = 100.0
        scale[des.slices["batter"]] = np.sqrt(lp / lb)
        Xs = des.X @ sparse.diags(scale)
        # Newton's method with an exact Hessian: ~2,300 columns, converges in a few iterations.
        clf = LogisticRegression(C=1.0 / prec_p, fit_intercept=False, solver="newton-cholesky",
                                 max_iter=100, tol=1e-8)
        clf.fit(Xs, y)
        gamma = clf.coef_.ravel() * scale              # back to unscaled logit coefficients
        park = gamma[des.slices["park"]]
        counts = ridge.block_counts(des, "park")
        logit_eff = p * (1 - p) * ridge.center(park, counts)
        lin = eff["point"][:, outcomes.index(key)]
        order = [list(eff["parks"]).index(k) for k in des.levels["park"]]
        lin = lin[order]
        mod = np.asarray(des.levels["park"]) != "other"
        rows.append({"outcome": key, "league_rate": p, "corr": np.corrcoef(lin[mod], logit_eff[mod])[0, 1],
                     "max_abs_diff": np.abs(lin[mod] - logit_eff[mod]).max(),
                     "sd_linear": lin[mod].std(), "sd_logistic": logit_eff[mod].std(),
                     "top_park_linear": des.levels["park"][mod][np.argmax(lin[mod])],
                     "top_park_logistic": des.levels["park"][mod][np.argmax(logit_eff[mod])],
                     "converged_iter": int(clf.n_iter_[0])})
        log.info("%s: logistic vs linear park effects r = %.4f, max |diff| = %.4f", key,
                 rows[-1]["corr"], rows[-1]["max_abs_diff"])
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "logistic_vs_linear.csv", index=False)
    log_df(log, out.round(4), "Logistic vs linear-probability park effects (probability scale, pooled fit):",
           index=False)


if __name__ == "__main__":
    main()
