# Statistical analysis checklist

1. **Look at the data first.** Parse dates, sort, and infer the real sampling frequency from the gaps
   between dates (annualise with 252 for daily, 52 for weekly, 12 for monthly data), whatever the request
   says. Check for missing values, late starts, duplicated or implausible values, sudden jumps (e.g. a stock
   split that halves the price and doubles the volume) and units.
2. **Do not trust a user's description over the data.** If they conflict, say so and use what the data show.
3. **Missing prices.** Do not forward-fill or interpolate prices to estimate risk (it understates
   volatility and correlation), and do not drop incomplete rows and then compute returns (multi-day returns
   inflate volatility). Compute returns between consecutive observed prices and use pairwise-complete
   covariances; a pairwise covariance matrix may not be positive semi-definite - repair it (clip negative
   eigenvalues / nearest correlation matrix) before optimising.
4. **Non-stationary series.** Regressions or correlations between price levels (random walks) are spurious.
   Use returns / differences, or test for cointegration.
5. **Dependent observations.** Autocorrelated time series and clustered data (students in classes, repeated
   measurements) violate the independence assumption of ordinary t-tests and OLS standard errors. Model the
   dependence (AR errors, HAC standard errors, cluster-level analysis, mixed models).
6. **Multiple testing.** When you search over many candidates, correct for it (Bonferroni / Holm / FDR) or
   confirm on held-out data.
7. **Confounding.** Compare groups within levels of pre-treatment variables that drive both treatment and
   outcome (stratify / adjust); an aggregate comparison can reverse the true effect.
8. **Leakage.** For a predictive model, use only information available at prediction time; exclude columns
   recorded after the outcome.
9. **Report honestly.** State the data issues you found and the choices you made, and give a confidence that
   reflects them.
