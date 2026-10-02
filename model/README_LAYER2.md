# Layer 2: End-user sell-through

Open `03_layer2_enduser_demand.ipynb` after completing Layer 1. The package includes already-fitted results. To run from the terminal:

```bash
python layer2.py --data data --layer1 results/layer1 --output results/layer2
```

Add `--no-evaluate` to exclude all optional ground-truth evaluation. Dependencies remain NumPy, pandas, and SciPy; neither statsmodels nor PyBLP is required. The notebook uses the same code, with truth evaluation disabled by default.

## Reporting probit

Estimate V = 1 if sell-through is observed, using the full panel. The index includes intercept, log reseller size, reporting integration score, reseller type, and quarter index. Probabilities are Phi(index); observed rows receive raw inverse-probability weights 1/pi. The likelihood uses stable log-normal-CDF computations and analytic gradients.

This baseline deliberately matches the simulator's reporting equation. It does not add reseller reporting fixed effects: time-invariant size, type, and integration score would be collinear with them. It does not fit product reporting effects, which the generator does not contain. These are specification choices for the artificial experiment, rather than a verbatim implementation of every optional fixed effect in the paper. Reporting heterogeneity and the appropriate fixed effects must be revisited for company data.

No probabilities are silently clipped and no weights are silently trimmed. The outputs report overlap, weight quantiles, maximum weight, and effective reporting-sample size. Very small probabilities indicate an overlap concern. If you choose trimming or stabilization later, implement and report it explicitly.

## IPW-PPML

Conditional mean = exp(intercept + b log(1+R_hat) + promotion coefficient × promotion + replenishment coefficient × days + reseller effects + product effects + market effects + ordering-period effects).

Fit only observed sell-through, preserving observed zero sales. Maximize the weighted criterion sum (1/pi) × [Y log(mu) - mu]. This is a pseudo-likelihood criterion: the estimator does not require the outcome to have a Poisson distribution when its conditional mean is correctly specified. An unweighted PPML benchmark is also saved. Reporting integration score is excluded from the outcome equation by the artificial DGP.

The code uses sparse dummy matrices and Newton updates with analytic score/Hessian and a backtracking line search. Dummy normalization omits one reseller per region and one product per category, plus a reference market and ordering period, because market effects already contain region/category. The design is checked for full rank. The normalized intercept and individual fixed effects should not be compared directly with raw generating values; compare invariant fitted means and core slopes.

The fixed-effect design assumes the balanced simulation's nested region/category structure. This is not a generic arbitrary-panel high-dimensional-fixed-effect library. Prediction requires all fixed-effect levels to have reporting support. The code fails on rank deficiency and obvious all-zero dummy outcome groups instead of silently assigning effects. Other separation patterns may still require a specialized PPML separation diagnostic.

All market/product/reseller effects are estimated from observed sales. No true fixed effects or true demand enter the estimator. No penalties or ridge regularization are applied. Large Hessian condition numbers are reported and should be investigated for empirical data. Conditional standard errors are intentionally not supplied; inference must account for estimated reporting weights, Layer 1 parameters, and persistent reseller/product dependence through a suitable full-pipeline bootstrap.

## Predictions and completion

Every panel row receives a predicted end-user conditional mean. Completed demand uses actual observed sell-through for reporters, and the prediction for nonreporters. The code also saves model-predicted demand for all rows as a distinct column; completed totals and all-model totals answer different questions.

Aggregate completed demand by SKU/category/region/quarter/ordering period and by SKU across all periods. The three monthly periods are disjoint, so totals do not double-count within-quarter scarcity states.

These are reconstructed sell-through quantities/means under the assumed mapping. The simulator has no downstream stockout censoring or dynamic inventory accounting. The outputs therefore do not separately identify unmet end-user demand or recover downstream inventory.

## Optional evaluation

Only after ordinary estimation, the evaluation branch opens ground_truth.csv. It reports prediction error, missing-sample and overall total error, and reporting probability accuracy. It also fits an oracle IPW-PPML using true reseller demand, strictly for evaluation. The oracle isolates Layer 2 coefficient recovery from Layer 1's generated-regressor and integrated-latent-taste issues. It must never be used as an empirical estimator.

The current simulator selects reporting on observables (MAR). Consequently, baseline IPW-PPML is the implemented selection correction. An inverse-Mills/control-function extension is not included: validating it requires a separately specified selection-on-unobservables DGP and a valid nonlinear selection correction, not merely adding a Mills ratio to this simulation.

## Files in results/layer2

| File | Contents |
|---|---|
| reporting_coefficients.csv | Estimated probit coefficients |
| ppml_coefficients.csv | IPW and unweighted slopes/fixed effects |
| enduser_predictions.csv | Full panel, probabilities, weights, predictions, completed quantities |
| product_region_quarter_period_demand.csv | Detailed demand aggregates |
| product_total_demand.csv | SKU totals across regions and time |
| fit_diagnostics.json | Convergence, rank, overlap, and weight diagnostics |
| core_parameter_recovery.csv | Optional estimated-versus-true and oracle slopes |
| evaluation_metrics.json | Optional truth-based prediction and aggregate checks |

Optional evaluation files in the distributed package refer to its included default-seed run. Running with evaluation disabled does not regenerate those files. Delete old evaluation files before using a different dataset, or rerun with evaluation enabled, to avoid comparing stale outputs.

## Default-seed results

Both PPML fits converged in seven iterations. The reporting fit converged, all 177 outcome design columns were linearly independent, and the exported panel contains all 69,120 rows. Reporting was observed for 46,205 rows, including 316 zero sales. The smallest fitted reporting probability was 0.0399, the largest weight was 25.09, and the IPW effective reporting-sample size was about 33,003. No trimming was applied.

| Core coefficient | IPW with R_hat | Evaluation oracle | Generating value |
|---|---:|---:|---:|
| log(1+reseller demand) | 0.8388 | 0.8459 | 0.8500 |
| Promotion | 0.1175 | 0.1212 | 0.1200 |
| Replenishment days | -0.00605 | -0.00601 | -0.00600 |

Completed total demand was 962,801.46 units, versus 961,822 realized simulated units (+0.102%). Predicted quantities for the nonreporting sample totaled 307,630.46, versus 306,651 hidden realized units (+0.319%). These are in-sample, default-seed checks, not out-of-sample accuracy guarantees. Individual mean-prediction RMSE was about 2.71 units. The oracle comparison helps distinguish generated-regressor error from downstream estimation error.

Validation checked the weighted-mean PPML optimum, invariance to multiplying all weights by a constant, rejection of all-zero outcomes, correct preservation of observed/missing sell-through, and agreement between row-level and SKU totals.
