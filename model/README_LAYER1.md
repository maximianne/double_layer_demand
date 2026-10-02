# Layer 1: Reseller demand

Open `02_layer1_reseller_demand.ipynb` and run the cells in order. Alternatively, from this folder run:

```bash
python layer1.py --data data --output results/layer1
```

The implementation follows the demand-only BLP contraction and concentrated IV/GMM procedure described in the paper and in https://pyblp.readthedocs.io/en/stable/background.html . It uses NumPy/SciPy directly; PyBLP is not required.

## Implemented specification

Mean utility: intercept + quality coefficient × quality + price coefficient × price + three scarcity indicators + unobserved demand shock. Positive price sensitivity alpha is the negative of the reported price coefficient. No scarcity is the omitted category.

The random quality coefficient is beta_quality + pi × log_size + sigma × normal_taste. Estimate sigma and pi. Normal tastes are integrated using nine-node Gauss-Hermite quadrature. Aggregate shares weight resellers by purchasing capacity and include the outside option. This implementation matches the simulator's one-dimensional quality heterogeneity; it does not implement arbitrary covariance matrices, heterogeneous scarcity effects, or arbitrary assortments.

Inside shares are based on requested quantities; products at zero inventory remain orderable through backorders. All product shares must be strictly positive for this baseline. An explicit zero-share estimator is needed before using different simulated or company data with zero shares.

Excluded instruments: acquisition cost, freight, supplier delay, delay squared/cubed, and interactions of cost/delay/delay squared with quality. Intercept and quality are included exogenous regressors. Endogenous price and scarcity indicators are not their own instruments. The cost/delay instruments are valid by the artificial data-generating construction; their validity is not asserted for company data.

Two additional micro moments use observed orders: the covariance-type moment between reseller log size and purchased quality per capacity unit, and the second moment of purchased quality per capacity unit. The latter integrates both normal taste heterogeneity and multinomial sampling variance. Outside purchases contribute zero quality. These moments use observable data only. They help identify taste variation when aggregate IV moments alone are weak. Use `Layer1('data', micro_moments=False)` to run the aggregate-only comparison.

Step 1 uses an inverse instrument second-moment block plus identity weights on the two micro moments. Step 2 uses a covariance of moment sums within category-region-quarter clusters. Micro moment rows are repeated across products so their averaged contribution is the average market-period moment in this balanced data. This is a transparent custom GMM implementation, not PyBLP's micro-moment covariance implementation.

The second-step covariance is a working weighting matrix for the simulation. Persistent reseller tastes and repeated product attributes imply dependence across these market clusters. Standard errors, confidence intervals, and valid inference are intentionally not supplied; use a full-pipeline bootstrap respecting reseller histories/product and market dependence before research inference. The code reports rank, conditioning, convergence, bounds, and simple partial first-stage F statistics. Those F statistics are diagnostics for individual regressors, not a joint weak-identification test for the nonlinear model.

Two starting values are tried. Successful numerical optimization does not certify a global optimum or strong identification. Bounds are sigma in [0, 1.5], pi in [-0.8, 0.8]. Inspect sensitivity to starting values, bounds, instruments, and integration accuracy before empirical use.

## Prediction

`R_hat` equals reseller purchasing capacity times its fitted choice probability, integrating latent normal tastes conditional on log size. Inverted mean utility retains fitted market demand shocks and exactly matches observed aggregate shares. This is an in-sample prediction; an out-of-sample counterfactual needs an explicit treatment of future unobserved demand shocks.

Individual realized latent tastes are not recovered. The simulator holds those tastes fixed for each reseller, so the saved truth's individual demand can differ from the conditional-mean prediction even with correct structural parameters. Consequently, Layer 2's PPML mapping generated from realized individual latent demand need not recover its coefficients exactly when using integrated `R_hat`. That generated-regressor issue requires checking in the next stage.

## Outputs

- coefficients.csv: BLP estimates and homogeneous logit-IV benchmark.
- product_market_results.csv: inverted utility, fitted demand shock, and aggregate shares.
- reseller_demand_predictions.csv: reseller-SKU-period R_hat for Layer 2.
- instrument_diagnostics.csv: partial first-stage diagnostics.
- fit_diagnostics.json: convergence, ranks, conditions, bounds, and share-fit diagnostics.
- parameter_recovery.csv: comparison to true parameters, evaluated only after fitting.
- layer2_input.csv: created by the notebook, joining R_hat to the observed panel.

Ground truth is excluded from both estimation and prediction. Share fit is mechanical after inversion and should not be interpreted as proof of parameter recovery.

## Default-seed verification

The two starts reached the same first-step objective. The second-step optimizer converged without hitting parameter bounds. Maximum inside-share error was about 1.7e-11. Raising quadrature from nine to seventeen nodes at the fitted parameters changed shares by less than 6e-13. The 69,120 reseller predictions aggregated to fitted product quantities and merged one-to-one into the Layer 2 panel.

| Parameter | Estimate | Generating value |
|---|---:|---:|
| Intercept | 1.230 | 1.300 |
| Quality mean | 0.746 | 0.700 |
| Price coefficient | -0.785 | -0.800 |
| Low scarcity | 0.110 | 0.100 |
| Medium scarcity | 0.240 | 0.250 |
| Zero inventory | 0.455 | 0.450 |
| Quality taste standard deviation | 0.294 | 0.300 |
| Quality-size interaction | 0.076 | 0.200 |

The size interaction is not close to its generating value in this sample. Evaluation of the saved latent truth shows a realized sample slope of quality coefficients on size of 0.081. The 120 simulated reseller tastes have an accidental association with size, despite their independent generating draws. This helps explain the fitted interaction; it does not establish accurate population recovery. Larger reseller samples and repeated-seed experiments are needed to assess bias and recovery reliably.

An initial aggregate-only fit estimated sigma at zero. The additional purchase micro moments recovered its variation more closely. This is why the default implementation includes micro moments instead of relying on aggregate share fit alone.
