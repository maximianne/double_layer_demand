# Inferring End-User Demand in a Multi-Tier Supply Chain

This project implements a two-layer demand framework for inferring end-user purchases when an intermediary observes reseller orders broadly but observes downstream sell-through only partially.

The current implementation uses simulated data. It provides a reproducible environment for developing the estimator, checking its numerical behavior, and comparing estimates with known simulation truth before applying the model to company data.

## 1. Economic objective

The supply chain studied here is:

**Intermediary firm → Reseller → End user**

The intermediary observes what resellers request or purchase. Those quantities carry information about final demand, but also reflect inventory, replenishment, expectations, prices, and supply constraints. Consequently, reseller orders cannot simply be interpreted as end-user purchases.

The model separates the problem into two layers:

1. **Layer 1: Reseller demand.** Estimate reseller purchasing behavior using a random-coefficients differentiated-products demand model and construct predicted reseller demand.
2. **Layer 2: End-user sell-through.** Estimate how predicted reseller demand maps into observed end-user purchases, correcting for incomplete reporting, and predict sell-through where it is missing.

The implemented sequence is:

**Requested reseller orders → BLP estimation → predicted reseller demand → reporting probit → IPW-PPML → completed end-user quantities → aggregation**

## 2. Notation and markets

| Symbol | Meaning |
|---|---|
| `i` | Reseller |
| `j` | Product or SKU |
| `m` | Category × region × year-quarter market |
| `ell` | Ordering period within a quarter |
| `Q` | Observed requested reseller quantity |
| `M_i` | Reseller purchasing capacity within a market-period |
| `R_hat` | Predicted reseller demand from Layer 1 |
| `Y` | Observed end-user sell-through quantity |
| `V` | Indicator equal to one when sell-through is observed |
| `pi_hat` | Estimated probability of reporting sell-through |
| `mu_hat` | Predicted end-user conditional mean |

Each quarter contains three disjoint monthly ordering periods. `market_id` identifies the broader category-region-quarter market; `market_period_id` also identifies the ordering period.

The ordering-period index is a time partition, not a scarcity label shared by every product. Different products can have different scarcity states during the same ordering period.

## 3. Simulated data

The default simulation uses seed **20261002** and contains:

| Dimension | Count |
|---|---:|
| Resellers | 120 |
| Products | 24 |
| Product categories | 2 |
| Regions | 2 |
| Quarters | 8 |
| Ordering periods per quarter | 3 |
| Reseller-product-period observations | 69,120 |
| Product-market-period observations | 1,152 |
| Observations reporting sell-through | 46,205 |
| Observations missing sell-through | 22,915 |
| Observed zero sell-through outcomes | 316 |

Each reseller belongs to one region and purchases in both categories. The quarters begin at 2024Q1. Prices and quantities are artificial and should not be interpreted as actual company outcomes.

### Data files

| File in `data/` | Contents and purpose |
|---|---|
| `products.csv` | SKU, category, brand, quality, base cost, and pack size |
| `resellers.csv` | Reseller ID, region, size, type, and reporting integration score |
| `supply_inventory.csv` | Prices, scarcity, availability, fill rates, inventory proxy, and instruments |
| `reseller_capacity.csv` | Purchasing capacity and realized outside-option quantity |
| `reseller_panel.csv` | Complete requested/fulfilled-order panel and observed or missing sell-through |
| `sellthrough.csv` | Reporting observations, including observed zeros |
| `blp_product_market.csv` | Aggregate quantities, characteristics, instruments, and inside/outside shares |
| `ground_truth.csv` | Hidden demand, outcomes, tastes, demand shocks, and reporting probabilities |
| `true_parameters.json` | Coefficients used to generate the data |
| `validation_summary.json` | Simulation counts and checks |

**Ground truth is excluded from ordinary estimation and prediction.** It is opened only for optional evaluation after the model has been fitted.

## 4. How reseller orders are generated

Each reseller receives a purchasing capacity. This represents the number of purchasing units that can be allocated among the products and an outside option.

The outside option can be interpreted as purchasing elsewhere, postponing an order, using existing inventory, or otherwise not allocating capacity to the modeled products.

Requested quantities are multinomial draws from random-coefficients logit probabilities. Utility depends on product quality, price, scarcity, reseller size, an unobserved reseller taste, and an unobserved product-market-period demand shock.

The quality coefficient is:

```text
quality_coefficient_i = beta_quality + pi × log_size_i + sigma × normal_taste_i
```

The latent taste draw is held fixed for each reseller throughout the simulation. Price and scarcity are endogenous because they also depend on the demand shock. Cost, freight, and supplier-delay variation provide excluded instruments independent of that shock by construction.

### Requested versus fulfilled orders

The simulation separates requested orders, fulfilled orders, and backorders:

```text
requested_quantity = fulfilled_quantity + backorder_quantity
```

Products remain orderable at zero upstream inventory because backorders are accepted. Layer 1 therefore uses **requested orders**. Fulfilled quantities alone would mix demand with supply constraints.

The inventory variable is a simplified fulfillment proxy. It is not a dynamic inventory ledger, and the simulation does not impose downstream stock-flow accounting.

## 5. Layer 1: Estimating reseller demand

### Step 1: Construct market shares

Aggregate requested quantities across resellers, then divide by total purchasing capacity:

```text
market_size = sum_i purchasing_capacity_i
observed_product_share_j = requested_quantity_j / market_size
outside_share = 1 - sum_j observed_product_share_j
```

Dividing by total observed inside purchases would remove the outside option, so the denominator includes capacity allocated outside the modeled products.

### Step 2: Specify mean utility and heterogeneity

Mean utility includes an intercept, quality, price, three scarcity indicators, and an unobserved demand shock. No scarcity is the reference category. The reported price coefficient is negative; positive price sensitivity `alpha` is its negative.

The nonlinear parameters are the standard deviation of quality tastes and the quality-size interaction. This baseline implements one-dimensional quality heterogeneity rather than an unrestricted random-coefficient covariance matrix.

### Step 3: Invert shares with the BLP contraction

For candidate nonlinear parameters, iterate:

```text
delta_new = delta_old + log(observed_share) - log(predicted_share)
```

Continue until mean utilities converge. Start from the homogeneous-logit utility:

```text
delta_start = log(observed_share) - log(outside_share)
```

Normal tastes are integrated with nine-node Gauss-Hermite quadrature. Aggregate predicted shares weight reseller probabilities by purchasing capacity.

### Step 4: Estimate linear coefficients by IV

At each candidate nonlinear parameter vector, concentrate out the linear utility coefficients using instrumental variables. Price and scarcity indicators are treated as endogenous.

The instrument set includes intercept and quality, plus acquisition cost, freight, supplier delay, delay squared/cubed, and selected interactions with quality. Endogenous price and scarcity indicators are not used as their own instruments.

Instrument validity follows from the artificial data construction. It must be justified separately for company data.

### Step 5: Estimate heterogeneity with two-step GMM

Choose nonlinear parameters to minimize the GMM objective. The second step updates the working weighting matrix using moment sums within category-region-quarter clusters. Two starting values are tried.

Aggregate IV moments alone produced an estimate of zero taste variation in the initial fit. The default estimator therefore adds two micro moments from observed reseller purchases:

1. A size-quality moment relating reseller log size to purchased quality per capacity unit.
2. A second-moment condition measuring variation in purchased quality per capacity unit.

The second moment integrates both latent taste variation and multinomial sampling variance. These moments use observed purchases, not hidden simulation tastes.

### Step 6: Predict reseller demand

```text
R_hat_ij = purchasing_capacity_i × predicted_choice_probability_ij
```

These are in-sample conditional-mean predictions. They integrate over unobserved normal tastes conditional on observed size; they do not recover each reseller's realized latent taste. Inverted mean utility retains the fitted market demand shock.

## 6. How end-user sales and reporting are generated

End-user sales are Poisson draws from a log-link mean using **true expected reseller demand**, promotions, replenishment days, and reseller/product/time/ordering-period effects.

Reporting is generated separately. Its probit index depends on observed size, reseller type, integration status, and quarter index. About 67% of rows report sell-through.

The baseline assumes selection on observables: conditional on those reporting characteristics, reporting is independent of the end-user outcome.

Missing outcomes remain missing:

- `V = 1, Y = 0`: sell-through was reported and sales were zero.
- `V = 0, Y = NaN`: sell-through was not reported.

Missing sell-through is never replaced with zero during estimation.

## 7. Layer 2: Estimating end-user sell-through

### Step 1: Fit reporting probit

Estimate reporting probabilities on the complete panel:

```text
pi_hat = Phi(intercept + size + integration + reseller_type + quarter_index)
```

The implemented reporting regressors match the simulator. This baseline does not include reseller reporting fixed effects, because the time-invariant reporting regressors would be collinear with them. Product reporting effects are also omitted because they are absent from the generating process.

### Step 2: Construct inverse-probability weights

Reporting rows receive:

```text
ipw = 1 / pi_hat
```

Less-likely reporting observations receive greater weight. No automatic clipping or trimming is applied. The outputs report probability ranges, weight diagnostics, and effective sample size.

### Step 3: Fit IPW-PPML

Estimate the conditional mean on observed sell-through:

```text
mu_hat = exp(
    intercept
    + b × log(1 + R_hat)
    + promotion_effect × promotion
    + replenishment_effect × replenishment_days
    + reseller_effect + product_effect + market_effect + ordering_period_effect
)
```

Maximize the weighted criterion:

```text
sum_reporting_rows ipw × [Y × log(mu) - mu]
```

Observed zeros contribute to estimation. The PPML criterion targets the conditional mean; it does not require a Poisson outcome distribution when that mean is correctly specified. An unweighted benchmark is also fitted.

Fixed effects are normalized to remove linear dependencies from nested region/category membership. The normalized intercept and fixed effects cannot be compared directly with raw generating values; compare core slopes and fitted means instead.

### Step 4: Predict and complete end-user quantities

Every row receives a predicted end-user conditional mean. Completed quantities retain actual sales for reporters and impute predictions for nonreporters:

```text
completed_quantity = observed_sellthrough   if V = 1
completed_quantity = predicted_mean        if V = 0
```

The all-model prediction is saved separately from the completed quantity.

### Step 5: Aggregate

Sum completed quantities by product, category, region, quarter, and ordering period, and by product across the entire sample. The ordering periods are disjoint, so these totals do not count the same time interval repeatedly.

## 8. How to run the project

Place this README in the extracted `demand_simulation/` folder alongside the scripts and notebooks. Use Python 3.10 or newer.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Notebook workflow

Open these notebooks in VS Code and run them in order:

1. `01_inspect_simulated_data.ipynb`: inspect or regenerate the simulation.
2. `02_layer1_reseller_demand.ipynb`: fit reseller demand and save `R_hat`.
3. `03_layer2_enduser_demand.ipynb`: fit reporting and sell-through, then aggregate demand.

Select the Python environment containing the installed dependencies. Run the notebooks from the project folder so their relative paths resolve correctly.

### Terminal workflow

```bash
python simulate_data.py --output data
python layer1.py --data data --output results/layer1
python layer2.py --data data --layer1 results/layer1 --output results/layer2
```

To exclude optional truth-based Layer 2 evaluation:

```bash
python layer2.py --data data --layer1 results/layer1 --output results/layer2 --no-evaluate
```

The distributed package already contains completed default-seed results. Layer 1 can take several minutes to refit. The Layer 2 notebook disables truth evaluation by default.

## 9. Main outputs

| Location | Output | Purpose |
|---|---|---|
| `results/layer1/` | `coefficients.csv` | BLP coefficients and homogeneous-logit IV benchmark |
| `results/layer1/` | `product_market_results.csv` | Inverted utilities, demand shocks, and predicted shares |
| `results/layer1/` | `reseller_demand_predictions.csv` | Reseller-level `R_hat` |
| `results/layer1/` | `layer2_input.csv` | Observed panel joined to Layer 1 predictions |
| `results/layer1/` | `parameter_recovery.csv` | Optional comparison against generating coefficients |
| `results/layer2/` | `reporting_coefficients.csv` | Probit estimates |
| `results/layer2/` | `ppml_coefficients.csv` | IPW and unweighted outcome estimates |
| `results/layer2/` | `enduser_predictions.csv` | Full panel with reporting probabilities, weights, predicted means, and completed quantities |
| `results/layer2/` | `product_region_quarter_period_demand.csv` | Detailed demand aggregates |
| `results/layer2/` | `product_total_demand.csv` | Product totals across the sample |
| `results/layer2/` | `core_parameter_recovery.csv` | Core slope recovery and evaluation-only oracle comparison |
| `results/layer2/` | `evaluation_metrics.json` | Truth-based prediction and total-error measures |

Both result folders also contain convergence and validation diagnostics. Evaluation files bundled with the package refer to its default-seed run; do not interpret old evaluation files as results for newly generated data.

## 10. Default-seed results

### Layer 1 coefficient recovery

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

The size interaction recovered less accurately. In this sample of 120 resellers, latent taste draws have an accidental association with size: the realized sample slope of quality coefficients on size is approximately 0.081. That helps explain the fitted interaction, but does not establish accurate population recovery. Larger samples and repeated seeds are needed to assess it reliably.

### Layer 2 coefficient recovery

| Parameter | IPW estimate using `R_hat` | Evaluation oracle | Generating value |
|---|---:|---:|---:|
| `log(1 + reseller demand)` | 0.8388 | 0.8459 | 0.8500 |
| Promotion | 0.1175 | 0.1212 | 0.1200 |
| Replenishment days | -0.00605 | -0.00601 | -0.00600 |

The oracle fits the outcome equation using true reseller demand only after ordinary estimation. It helps isolate downstream estimation error from uncertainty and approximation in `R_hat`; it is not an empirical estimator.

### Completed quantities

| Sample | Completed quantity | True realized quantity | Relative difference |
|---|---:|---:|---:|
| All observations | 962,801.46 | 961,822 | +0.102% |
| Nonreporting observations | 307,630.46 | 306,651 | +0.319% |

Individual mean-prediction RMSE is approximately 2.71 units. These are in-sample checks for one seed, not an out-of-sample accuracy guarantee.

## 11. Validation performed

- Requested quantities plus outside-option allocations exhaust purchasing capacities.
- Fulfilled quantities plus backorders equal requested quantities.
- Reporting indicators agree with missing versus observed sell-through.
- The BLP implementation recovers the homogeneous-logit limit and known nonlinear shares.
- Capacity-weighted reseller predictions aggregate to fitted product quantities.
- Both Layer 1 starting values reach the same first-step objective; the final fit does not hit nonlinear parameter bounds.
- Increasing quadrature from nine to seventeen nodes changes fitted shares by less than `6e-13`.
- Both PPML fits converge in seven iterations with a full-rank, 177-column design.
- PPML passes weighted-mean and weight-scale-invariance checks and rejects an all-zero outcome sample.
- Completed quantities retain observed sales and use predictions only for missing sales.
- All 69,120 exported predictions are present and aggregate correctly.

BLP share fit is mechanical after inversion. It is a numerical check, not independent evidence that structural parameters are accurately recovered.

## 12. Scope and remaining work

The implemented model is a simulation baseline for the two-layer framework. Its main limits are:

- **Selection correction:** reporting is selection on observables. The control-function/selection-on-unobservables extension is not implemented or validated.
- **Availability:** zero-inventory products remain orderable. A changing-choice-set or censored-fulfilled-sales version requires additional work.
- **Inventory:** no downstream inventory dynamics or stock-flow constraints are estimated.
- **Demand interpretation:** the reconstructed outcome is sell-through under the assumed mapping. Unmet end-user demand is not separately identified.
- **Heterogeneity:** reseller latent tastes are integrated out, while the generator holds realized tastes fixed. This can affect the downstream generated-regressor relationship.
- **Market size:** purchasing capacity is observed in the artificial experiment. Its construction is an empirical research decision for company data.
- **Generalization:** the scripts assume the balanced simulation's assortment and nested fixed-effect structure. Available zero-share products, new reseller/product levels, and separation need explicit treatment.
- **Inference:** standard errors and confidence intervals are not supplied. The working GMM weighting matrix does not solve dependence from persistent reseller tastes. Research inference needs a full-pipeline bootstrap with an appropriate dependence structure.
- **Recovery assessment:** repeat the experiment across seeds and larger reseller samples, check instrument/nonlinear identification, and test out-of-sample prediction before drawing empirical conclusions.

For detailed implementation notes, see `README_LAYER1.md` and `README_LAYER2.md` in the project package.
