# Two-layer demand simulation

Synthetic data for coding and checking the third-year-paper model. These are artificial quantities and prices, not company data or empirical estimates.

## Run

Use Python 3.10+ and install `numpy pandas scipy jupyter ipykernel`.
Run `python simulate_data.py --output data` from this folder, or open `01_inspect_simulated_data.ipynb`. The seed defaults to 20261002. All CSV files are included.

## Design

120 resellers, 24 products in two categories, two regions, eight quarters, and three disjoint monthly ordering periods per quarter. Each reseller belongs to one region and buys in both categories. Market identifiers use category × region × quarter; market_period_id additionally includes ordering period. The ordering-period index is a time partition, not a common scarcity label: products may have different scarcity states in the same period.

Each reseller-period-category has an observed synthetic purchasing capacity. Repeated purchasing units are allocated by a multinomial draw from random-coefficients logit probabilities, including an outside option normalized to zero. The quality coefficient depends on reseller size and a persistent normal taste draw. Aggregate shares must weight reseller probabilities by purchasing capacity, rather than averaging resellers equally.

Price and scarcity are endogenous through an unobserved product-market-period demand shock. Acquisition cost, freight, and supplier-delay shocks are generated independently of that shock. Rival quality sums are also supplied; their identifying variation is limited in this fixed assortment. Instrument relevance and rank must still be checked when choosing the estimator.

Zero upstream inventory does not remove a product from the requested-order choice set: backorders are accepted. Fit Layer 1 to quantity_requested; fulfilled quantities are supply constrained. Upstream inventory is a simplified fulfillment proxy, not a dynamic stock-flow ledger. No downstream inventory accounting is imposed. Thus this baseline checks the paper's reduced-form sell-through mapping, rather than a dynamic inventory model or a stockout-censored latent-demand model.

End-user quantities are Poisson draws from the specified log-link conditional mean using true expected reseller demand (not realized orders). Reporting uses a probit index with observed size, reseller type, quarter index, and integration score. Integration score is excluded from the end-user equation by construction. Reporting is MAR with overlap; the inverse Mills/control-function extension is not tested by this baseline. No claim of validity for real-world exclusion restrictions follows from this construction.

Sell-through reporting is observed at reseller-SKU-period level and varies over time. Sell-through omissions are NaN; observed zero sales remain zero. All reseller-product-period rows, including zero orders, appear in the panel. Product/reseller/time effects enter the outcome model. A real reseller that never reports would have an unestimable reseller fixed effect; prediction for such resellers requires a different specification. This synthetic reporting design does not solve that problem.

## Files

| File | Contents |
|---|---|
| products.csv | SKU/category/brand/quality/base cost |
| resellers.csv | Reseller/region/size/type/reporting integration |
| supply_inventory.csv | Price, instruments, availability, scarcity, fill rate |
| reseller_capacity.csv | Capacity and outside quantity by reseller-period-category |
| reseller_panel.csv | Complete panel, requested/fulfilled orders, observed or missing sell-through |
| sellthrough.csv | Reporting observations only, including observed zeros |
| blp_product_market.csv | Product characteristics, instruments, market size, inside/outside shares |
| ground_truth.csv | Hidden tastes, demand shocks, expected demands, full outcomes, reporting probabilities |
| true_parameters.json | Data-generating coefficients |
| validation_summary.json | Row counts and diagnostics |

Keep ground_truth.csv and true_parameters.json out of estimation inputs; use them only for recovery checks. A reporting sample may not contain every fixed-effect level in alternative seeds/smaller samples. Check support before fitting.

## Estimation sequence

1. Inspect data, reporting overlap, zero shares, and instrument variation.
2. Fit homogeneous logit-IV as a debugging benchmark, then random-coefficients BLP with capacity-weighted integration.
3. Generate reseller-level expected quantities using estimated tastes/demographic interactions and capacities. Aggregate BLP alone does not identify individual realized latent taste draws; integrate conditional on observed reseller characteristics or estimate individual heterogeneity with micro data.
4. Fit reporting probit and IPW-PPML. Check separation, extreme weights, and fixed-effect support.
5. Predict missing sell-through and aggregate; compare fitted conditional means and coefficients with truth. Bootstrap the full pipeline for inference.

Available products with observed zero purchase shares require an explicit estimation strategy; do not silently drop them or add arbitrary epsilon values. The simulator reports their count. Market size is observed in this synthetic experiment; empirical market-size construction remains a research decision.
