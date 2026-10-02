"""Synthetic two-layer demand data. Run: python simulate_data.py --output data"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import softmax, ndtr


def simulate(seed=20261002, n_resellers=120, n_quarters=8):
    rng = np.random.default_rng(seed)
    products = pd.DataFrame({
        'sku': [f'SKU{j:03d}' for j in range(24)],
        'category': np.repeat(['C1', 'C2'], 12),
        'brand': np.tile(np.repeat(['B1', 'B2', 'B3'], 4), 2),
        'quality': rng.uniform(0, 2, 24),
        'base_cost': rng.uniform(1, 2, 24),
        'pack_size': np.ones(24, dtype=int),
    })
    resellers = pd.DataFrame({
        'reseller_id': [f'R{i:03d}' for i in range(n_resellers)],
        'region': np.resize(['North', 'South'], n_resellers),
        'log_size': rng.normal(0, .6, n_resellers),
        'reseller_type': rng.integers(0, 2, n_resellers),
        'integration_score': rng.normal(size=n_resellers),
    })
    # Persistent simulated tastes are ground truth, never estimation inputs.
    tastes = rng.normal(size=n_resellers)
    product_fe = rng.normal(0, .15, 24)
    reseller_fe = .15 * resellers.log_size.to_numpy()
    periods = pd.period_range('2024Q1', periods=n_quarters, freq='Q')
    supply_rows, panel_rows, capacity_rows, truth_rows = [], [], [], []
    for t, quarter in enumerate(periods):
        time_fe = .08 * np.sin(t)
        for region in ['North', 'South']:
            ids = np.flatnonzero(resellers.region.eq(region))
            for category in ['C1', 'C2']:
                subset = products[products.category.eq(category)]
                js = subset.index.to_numpy()
                quality = subset.quality.to_numpy()
                for ell in range(1, 4):
                    market = f'{category}_{region}_{quarter}'
                    key = f'{market}_P{ell}'
                    date = quarter.start_time + pd.DateOffset(months=ell-1)
                    xi = rng.normal(0, .25, len(js))
                    cost_shock = rng.normal(0, .2, len(js))
                    freight = rng.uniform(.05, .25, len(js))
                    delay = rng.normal(size=len(js))
                    # Instruments independent of xi; price/scarcity endogenous via xi.
                    cost = subset.base_cost.to_numpy() + cost_shock
                    price = 1.2 + cost + freight + .5 * xi
                    scarcity_index = delay + .7 * xi + rng.normal(0, .35, len(js))
                    state = np.digitize(scarcity_index, [-.7, 0, .7])
                    scarcity_effect = np.array([0., .1, .25, .45])[state]
                    delta = 1.3 + .7 * quality - .8 * price + scarcity_effect + xi
                    size = resellers.iloc[ids].log_size.to_numpy()
                    capacity = np.maximum(1, rng.poisson(400 * np.exp(size)))
                    beta_quality = .7 + .2 * size + .3 * tastes[ids]
                    utility = delta[None, :] + (beta_quality-.7)[:, None] * quality
                    probabilities = softmax(np.column_stack([np.zeros(len(ids)), utility]), axis=1)
                    quantity = np.array([rng.multinomial(int(m), p) for m, p in zip(capacity, probabilities)])
                    total_capacity = int(capacity.sum())
                    for a, i in enumerate(ids):
                        reseller = resellers.iloc[i]
                        capacity_rows.append(dict(reseller_id=reseller.reseller_id, market_id=market,
                            market_period_id=key, ordering_period=ell, purchasing_capacity=int(capacity[a]),
                            outside_quantity=int(quantity[a, 0])))
                    for b, j in enumerate(js):
                        sku = products.iloc[j].sku
                        fill_rate = np.array([1., .9, .65, .0])[state[b]]
                        supply_rows.append(dict(sku=sku, market_id=market, market_period_id=key,
                            category=category, region=region, quarter=str(quarter), ordering_period=ell,
                            date=str(date.date()), price=price[b], acquisition_cost=cost[b],
                            freight_cost=freight[b], supplier_delay_shock=delay[b],
                            planned_replenishment_days=14 + 3 * delay[b],
                            scarcity_state=['No', 'Low', 'Medium', 'ZeroInventory'][state[b]],
                            scarcity_low=int(state[b]==1), scarcity_medium=int(state[b]==2),
                            scarcity_zero=int(state[b]==3), accepts_backorders=1,
                            available_for_order=1, fill_rate=fill_rate,
                            upstream_inventory=int(quantity[:, b+1].sum() * fill_rate),
                            market_size=total_capacity,
                            rival_quality_sum=float(quality.sum()-quality[b])))
                        for a, i in enumerate(ids):
                            reseller = resellers.iloc[i]
                            true_r = capacity[a] * probabilities[a, b+1]
                            promotion = int(rng.random() < .15)
                            replenishment_days = rng.uniform(7, 30)
                            mu = np.exp(-.5 + .85*np.log1p(true_r) + .12*promotion
                                - .006*replenishment_days + reseller_fe[i] + product_fe[j]
                                + time_fe + .03*(ell-1))
                            y = int(rng.poisson(mu))
                            # MAR: selection independent of outcome conditional on observables.
                            reporting_index = .15 + .35*size[a] + .8*reseller.integration_score
                            reporting_index += .25*reseller.reseller_type + .08*t
                            pi = float(ndtr(reporting_index))
                            v = int(rng.random() < pi)
                            ordered = int(quantity[a, b+1])
                            fulfilled = int(rng.binomial(ordered, fill_rate))
                            panel_rows.append(dict(reseller_id=reseller.reseller_id, sku=sku,
                                market_id=market, market_period_id=key, category=category, region=region,
                                quarter=str(quarter), ordering_period=ell, date=str(date.date()),
                                quantity_requested=ordered, quantity_fulfilled=fulfilled,
                                backorder_quantity=ordered-fulfilled, price=price[b],
                                promotion=promotion, replenishment_days=replenishment_days,
                                reporting_indicator=v, sellthrough_quantity=y if v else np.nan))
                            truth_rows.append(dict(reseller_id=reseller.reseller_id, sku=sku,
                                market_period_id=key, true_reseller_demand=true_r,
                                true_enduser_mean=mu, full_sellthrough_quantity=y,
                                true_reporting_probability=pi, true_delta=delta[b], xi=xi[b],
                                quality_coefficient=beta_quality[a], reseller_enduser_fe=reseller_fe[i],
                                product_enduser_fe=product_fe[j], time_enduser_fe=time_fe))
    panel = pd.DataFrame(panel_rows)
    supply = pd.DataFrame(supply_rows)
    capacities = pd.DataFrame(capacity_rows)
    agg = panel.groupby(['market_period_id', 'sku'], as_index=False).quantity_requested.sum()
    blp = supply.merge(agg, on=['market_period_id', 'sku'], validate='one_to_one')
    blp = blp.merge(products[['sku','brand','quality','pack_size']], on='sku', validate='many_to_one')
    blp['share'] = blp.quantity_requested / blp.market_size
    blp['outside_share'] = 1 - blp.groupby('market_period_id').share.transform('sum')
    tables = dict(products=products, resellers=resellers, supply_inventory=supply,
        reseller_capacity=capacities, reseller_panel=panel,
        sellthrough=panel.loc[panel.reporting_indicator.eq(1),
            ['reseller_id','sku','market_period_id','date','sellthrough_quantity']],
        blp_product_market=blp, ground_truth=pd.DataFrame(truth_rows))
    parameters = dict(seed=seed, n_resellers=n_resellers, n_quarters=n_quarters,
        blp=dict(intercept=1.3, quality_mean=.7, price_coefficient=-.8,
            quality_size_interaction=.2, quality_random_sd=.3,
            scarcity_low=.1, scarcity_medium=.25, scarcity_zero=.45),
        ppml=dict(intercept=-.5, log1p_reseller_demand=.85, promotion=.12,
            replenishment_days=-.006, ordering_period_increment=.03),
        reporting=dict(intercept=.15, log_size=.35, integration_score=.8,
            reseller_type=.25, quarter_index=.08))
    validate(tables)
    return tables, parameters


def validate(tables):
    p, b, c, truth = [tables[k] for k in ['reseller_panel','blp_product_market','reseller_capacity','ground_truth']]
    keys = ['reseller_id','sku','market_period_id']
    assert not p.duplicated(keys).any()
    assert p.sellthrough_quantity.isna().eq(p.reporting_indicator.eq(0)).all()
    assert (p.quantity_fulfilled + p.backorder_quantity).eq(p.quantity_requested).all()
    assert b.share.between(0, 1).all() and b.outside_share.between(0, 1, inclusive='neither').all()
    q = p.groupby(['reseller_id','market_period_id']).quantity_requested.sum()
    cap = c.set_index(['reseller_id','market_period_id'])
    assert (q + cap.outside_quantity).eq(cap.purchasing_capacity).all()
    assert truth.true_reporting_probability.between(0,1,inclusive='neither').all()
    assert (truth.true_enduser_mean > 0).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data')
    parser.add_argument('--seed', type=int, default=20261002)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tables, params = simulate(seed=args.seed)
    for name, frame in tables.items():
        frame.to_csv(out / f'{name}.csv', index=False)
    (out / 'true_parameters.json').write_text(json.dumps(params, indent=2))
    summary = {name: len(frame) for name, frame in tables.items()}
    summary.update(reporting_rate=float(tables['reseller_panel'].reporting_indicator.mean()),
        zero_inside_share_rows=int(tables['blp_product_market'].share.eq(0).sum()),
        observed_zero_sellthrough=int(tables['sellthrough'].sellthrough_quantity.eq(0).sum()))
    (out / 'validation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
