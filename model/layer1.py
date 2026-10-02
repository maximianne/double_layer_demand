"""Demand-only BLP: capacity-weighted quadrature, contraction, two-step IV/GMM.

Run from this folder: python layer1.py --data data --output results/layer1
No ground-truth columns are used to estimate or predict demand.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize

LINEAR = ['intercept', 'quality', 'price', 'scarcity_low', 'scarcity_medium', 'scarcity_zero']


class Layer1:
    def __init__(self, data='data', quadrature_nodes=9, micro_moments=True):
        data = Path(data)
        self.products = pd.read_csv(data/'blp_product_market.csv').sort_values(['market_period_id','sku']).reset_index(drop=True)
        self.resellers = pd.read_csv(data/'resellers.csv')
        self.cap = pd.read_csv(data/'reseller_capacity.csv').merge(
            self.resellers[['reseller_id','log_size']], on='reseller_id', validate='many_to_one')
        f = self.products
        if f.duplicated(['market_period_id','sku']).any():
            raise ValueError('Duplicate product-market observations')
        if not ((f.share > 0) & (f.outside_share > 0)).all():
            raise ValueError('Positive inside/outside shares required; choose an explicit zero-share strategy.')
        self.markets = list(f.market_period_id.unique())
        counts = f.groupby('market_period_id').size()
        if counts.nunique()!=1:
            raise ValueError('This baseline supports the balanced simulated assortment only.')
        self.M, self.J = len(self.markets), int(counts.iloc[0])
        self.N = len(f)
        self.s = f.share.to_numpy().reshape(self.M,self.J)
        self.quality = f.quality.to_numpy().reshape(self.M,self.J)
        groups = [self.cap[self.cap.market_period_id.eq(m)].sort_values('reseller_id') for m in self.markets]
        if len({len(g) for g in groups})!=1:
            raise ValueError('Balanced reseller counts per market required for this baseline.')
        self.groups = groups
        self.use_micro = micro_moments
        panel = pd.read_csv(data/'reseller_panel.csv')
        scores=[]
        for m,g in zip(self.markets, groups):
            pm=f[f.market_period_id.eq(m)]
            q=panel[panel.market_period_id.eq(m)].pivot(index='reseller_id',columns='sku',values='quantity_requested')
            q=q.loc[g.reseller_id,pm.sku].to_numpy()
            scores.append(q@pm.quality.to_numpy()/g.purchasing_capacity.to_numpy())
        self.observed_quality_score=np.stack(scores)
        self.capacity = np.stack([g.purchasing_capacity.to_numpy() for g in groups])
        self.size = np.stack([g.log_size.to_numpy() for g in groups])
        totals = self.capacity.sum(axis=1)
        if not np.allclose(totals, f.groupby('market_period_id',sort=True).market_size.first().to_numpy()):
            raise ValueError('Capacity totals do not match market sizes')
        self.weights = self.capacity/totals[:,None]
        nodes, weights = hermgauss(quadrature_nodes)
        self.nodes, self.node_weights = np.sqrt(2)*nodes, weights/np.sqrt(np.pi)
        self.X = np.column_stack([np.ones(self.N), f[LINEAR[1:]].to_numpy()])
        # Endogenous scarcity indicators are NOT included among their own instruments.
        delay = f.supplier_delay_shock.to_numpy()
        cost = f.acquisition_cost.to_numpy()
        quality = f.quality.to_numpy()
        raw = np.column_stack([np.ones(self.N), quality, cost, f.freight_cost,
            delay, delay**2, delay**3, cost*quality, delay*quality, delay**2*quality])
        self.instrument_names = ['intercept','quality','acquisition_cost','freight_cost',
            'delay','delay_squared','delay_cubed','cost_quality','delay_quality','delay_squared_quality']
        self.zmean = raw.mean(axis=0)
        self.zscale = raw.std(axis=0)
        self.zmean[0], self.zscale[0] = 0., 1.
        self.Z = (raw-self.zmean)/self.zscale
        if np.linalg.matrix_rank(self.Z)!=self.Z.shape[1]:
            raise ValueError('Instrument matrix not full rank')
        self.ZX = self.Z.T@self.X/self.N
        if np.linalg.matrix_rank(self.ZX)!=self.X.shape[1]:
            raise ValueError('Instrument-regressor cross-moment not full rank')
        self.delta0 = np.log(self.s)-np.log(f.outside_share.to_numpy().reshape(self.M,self.J))
        self.evaluations = 0

    def micro(self, delta, theta):
        numer=np.exp(delta[:,None,None,:])*self.exp_heterogeneity(theta)
        p=numer/(1+numer.sum(axis=-1,keepdims=True))
        score=np.einsum('maqj,mj->maq',p,self.quality)
        second=np.einsum('maqj,mj->maq',p,self.quality**2)
        expected=np.einsum('maq,q->ma',score,self.node_weights)
        expected_squared=np.einsum('maq,q->ma',score**2+(second-score**2)/self.capacity[:,:,None],self.node_weights)
        # Score = total ordered quality / capacity; outside units contribute zero.
        g1=np.mean(self.size*(self.observed_quality_score-expected),axis=1)
        g2=np.mean(self.observed_quality_score**2-expected_squared,axis=1)
        return np.repeat(np.column_stack([g1,g2]),self.J,axis=0)

    def exp_heterogeneity(self, theta):
        sigma, pi = theta
        deviation = pi*self.size[:,:,None] + sigma*self.nodes[None,None,:]
        return np.exp(deviation[:,:,:,None]*self.quality[:,None,None,:])

    def probabilities(self, delta, heterogeneity):
        numer = np.exp(delta[:,None,None,:])*heterogeneity
        prob = numer/(1+numer.sum(axis=-1,keepdims=True))
        individual = np.einsum('maqj,q->maj', prob, self.node_weights)
        shares = np.einsum('maj,ma->mj', individual, self.weights)
        return shares, individual

    def invert(self, theta, tolerance=1e-10, max_iterations=3000):
        heterogeneity = self.exp_heterogeneity(theta)
        delta = self.delta0.copy()
        for iteration in range(max_iterations):
            shares, _ = self.probabilities(delta,heterogeneity)
            if not np.isfinite(shares).all() or (shares<=0).any():
                raise RuntimeError('Invalid predicted shares')
            change = np.log(self.s)-np.log(shares)
            delta += change
            if np.max(np.abs(change))<tolerance:
                return delta, iteration+1
        raise RuntimeError('BLP contraction failed to converge')

    def concentrate(self, delta, W, micro=None):
        Zy = self.Z.T@delta.ravel()/self.N
        ZX=self.ZX
        if micro is not None:
            Zy=np.r_[Zy,micro.mean(axis=0)]
            ZX=np.vstack([ZX,np.zeros((2,self.X.shape[1]))])
        beta = np.linalg.solve(ZX.T@W@ZX, ZX.T@W@Zy)
        residual = delta.ravel()-self.X@beta
        g = self.Z.T@residual/self.N
        if micro is not None: g=np.r_[g,micro.mean(axis=0)]
        return beta,residual,g

    def objective(self, theta, W):
        self.evaluations += 1
        delta,_ = self.invert(theta)
        micro=self.micro(delta,theta) if self.use_micro else None
        _,_,g = self.concentrate(delta,W,micro)
        return float(self.N*g@W@g)

    def fit(self, starts=((.2,.1),(.5,.3)), max_iterations=100):
        W1 = np.linalg.inv(self.Z.T@self.Z/self.N)
        benchmark,_,_ = self.concentrate(self.delta0,W1)
        if self.use_micro:
            full=np.eye(W1.shape[0]+2)
            full[:-2,:-2]=W1
            W1=full
        bounds = [(0.,1.5),(-.8,.8)]
        first = [minimize(self.objective, s, args=(W1,), method='L-BFGS-B',bounds=bounds,
            options={'maxiter':max_iterations,'ftol':1e-10,'gtol':1e-5}) for s in starts]
        converged = [r for r in first if r.success]
        if not converged:
            raise RuntimeError(f'All first-step optimizations failed: {[r.message for r in first]}')
        best = min(converged,key=lambda r:r.fun)
        d1,_ = self.invert(best.x)
        micro1=self.micro(d1,best.x) if self.use_micro else None
        _,xi1,_ = self.concentrate(d1,W1,micro1)
        # Cluster moment sums by category-region-quarter, across monthly periods.
        cluster = self.products.market_id.to_numpy()
        moment_rows = self.Z*xi1[:,None]
        if micro1 is not None: moment_rows=np.column_stack([moment_rows,micro1])
        sums = pd.DataFrame(moment_rows).groupby(cluster).sum().to_numpy()
        S = sums.T@sums/self.N
        if np.linalg.matrix_rank(S)!=S.shape[0]:
            raise RuntimeError('Cluster moment covariance singular; more clusters/instrument revision needed.')
        W2 = np.linalg.inv(S)
        second = minimize(self.objective,best.x,args=(W2,),method='L-BFGS-B',bounds=bounds,
            options={'maxiter':max_iterations,'ftol':1e-10,'gtol':1e-5})
        if not second.success:
            raise RuntimeError(f'Second-step optimization failed: {second.message}')
        delta,iters = self.invert(second.x)
        beta,xi,g = self.concentrate(delta,W2,self.micro(delta,second.x) if self.use_micro else None)
        shares, individual = self.probabilities(delta,self.exp_heterogeneity(second.x))
        self.theta, self.beta, self.delta = second.x,beta,delta
        self.fitted_individual = individual
        diagnostics = dict(step1_theta=best.x.tolist(), step2_theta=second.x.tolist(),
            step1_objective=float(best.fun),step2_objective=float(second.fun),
            optimizer_success=bool(second.success),optimizer_message=str(second.message),
            micro_moments=self.use_micro,
            objective_evaluations=self.evaluations,final_contraction_iterations=iters,
            max_share_error=float(np.max(np.abs(shares-self.s))),
            instrument_rank=int(np.linalg.matrix_rank(self.Z)),
            instrument_regressor_rank=int(np.linalg.matrix_rank(self.ZX)),
            cross_moment_condition=float(np.linalg.cond(self.ZX)),
            cluster_count=len(sums),weight_covariance_condition=float(np.linalg.cond(S)),
            nonlinear_bound_hit=bool(np.any(np.isclose(second.x,[0.,-.8],atol=1e-4))
                or np.any(np.isclose(second.x,[1.5,.8],atol=1e-4))),
            first_step_starts=[dict(start=list(s),success=bool(r.success),objective=float(r.fun))
                for s,r in zip(starts,first)])
        coefficients = pd.DataFrame({'parameter':LINEAR+['quality_random_sd','quality_size_interaction'],
            'estimate':np.r_[beta,second.x], 'logit_iv':np.r_[benchmark,np.nan,np.nan]})
        product_results = self.products.copy()
        product_results['delta_hat'] = delta.ravel()
        product_results['xi_hat'] = xi
        product_results['predicted_share'] = shares.ravel()
        # Instrument relevance: partial first-stage F for four endogenous regressors.
        included = self.Z[:,:2]
        first_stage=[]
        for col in ['price','scarcity_low','scarcity_medium','scarcity_zero']:
            y=self.products[col].to_numpy()
            e0=y-included@np.linalg.lstsq(included,y,rcond=None)[0]
            e1=y-self.Z@np.linalg.lstsq(self.Z,y,rcond=None)[0]
            fstat=((e0@e0-e1@e1)/(self.Z.shape[1]-2))/(e1@e1/(self.N-self.Z.shape[1]))
            first_stage.append(dict(regressor=col,partial_F=float(fstat)))
        return coefficients,product_results,diagnostics,pd.DataFrame(first_stage)

    def predict(self):
        """Integrate latent tastes conditional on observed size; do not recover realized tastes."""
        rows=[]
        for m,g in enumerate(self.groups):
            p=self.products[self.products.market_period_id.eq(self.markets[m])]
            for a,reseller in enumerate(g.itertuples(index=False)):
                for j,product in enumerate(p.itertuples(index=False)):
                    rows.append(dict(reseller_id=reseller.reseller_id,sku=product.sku,
                        market_period_id=product.market_period_id,market_id=product.market_id,
                        ordering_period=product.ordering_period,
                        predicted_reseller_share=float(self.fitted_individual[m,a,j]),
                        R_hat=float(self.capacity[m,a]*self.fitted_individual[m,a,j])))
        return pd.DataFrame(rows)


def run(data='data',output='results/layer1'):
    model=Layer1(data)
    coefficients,product_results,diagnostics,first_stage=model.fit()
    predictions=model.predict()
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    coefficients.to_csv(out/'coefficients.csv',index=False)
    product_results.to_csv(out/'product_market_results.csv',index=False)
    predictions.to_csv(out/'reseller_demand_predictions.csv',index=False)
    panel=pd.read_csv(Path(data)/'reseller_panel.csv').merge(
        predictions[['reseller_id','sku','market_period_id','R_hat']],
        on=['reseller_id','sku','market_period_id'],how='left',validate='one_to_one')
    if panel.R_hat.isna().any(): raise RuntimeError('Missing reseller predictions')
    panel.to_csv(out/'layer2_input.csv',index=False)
    first_stage.to_csv(out/'instrument_diagnostics.csv',index=False)
    (out/'fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    # Optional ground truth is accessed only AFTER estimation and prediction.
    truth_path=Path(data)/'true_parameters.json'
    if truth_path.exists():
        truth=json.loads(truth_path.read_text())['blp']
        mapping={'price':'price_coefficient','quality':'quality_mean'}
        coefficients['true_value']=[truth[mapping.get(k,k)] for k in coefficients.parameter]
        coefficients['error']=coefficients.estimate-coefficients.true_value
        coefficients.to_csv(out/'parameter_recovery.csv',index=False)
    print(coefficients.to_string(index=False))
    print(json.dumps(diagnostics,indent=2))
    return model,coefficients,predictions,diagnostics


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data')
    parser.add_argument('--output',default='results/layer1')
    args=parser.parse_args()
    run(args.data,args.output)
