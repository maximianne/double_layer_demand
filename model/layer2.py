"""Reporting probit and IPW-PPML. Run: python layer2.py --data data --layer1 results/layer1 --output results/layer2"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr
from scipy.optimize import minimize
from scipy import sparse
from scipy.linalg import solve


def probit(X, y):
    if set(np.unique(y))!={0,1}:raise ValueError('Reporting requires both observed and missing outcomes')
    sign=2*y-1
    def loss(beta):
        q=sign*(X@beta)
        logp=log_ndtr(q)
        ratio=np.exp(-.5*q*q-.5*np.log(2*np.pi)-logp)
        return -logp.mean(), -(X.T@(sign*ratio))/len(y)
    result=minimize(loss,np.zeros(X.shape[1]),jac=True,method='BFGS',
        options={'gtol':1e-8,'maxiter':300})
    value,gradient=loss(result.x)
    if np.max(np.abs(gradient))>1e-6:
        raise RuntimeError(f'Reporting probit failed: {result.message}')
    pi=ndtr(X@result.x)
    if not ((pi>0)&(pi<1)).all():raise ValueError('Numerical reporting probabilities at zero or one')
    return result.x,pi,dict(success=bool(result.success),message=str(result.message),
        max_average_score=float(np.max(np.abs(gradient))),negative_loglik=float(value))


def design(frame, r_column='R_hat'):
    """Sparse design with identifiable normalizations for nested region/category effects."""
    names=['intercept','log1p_reseller_demand','promotion','replenishment_days']
    continuous=np.column_stack([np.ones(len(frame)),np.log1p(frame[r_column]),
        frame.promotion.to_numpy(),frame.replenishment_days.to_numpy()])
    blocks=[sparse.csr_matrix(continuous)]
    # Region effects are in market FE, category effects likewise. Omit a reseller
    # per region and a product per category to remove those linear dependencies.
    reference_resellers=set(frame.groupby('region').reseller_id.min())
    reference_products=set(frame.groupby('category').sku.min())
    for column,omit in [('reseller_id',reference_resellers),('sku',reference_products),
        ('market_id',{frame.market_id.min()}),('ordering_period',{frame.ordering_period.min()})]:
        levels=sorted(set(frame[column])-omit)
        mapping={v:k for k,v in enumerate(levels)}
        codes=frame[column].map(mapping)
        rows=np.flatnonzero(codes.notna())
        blocks.append(sparse.csr_matrix((np.ones(len(rows)),(rows,codes.iloc[rows].astype(int))),
            shape=(len(frame),len(levels))))
        names.extend([f'{column}[{v}]' for v in levels])
    return sparse.hstack(blocks,format='csr'),names


def ppml(X,y,weights=None,tolerance=1e-7,max_iterations=100):
    weights=np.ones(len(y)) if weights is None else np.asarray(weights)
    if (y<0).any() or not np.isfinite(y).all():raise ValueError('PPML requires observed nonnegative outcomes')
    if (weights<=0).any() or not np.isfinite(weights).all():raise ValueError('Invalid PPML weights')
    if np.sum(y)<=0:raise ValueError('All-zero outcome: no finite PPML intercept')
    scale=weights.sum()
    gram=(X.T@X).toarray()
    rank=int(np.linalg.matrix_rank(gram))
    if rank!=X.shape[1]:raise ValueError(f'PPML design rank {rank}, columns {X.shape[1]}')
    # Categorical all-zero outcome groups cause boundary/separation problems.
    total=np.asarray(X.T@y).ravel()
    support=np.asarray(X.sum(axis=0)).ravel()
    if ((total[4:]<=0)&(support[4:]>0)).any():raise ValueError('All-zero fixed-effect group; choose explicit separation treatment')
    beta=np.zeros(X.shape[1]);beta[0]=np.log(np.average(y,weights=weights))
    def objective(b):
        eta=np.asarray(X@b).ravel()
        if np.max(eta)>60:return np.inf
        return float(np.sum(weights*(np.exp(eta)-y*eta))/scale)
    converged=False
    for iteration in range(max_iterations):
        eta=np.asarray(X@beta).ravel();mu=np.exp(eta)
        gradient=np.asarray(X.T@(weights*(mu-y))).ravel()/scale
        if np.max(np.abs(gradient))<tolerance:
            converged=True;break
        H=(X.T@X.multiply((weights*mu)[:,None])).toarray()/scale
        direction=solve(H,gradient,assume_a='pos')
        old=objective(beta);step=1.
        while step>1e-10:
            candidate=beta-step*direction
            if objective(candidate)<=old-1e-4*step*(gradient@direction):break
            step*=.5
        if step<=1e-10:raise RuntimeError('PPML Newton line search stalled')
        beta=candidate
    if not converged:raise RuntimeError('PPML did not converge; inspect separation and conditioning')
    mu=np.exp(X@beta)
    H=(X.T@X.multiply((weights*mu)[:,None])).toarray()/scale
    return beta,dict(converged=True,iterations=iteration+1,design_rank=rank,
        columns=X.shape[1],max_average_score=float(np.max(np.abs(gradient))),
        weighted_objective=objective(beta),hessian_condition=float(np.linalg.cond(H)))


def run(data='data',layer1='results/layer1',output='results/layer2',evaluate=True):
    data=Path(data);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    panel=pd.read_csv(Path(layer1)/'layer2_input.csv')
    resellers=pd.read_csv(data/'resellers.csv')
    frame=panel.merge(resellers,on=['reseller_id','region'],validate='many_to_one')
    keys=['reseller_id','sku','market_period_id']
    if frame.duplicated(keys).any():raise ValueError('Duplicate panel keys')
    if not frame.sellthrough_quantity.isna().eq(frame.reporting_indicator.eq(0)).all():
        raise ValueError('Reporting indicator disagrees with missing outcome')
    if frame.R_hat.isna().any() or (frame.R_hat<0).any():raise ValueError('Missing or negative Layer 1 predictions')
    quarters=sorted(frame.quarter.unique());quarter_map={q:i for i,q in enumerate(quarters)}
    frame['quarter_index']=frame.quarter.map(quarter_map)
    report_names=['intercept','log_size','integration_score','reseller_type','quarter_index']
    W=np.column_stack([np.ones(len(frame)),frame[report_names[1:]].to_numpy()])
    v=frame.reporting_indicator.to_numpy(dtype=int)
    rho,pi,report_diag=probit(W,v)
    frame['pi_hat']=pi
    # No silent trimming: raw IPW baseline, with explicit overlap diagnostics.
    frame['ipw']=np.where(v==1,1/pi,np.nan)
    observed=v==1
    X,names=design(frame)
    y=frame.loc[observed,'sellthrough_quantity'].to_numpy()
    beta_unweighted,diag_unweighted=ppml(X[observed],y)
    beta,diag=ppml(X[observed],y,frame.loc[observed,'ipw'].to_numpy())
    frame['predicted_enduser_mean']=np.exp(X@beta)
    frame['completed_enduser_quantity']=np.where(observed,frame.sellthrough_quantity,
        frame.predicted_enduser_mean)
    if not np.isfinite(frame.predicted_enduser_mean).all():raise RuntimeError('Invalid end-user predictions')
    pd.DataFrame({'parameter':report_names,'estimate':rho}).to_csv(out/'reporting_coefficients.csv',index=False)
    coefficients=pd.DataFrame({'parameter':names,'ipw_ppml':beta,'unweighted_ppml':beta_unweighted})
    coefficients.to_csv(out/'ppml_coefficients.csv',index=False)
    frame.to_csv(out/'enduser_predictions.csv',index=False)
    group_cols=['sku','category','region','quarter','ordering_period']
    aggregate=frame.groupby(group_cols,as_index=False).agg(
        completed_enduser_quantity=('completed_enduser_quantity','sum'),
        predicted_enduser_mean=('predicted_enduser_mean','sum'),
        observed_sellthrough=('sellthrough_quantity','sum'),
        reporting_observations=('reporting_indicator','sum'),observations=('reseller_id','size'))
    aggregate.to_csv(out/'product_region_quarter_period_demand.csv',index=False)
    frame.groupby('sku',as_index=False)[['completed_enduser_quantity','predicted_enduser_mean']].sum().to_csv(
        out/'product_total_demand.csv',index=False)
    w=frame.loc[observed,'ipw'].to_numpy()
    diagnostics=dict(reporting=report_diag,ipw_ppml=diag,unweighted_ppml=diag_unweighted,
        rows=len(frame),reporting_rate=float(v.mean()),missing_sales=int((~observed).sum()),
        observed_zero_sales=int((y==0).sum()),min_pi=float(pi.min()),max_pi=float(pi.max()),
        max_ipw=float(w.max()),ipw_effective_sample_size=float(w.sum()**2/(w@w)),
        pi_below_005=int((pi<.05).sum()),weights_trimmed=False,
        pi_quantiles={str(q):float(np.quantile(pi,q)) for q in [.01,.05,.5,.95,.99]},
        weighted_score_norm=float(diag['max_average_score']))
    (out/'fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    if evaluate and (data/'ground_truth.csv').exists():
        truth=pd.read_csv(data/'ground_truth.csv')
        assessment=frame.merge(truth[keys+['true_reseller_demand','true_enduser_mean',
            'full_sellthrough_quantity','true_reporting_probability']],on=keys,validate='one_to_one')
        metrics={}
        for label,mask in [('all',np.ones(len(frame),dtype=bool)),('nonreporting',~observed)]:
            a=assessment.loc[mask]
            metrics[label]=dict(mean_prediction_rmse=float(np.sqrt(np.mean(
                (a.predicted_enduser_mean-a.true_enduser_mean)**2))),
                completed_total=float(a.completed_enduser_quantity.sum()),
                true_realized_total=float(a.full_sellthrough_quantity.sum()),
                completed_total_relative_error=float(a.completed_enduser_quantity.sum()/a.full_sellthrough_quantity.sum()-1))
        metrics['reporting_probability_rmse']=float(np.sqrt(np.mean((pi-assessment.true_reporting_probability)**2)))
        # Oracle comparison isolates Layer 2 from uncertainty in Layer 1 predictions.
        Xoracle,oracle_names=design(assessment,r_column='true_reseller_demand')
        beta_oracle,oracle_diag=ppml(Xoracle[observed],y,w)
        oracle=pd.DataFrame({'parameter':oracle_names,'oracle_ipw_ppml':beta_oracle})
        core=coefficients.iloc[1:4].merge(oracle,on='parameter',validate='one_to_one')
        core['true_value']=[.85,.12,-.006]
        core.to_csv(out/'core_parameter_recovery.csv',index=False)
        metrics['oracle_ppml']=oracle_diag
        (out/'evaluation_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(coefficients.head(4).to_string(index=False))
    print(json.dumps(diagnostics,indent=2))
    return frame,coefficients,diagnostics


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data');parser.add_argument('--layer1',default='results/layer1')
    parser.add_argument('--output',default='results/layer2');parser.add_argument('--no-evaluate',action='store_true')
    args=parser.parse_args()
    run(args.data,args.layer1,args.output,evaluate=not args.no_evaluate)
