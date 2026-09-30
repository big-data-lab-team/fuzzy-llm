#!/usr/bin/env python3
"""Validated postprocessing of the frozen two-site, t=6/7/8, 32-seed study.

Stages: audit (complete capture gate), cell (one site/t, all seed budgets),
report (merge and plot), selftest. No inference or scheduler mutation here.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import decompose as dec
import distributions as dist

SITES = ('lm_head', 'mlp_c_proj')
BITS = (6, 7, 8)
BUDGETS = (8, 16, 32)
SETTINGS = dict(context_length=256, max_tokens=1024, num_windows=1,
                window_idx=0, n_scored=1020, vocab=50257)
VERSION = 'drift-study-1'


def sha(path):
    return dist.sha256_of(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(path.name+'.partial')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def settings_match(meta, expected, name):
    for key, want in expected.items():
        if meta.get(key) != want:
            raise ValueError(f'{name}: {key}={meta.get(key)!r}, expected {want!r}')


def check_capture(path, ref, settings, site, bits, mode, seed, full=False):
    """Metadata/target gate; full adds a streaming finite scan and source hashes."""
    path = Path(path)
    meta = json.loads((path/'meta.json').read_text())
    settings_match(meta, dict(settings, site=site, precision=bits, mode=mode), path)
    if str(meta.get('seed')) != str(seed) or meta.get('nonfinite_delta') != 0:
        raise ValueError(f'{path}: seed/nonfinite metadata invalid')
    backend = meta.get('vfc_backends', '')
    if 'libinterflop_prism.so' not in backend or ('--mode=rn' in backend) != (mode == 'rn'):
        raise ValueError(f'{path}: background backend does not match the capture protocol')
    if not math.isfinite(meta['perplexity']) or meta['perplexity'] <= 0:
        raise ValueError(f'{path}: invalid perplexity')
    d = np.load(path/'delta.npy', mmap_mode='r')
    shape = (settings['n_scored'], settings['vocab'])
    if d.shape != shape or d.dtype != np.float32:
        raise ValueError(f'{path}: expected float32 delta {shape}, got {d.shape}/{d.dtype}')
    with np.load(path/'scalars.npz') as z, np.load(Path(ref)/'scalars.npz') as rz:
        if not np.array_equal(z['targets'], rz['targets']):
            raise ValueError(f'{path}: scored targets differ from reference')
        for key in ('Ep_d','d_y','dL','W','l2','nll_pert'):
            if z[key].shape != (shape[0],) or not np.isfinite(z[key]).all():
                raise ValueError(f'{path}: invalid scalar {key}')
        D = z['Ep_d']-z['d_y']
        K = z['dL']-D
        if K.min() < -1e-8:
            raise ValueError(f'{path}: negative exact KL')
        if not np.allclose(z['nll_pert']-rz['nll'], z['dL'], atol=1e-10, rtol=1e-10):
            raise ValueError(f'{path}: inconsistent reference NLL')
        expected_loss = math.log(meta['perplexity'])-float(rz['nll'].mean())
        if abs(float(z['dL'].mean())-expected_loss) > 1e-4*max(1,abs(expected_loss)):
            raise ValueError(f'{path}: harness log-PPL discrepancy')
    if full:
        for start in range(0, shape[0], 32):
            if not np.isfinite(d[start:start+32]).all():
                raise ValueError(f'{path}: nonfinite delta')
    return dict(path=str(path), files={f: dict(size=(path/f).stat().st_size,
                mtime_ns=(path/f).stat().st_mtime_ns,
                **({'sha256': sha(path/f)} if full else {}))
                for f in ('meta.json','scalars.npz','delta.npy')})


def reference_gate(root, settings):
    ref = root/'reference'
    rm = json.loads((ref/'meta.json').read_text())
    settings_match(rm, dict(settings, site='none', precision=24, mode='rn'), ref)
    logits = np.load(ref/'logits.npy', mmap_mode='r')
    with np.load(ref/'scalars.npz') as z:
        targets, nll = z['targets'], z['nll']
    chk = root/'reference_check'/'rn_seed2'
    cm = json.loads((chk/'meta.json').read_text())
    settings_match(cm, dict(settings, site='none', precision=24, mode='rn'), chk)
    replica = np.load(chk/'logits.npy', mmap_mode='r')
    with np.load(chk/'scalars.npz') as z:
        if not np.array_equal(targets, z['targets']):
            raise ValueError('reference replicas have different targets')
    if logits.shape != replica.shape or logits.shape != (settings['n_scored'],settings['vocab']):
        raise ValueError('reference shape mismatch')
    error = 0.
    for start in range(0,len(targets),32):
        a = np.asarray(logits[start:start+32], dtype=np.float64)
        if not np.isfinite(a).all() or not np.array_equal(logits[start:start+32],replica[start:start+32]):
            raise ValueError('RN24 reference is nonfinite or not bitwise reproducible')
        ls = dec.log_softmax_np(a)
        y = targets[start:start+len(a)]
        error = max(error,float(np.abs(-ls[np.arange(len(a)),y]-nll[start:start+len(a)]).max()))
    if error > 1e-10:
        raise ValueError('reference NLL reconstruction failed')
    return dict(reference_nll_max_error=error, rn24_bitwise_identical=True,
                targets_sha256=hashlib.sha256(targets.tobytes()).hexdigest(),
                reference_files={f:sha(ref/f) for f in ('meta.json','logits.npy','scalars.npz')},
                reference_replica_sha256=sha(chk/'logits.npy'))


def audit(root, out):
    root, out = Path(root), Path(out)
    # Check presence before large reads or rebuilding any caches.
    paths = [(s,t,m,k,root/s/f't{t:02d}'/m/f'seed{k}')
             for s in SITES for t in BITS for m in ('sr','rn')
             for k in (range(1,33) if m=='sr' else (1,))]
    missing = [str(p/f) for *_,p in paths for f in ('meta.json','scalars.npz','delta.npy') if not (p/f).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} capture files missing; first: {missing[:3]}')
    for s in SITES:
        for t in BITS:
            for mode in ('sr','rn'):
                actual = set(dec.cell_seeds(root/s/f't{t:02d}'/mode))
                expected = {f'seed{k}' for k in (range(1,33) if mode=='sr' else (1,))}
                if actual != expected:
                    raise ValueError(f'{s}/{t}/{mode}: capture seed membership is not frozen pool')
    meta = reference_gate(root, SETTINGS)
    sources = []
    for s,t,m,k,p in paths:
        sources.append(check_capture(p,root/'reference',SETTINGS,s,t,m,k,full=True))
        print(f'audited {s} t{t} {m} seed{k}', flush=True)
    nulls = []
    for mode in ('rn','sr'):
        for seed in ((1,) if mode=='rn' else range(1,9)):
            p = root/'null'/'t24'/mode/f'seed{seed}'
            sources.append(check_capture(p,root/'reference',SETTINGS,'none',24,mode,seed,full=True))
            with np.load(p/'scalars.npz') as z:
                D = z['Ep_d']-z['d_y']
                K = z['dL']-D
                nulls.append(dict(mode=mode,seed=seed,D=float(D.mean()),K=float(K.mean()),dL=float(z['dL'].mean())))
                if mode=='rn' and (np.any(D) or np.any(z['dL']) or np.any(z['W'])):
                    raise ValueError('RN null scalars not exactly zero')
            if mode=='rn' and np.any(np.load(p/'delta.npy',mmap_mode='r')):
                raise ValueError('RN null delta not exactly zero')
    meta.update(version=VERSION, settings=SETTINGS, sources=sources, null_controls=nulls,
                root=str(root), source_hashes={p.name:sha(p) for p in (Path(__file__),Path(dist.__file__),Path(dec.__file__))},
                status='capture_gate_passed', created_unix=time.time())
    write_json(out/'capture_audit.json',meta)
    write_csv(out/'null_controls.csv',nulls)
    return meta


def selection(ref):
    lam = np.load(Path(ref)/'logits.npy',mmap_mode='r')
    with np.load(Path(ref)/'scalars.npz') as z:
        targets=z['targets']
    positions = np.linspace(0,len(targets)-1,4,dtype=int)
    rows=[]
    for pos in positions:
        # Stable vocabulary-id tie breaking; choices depend only on reference.
        top=np.argsort(-np.asarray(lam[pos]),kind='stable')[:5].tolist()
        ids=list(dict.fromkeys([int(targets[pos])]+top))
        rows.extend(dict(position=int(pos),vocab_id=j,is_target=j==int(targets[pos]),
                         reference_rank=(top.index(j)+1 if j in top else None)) for j in ids)
    return rows


def selected_export(root, site, bits, choices, out):
    ref=root/'reference'
    lam=np.load(ref/'logits.npy',mmap_mode='r')
    positions=sorted({r['position'] for r in choices})
    base=np.asarray(lam[positions],dtype=np.float64)
    p=np.exp(dec.log_softmax_np(base))
    rows=[]
    norm_error=0.
    for mode in ('sr','rn'):
        for seed in (range(1,33) if mode=='sr' else (1,)):
            d=np.asarray(np.load(root/site/f't{bits:02d}'/mode/f'seed{seed}'/'delta.npy',mmap_mode='r')[positions],dtype=np.float64)
            logits=base+d
            q=np.exp(dec.log_softmax_np(logits))
            norm_error=max(norm_error,float(np.abs(q.sum(axis=1)-1).max()))
            for choice in choices:
                i=positions.index(choice['position']); j=choice['vocab_id']
                rows.append(dict(site=site,t=bits,mode=mode,seed=seed,**choice,
                    reference_lambda=float(base[i,j]),reference_probability=float(p[i,j]),
                    logit=float(logits[i,j]),probability=float(q[i,j]),
                    delta=float(d[i,j]),probability_error=float(q[i,j]-p[i,j])))
    if norm_error>1e-12:
        raise ValueError('selected softmax normalization failed')
    write_csv(out/'selected.csv',rows)
    np.savez_compressed(out/'selected.npz',**{k:np.array([r[k] for r in rows]) for k in rows[0] if k!='reference_rank'})
    return norm_error


def load_gate(root,out,site=None,bits=None):
    meta=json.loads((out/'capture_audit.json').read_text())
    if meta['status']!='capture_gate_passed' or Path(meta['root']).resolve()!=root.resolve():
        raise ValueError('capture gate belongs to another root')
    for path in (Path(__file__),Path(dist.__file__),Path(dec.__file__)):
        if sha(path)!=meta['source_hashes'][path.name]:
            raise ValueError(f'{path.name}: analysis source changed after audit')
    for record in meta['sources']:
        p=Path(record['path'])
        if site and not (p.is_relative_to(root/site/f't{bits:02d}')):
            continue
        for name, prior in record['files'].items():
            st=(p/name).stat()
            if (st.st_size,st.st_mtime_ns)!=(prior['size'],prior['mtime_ns']):
                raise ValueError(f'{p/name}: changed after capture audit')
    for name,expected in meta['reference_files'].items():
        if sha(root/'reference'/name)!=expected:
            raise ValueError(f'reference {name} changed after audit')
    return meta


def cell(root,out,site,bits,budget_bytes=700_000_000):
    root,out=Path(root),Path(out)
    gate=load_gate(root,out,site,bits)
    dest=out/'cells'/f'{site}_t{bits:02d}'
    dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'COMPLETE.json'
    marker.unlink(missing_ok=True)
    ref=root/'reference'
    ref_logits=np.load(ref/'logits.npy',mmap_mode='r')
    scales=dist.logit_scales(ref_logits)
    summaries=[]; per_seed=[]
    for mode in ('sr','rn'):
        cdir=root/site/f't{bits:02d}'/mode
        wanted={f'seed{k}' for k in (range(1,33) if mode=='sr' else (1,))}
        # Cache provenance includes every delta hash and the reference hash.
        fp=dict(reference=gate['reference_files']['logits.npy'],deltas={Path(s['path']).name:s['files']['delta.npy']['sha256']
                  for s in gate['sources'] if Path(s['path']).parent==cdir})
        cache_meta=cdir/'gram_sources.json'
        rebuild=not cache_meta.exists() or json.loads(cache_meta.read_text())!=fp
        names,gram=dec.get_gram(str(cdir),str(ref),rebuild=rebuild)
        if set(names)!=wanted or len(names)!=len(wanted):
            raise ValueError('Gram seed membership differs from completed pool')
        write_json(cache_meta,fp)
        scal=dec.load_scalars(str(cdir),names)
        with np.load(ref/'scalars.npz') as z:
            ref_loss=float(z['nll'].mean())
        for i,name in enumerate(names):
            D=float((scal['Ep_d'][i]-scal['d_y'][i]).mean())
            loss=float(scal['dL'][i].mean())
            per_seed.append(dict(site=site,t=bits,mode=mode,seed=int(name[4:]),D=D,K=loss-D,dL=loss,ppl=math.exp(ref_loss+loss)))
        for S in (BUDGETS if mode=='sr' else (1,)):
            seed_names=[f'seed{k}' for k in range(1,S+1)]
            res=dist.cell_stats(str(cdir),str(ref),mode,bits,seed_names=seed_names,
                    budget_bytes=budget_bytes,jack=True,softmax=True,hist=True,
                    want_per_token=True,ref_logits=ref_logits,scales=scales)
            row=dist.to_row(res,site,bits,mode,scales,str(root),0,budget_bytes,
                            dist.chunk_budget(ref_logits.shape[1],S,budget_bytes),'float32')
            row.update(dist.gram_cross_check(str(cdir),str(ref),seed_names,res,mode=='rn'))
            if row.get('gram_check_skipped'):
                raise ValueError(row['gram_check_skipped'])
            for key in ('ep_d_abserr','d_y_abserr','dL_abserr','K_abserr'):
                if not math.isfinite(float(row[key])) or float(row[key])>5e-5:
                    raise ValueError(f'{site}/{bits}/{mode}/{S}: {key} failed')
            for key in ('stored_check_error','meta_mismatch','nonfinite_delta'):
                if row.get(key):
                    raise ValueError(f'{site}/{bits}/{mode}/{S}: {key}={row[key]}')
            for key in ('probability_sum_error','probability_bias_sum_error'):
                if float(row[key]) > 1e-12:
                    raise ValueError(f'{key} failed')
            for key,tol in [('D_decompose_abserr',5e-5),('Q_decompose_abserr',1e-9),('V_decompose_abserr',1e-9)]:
                if row.get(key,'')=='' or float(row[key])>tol:
                    raise ValueError(f'{key} failed: {row.get(key)}')
            if mode=='sr' and any(row.get(k,'')=='' for k in ('se_var_logq_y_mean','se_ep_var_logq_mean')):
                raise ValueError('missing log-probability variance uncertainty')
            idx=[names.index(n) for n in seed_names]
            subgram=gram[:,idx][:,:,idx]
            subscal={k:v[idx] for k,v in scal.items()}
            quad=dec.terms(subgram,subscal,list(range(S)),mode=='rn')
            qagg=dec.aggregate(quad)
            qse=dec.jackknife(subgram,subscal,mode=='rn')
            # Keep independent scalar-based D/R/loss beside reconstructed ones.
            row.update({f'quadratic_{k}':v for k,v in qagg.items()})
            row.update({f'quadratic_{k}':v for k,v in qse.items()})
            row.update(pool='seeds_1_'+str(S),status='complete',source_version=dist.SCRIPT_VERSION)
            summaries.append(row)
            np.savez_compressed(dest/f'{mode}_s{S}_tokens.npz',**res['per_token'],**{f'quadratic_{k}':v for k,v in quad.items()})
            np.savez_compressed(dest/f'{mode}_s{S}_hist.npz',**res['hist'])
            write_csv(dest/f'{mode}_s{S}.csv',[row])
            print(f'finished {site} t{bits} {mode} S={S}',flush=True)
    choices=selection(ref)
    norm=selected_export(root,site,bits,choices,dest)
    write_csv(dest/'summary.csv',summaries)
    write_csv(dest/'seed_terms.csv',per_seed)
    write_json(marker,dict(site=site,t=bits,status='complete',selection=choices,
        softmax_normalization_max_error=norm,capture_audit_sha256=sha(out/'capture_audit.json'),
        source_hashes={p.name:sha(p) for p in (Path(__file__),Path(dist.__file__),Path(dec.__file__))},
        outputs={str(p.relative_to(dest)):sha(p) for p in dest.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))


def plot_selected(path, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    rows=read_csv(path)
    positions=sorted({int(r['position']) for r in rows})
    with PdfPages(destination) as pdf:
        for pos in positions:
            part=[r for r in rows if int(r['position'])==pos]
            ids=list(dict.fromkeys(int(r['vocab_id']) for r in part))
            fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
            for ax,key,label,refkey in zip(axes.flat,['logit','probability','delta','probability_error'],
                     ['λ','softmax(λ)','λ − λ reference','probability − reference'],
                     ['reference_lambda','reference_probability',None,None]):
                for ci,j in enumerate(ids):
                    group=[r for r in part if int(r['vocab_id'])==j]
                    sr=sorted(float(r[key]) for r in group if r['mode']=='sr')
                    rn=float(next(r[key] for r in group if r['mode']=='rn'))
                    color=f'C{ci}'
                    target=group[0]['is_target']=='True'
                    ax.step(sr,np.arange(1,len(sr)+1)/len(sr),where='post',color=color,
                            label=f'{j}'+(' (target)' if target else ''))
                    ax.axvline(rn,color=color,ls='--',alpha=.65)
                    if refkey: ax.axvline(float(group[0][refkey]),color=color,ls=':',alpha=.45)
                if not refkey: ax.axvline(0,color='black',ls=':',alpha=.45)
                ax.set(xlabel=label,ylabel='Empirical CDF across SR seeds',ylim=(0,1.03))
                ax.grid(alpha=.15)
            axes[0,0].legend(title='Vocabulary ID',fontsize=8)
            fig.suptitle(f"{rows[0]['site']} t={rows[0]['t']}, scored position {pos}\n32 SR seeds: solid; RN realization: dashed; reference: dotted")
            pdf.savefig(fig); plt.close(fig)


def plot_summary(rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(destination) as pdf:
        for keys,title in [(['b_mean','b_rms','sigma2_mean','bq_rms','bq_l1_mean','varq_mean'],'Drift and fluctuation'),
                           (['D','Q','V','R','K','dL'],'Loss diagnostics (nats)')]:
            for site in SITES:
                fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
                for ax,key in zip(axes.flat,keys):
                    for S,color in zip(BUDGETS,['C0','C1','C2']):
                        rr=sorted([r for r in rows if r['site']==site and r['mode']=='sr' and int(r['seeds'])==S],key=lambda r:int(r['t']))
                        y=[float(r[key]) for r in rr]
                        err=[tcrit(S-1)*float(r.get('se_'+key) or 0) for r in rr]
                        ax.errorbar(BITS,y,yerr=err,marker='o',color=color,label=f'SR S={S}',capsize=3)
                    rn=sorted([r for r in rows if r['site']==site and r['mode']=='rn'],key=lambda r:int(r['t']))
                    if all(r.get(key,'')!='' for r in rn):
                        ax.plot(BITS,[float(r[key]) for r in rn],'k--s',label='RN (V=0 structural)' if key=='V' else 'RN')
                    ax.axhline(0,color='grey',lw=.5);ax.set(title=key,xlabel='Precision t',xticks=BITS)
                    ax.grid(alpha=.15)
                axes[0,0].legend(fontsize=8)
                fig.suptitle(f'{site}: {title}\nPointwise 95% seed-jackknife intervals; nested seed pools; fixed text')
                pdf.savefig(fig);plt.close(fig)


def tcrit(df):
    # Keep the repository's tested Student-t implementation.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    from composition_stats import t_crit
    return t_crit(df)


def report(root,out):
    root,out=Path(root),Path(out)
    gate=load_gate(root,out)
    rows=[]; seeds=[]; markers=[]
    for site in SITES:
        for bits in BITS:
            dest=out/'cells'/f'{site}_t{bits:02d}'
            marker=json.loads((dest/'COMPLETE.json').read_text())
            if marker['capture_audit_sha256']!=sha(out/'capture_audit.json'):
                raise ValueError('cell analysis used a different capture audit')
            for p,expected in marker['outputs'].items():
                if sha(dest/p)!=expected:
                    raise ValueError(f'{dest/p}: changed since analysis completed')
            markers.append(marker)
            rows.extend(read_csv(dest/'summary.csv'))
            seeds.extend(read_csv(dest/'seed_terms.csv'))
    if len(rows)!=24 or len(seeds)!=198:
        raise ValueError(f'expected 24 summary rows and 198 seed rows, got {len(rows)}/{len(seeds)}')
    write_csv(out/'distributions.csv',rows)
    write_csv(out/'seed_terms.csv',seeds)
    for S in BUDGETS:
        write_csv(out/f'distributions_s{S}.csv',[r for r in rows if int(r['seeds']) in (1,S)])
    figs=out/'figures';figs.mkdir(exist_ok=True)
    plot_summary(rows,figs/'aggregate.pdf')
    for site in SITES:
        for bits in BITS:
            plot_selected(out/'cells'/f'{site}_t{bits:02d}'/'selected.csv',figs/f'{site}_t{bits:02d}_selected.pdf')
    lines=['# Drift and fluctuation study','',
           'Complete capture pool: two sites × three precisions × (32 SR + one RN), on the leading fixed text slice.',
           'Context 256, max_tokens 1024, 1020 scored positions. Non-target arithmetic retains each capture arm’s mode at t=24.',
           'The reference is reproducible PRISM RN24. The eight-seed SR24 null is a separately labeled background control.', '',
           '| Site | t | Mode | S | D | Q | V | R | K |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if int(r['seeds']) not in (1,32):continue
        lines.append('| '+ ' | '.join([r['site'],r['t'],r['mode'],r['seeds']]+[f'{float(r[k]):.6g}' for k in ('D','Q','V','R','K')])+' |')
    lines.extend(['','Q uses the finite-seed correction and is not clipped at zero. K and total loss are exact diagnostics; R is a residual, not an independent validation.',
                  'Raw bias includes common logit shifts. Softmax RMS/L1 bias and coordinate variance summarize separately computed seed probabilities.',
                  'Figures use pointwise 95% seed-jackknife intervals. Nested 8/16/32 pools are not independent experiments.',
                  'Selected coordinates are the target plus reference top five at scored positions 0, 339, 679 and 1019, frozen across all cells.',
                  'RN has structural V=0; blank coordinate-variance fields are not estimated zeros.',
                  'The reconstructed logits inherit the stored float32 delta error; scalar audits use tolerance 5e-5 relative to max(1, magnitude).',
                  'All results concern rounding variability conditional on this fixed text. No across-text generalization is established.',
                  '', 'Reproduce: run `drift_study.py audit`, six `cell` commands, then `report`; command details and source hashes accompany the launch manifest.'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'DISTRIBUTIONS_COMPLETE.json',dict(status='complete',cells=markers,
        capture_audit_sha256=sha(out/'capture_audit.json'),
        artifacts={str(p.relative_to(out)):sha(p) for p in [out/'distributions.csv',out/'seed_terms.csv',out/'report.md',*figs.glob('*.pdf')]}))


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        rng=np.random.default_rng(33)
        lam=rng.normal(size=(12,9)); y=np.arange(12,dtype=np.int32)%9
        dist._write_tree(str(root),lam,y,[rng.normal(scale=.1,size=lam.shape) for _ in range(32)])
        ref=root/'reference'
        settings=dict(context_length=256,max_tokens=13,num_windows=1,window_idx=0,n_scored=12,vocab=9)
        p=root/'lm_head'/'t07'/'sr'/'seed1'
        # Synthetic helper lacks capture PPL; add the same metadata as inference.
        for cp in p.parent.glob('seed*'):
            m=json.loads((cp/'meta.json').read_text())
            with np.load(cp/'scalars.npz') as z: m['perplexity']=float(np.exp(z['nll_pert'].mean()))
            m['vfc_backends']='libinterflop_prism.so --seed='+m['seed']
            write_json(cp/'meta.json',m)
        check_capture(p,ref,settings,'lm_head',7,'sr',1,full=True)
        choices=selection(ref)
        assert sorted({r['position'] for r in choices})==[0,3,7,11]
        for pos in [0,3,7,11]:
            assert any(r['is_target'] for r in choices if r['position']==pos)
        zpath=p/'scalars.npz'
        with np.load(zpath) as z: zz={k:z[k] for k in z.files}
        zz['targets']=zz['targets'].copy();zz['targets'][0]=(zz['targets'][0]+1)%9
        np.savez(zpath,**zz)
        try:check_capture(p,ref,settings,'lm_head',7,'sr',1)
        except ValueError:pass
        else:raise AssertionError('accepted misaligned targets')
    print('drift_study selftest: all checks passed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    for name in ('audit','cell','report'):
        p=sub.add_parser(name);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if name=='cell':
            p.add_argument('--site',choices=SITES,required=True);p.add_argument('--t',type=int,choices=BITS,required=True)
            p.add_argument('--budget-bytes',type=int,default=700_000_000)
    sub.add_parser('selftest')
    args=ap.parse_args()
    if args.command=='selftest':selftest()
    elif args.command=='audit':audit(args.root,args.output)
    elif args.command=='cell':cell(args.root,args.output,args.site,args.t,args.budget_bytes)
    else:report(args.root,args.output)

if __name__=='__main__':main()
