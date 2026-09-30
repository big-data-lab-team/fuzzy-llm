"""Synthetic integration tests, including real on-disk 8/16/32 analysis."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import distributions as dist
import drift_study as study


class StudyTest(unittest.TestCase):
    def test_histogram_overflow(self):
        q=dist.hist_quantiles(np.array([10]), np.array([0.,1.]), 10,
                             quantiles=(.05,.5,.95),under=10,over=80)
        self.assertIsNone(q[.05])
        self.assertIsNone(q[.5])
        self.assertIsNone(q[.95])
        q=dist.hist_quantiles(np.array([80]),np.array([0.,1.]),80,
                             quantiles=(.5,),under=10,over=10)
        self.assertAlmostEqual(q[.5],.5)

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'captures';out=Path(tmp)/'report'
            rng=np.random.default_rng(1209)
            shape=(12,9)
            lam=rng.normal(size=shape);targets=np.arange(12,dtype=np.int32)%9
            for site in study.SITES:
                for t in study.BITS:
                    for mode in ('sr','rn'):
                        deltas=[rng.normal(scale=2**(5-t),size=shape) for _ in range(32 if mode=='sr' else 1)]
                        dist._write_tree(str(root),lam,targets,deltas,site=site,t=t,mode=mode)
            for mode in ('sr','rn'):
                deltas=[rng.normal(scale=1e-5,size=shape) for _ in range(8)] if mode=='sr' else [np.zeros(shape)]
                dist._write_tree(str(root),lam,targets,deltas,site='null',t=24,mode=mode)
            for path in root.glob('*/*/*/seed*/meta.json'):
                meta=json.loads(path.read_text())
                if meta['site']=='null':meta['site']='none'
                with np.load(path.parent/'scalars.npz') as z:
                    meta['perplexity']=float(np.exp(z['nll_pert'].mean()))
                meta['vfc_backends']='libinterflop_prism.so --seed='+meta['seed']+(' --mode=rn' if meta['mode']=='rn' else '')
                study.write_json(path,meta)
            replica=root/'reference_check'/'rn_seed2'
            shutil.copytree(root/'reference',replica)
            small=dict(context_length=256,max_tokens=13,num_windows=1,window_idx=0,n_scored=12,vocab=9)
            with patch.object(study,'SETTINGS',small):
                missing=root/'lm_head/t06/sr/seed32/meta.json'
                data=missing.read_text();missing.unlink()
                with self.assertRaisesRegex(ValueError,'missing'):study.audit(root,out)
                missing.write_text(data)
                gate=study.audit(root,out)
                self.assertTrue(gate['rn24_bitwise_identical'])
                for site in study.SITES:
                    for t in study.BITS:study.cell(root,out,site,t)
                if importlib.util.find_spec('matplotlib'):
                    study.report(root,out)
                    self.assertTrue((out/'figures/aggregate.pdf').is_file())
                    self.assertEqual(len(list((out/'figures').glob('*.pdf'))),7)
                else:
                    with patch.object(study,'plot_summary'),patch.object(study,'plot_selected'):
                        study.report(root,out)
                import csv
                rows=study.read_csv(out/'distributions.csv')
                self.assertEqual(len(rows),24)
                for row in rows:
                    if row['mode']=='sr':
                        self.assertGreater(float(row['se_var_logq_y_mean']),0)
                        self.assertGreater(float(row['se_ep_var_logq_mean']),0)
                        self.assertLess(float(row['probability_bias_sum_error']),1e-12)
                    else:self.assertEqual(row['sigma2_mean'],'')
                before=json.loads((out/'DISTRIBUTIONS_COMPLETE.json').read_text())
                self.assertEqual(before['status'],'complete')
                # Mutation after the audit cannot reuse the final analysis.
                path=root/'lm_head/t06/sr/seed1/meta.json'
                path.write_text(path.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'changed after'):study.report(root,out)


if __name__=='__main__':unittest.main()
