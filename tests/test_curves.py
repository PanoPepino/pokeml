"""Fixed-prefix predictions, reconstruction and historical diagnostic safeguards."""
import json
import numpy as np
import pandas as pd
import pytest

from pokeml.evaluation.curves import checkpoints, collect_curves, curve_summary, diagnose_run
from pokeml.models.factory import make_estimator
from pokeml.models.residual import BaselineCorrectionRegressor


@pytest.mark.parametrize('variant', ['cat_native','cat_ordinal','light_gbm','xgboost'])
@pytest.mark.parametrize('mode', ['direct','correction'])
def test_final_prefix_and_reconstructed_curve(variant, mode):
    rows=pd.read_csv('datasets/pkdx_min.csv').iloc[:80]
    options={'iterations':20,'depth':3,'allow_writing_files':False} if variant.startswith('cat_') else {'n_estimators':20,'fit_uncertainty':False}
    model=BaselineCorrectionRegressor(variant, make_estimator(variant,options), mode=mode)
    model.fit(rows.iloc[:60],rows.iloc[:60].total_stats)
    total=model.get_iteration_count()
    stages=list(model.predict_at_iterations(rows.iloc[60:],checkpoints(total,10)))
    assert stages[0][0]==1 and stages[-1][0]==total
    np.testing.assert_allclose(stages[-1][1],model.predict(rows.iloc[60:]),rtol=1e-8,atol=1e-8)
    raw=model.model_.predict_at_iterations(model.preprocessor_.transform(rows.iloc[60:]),[total])
    _, raw_prediction=next(raw)
    offset=model.baseline_.predict(rows.iloc[60:]) if mode=='correction' else 0
    np.testing.assert_allclose(stages[-1][1], raw_prediction+offset)
    curves=collect_curves(model,rows.iloc[:60],rows.iloc[60:],variant,9,10)
    assert set(curves.split)=={'train','validation'}
    assert not curves.duplicated(['model','generation','split','iteration']).any()
    summary=curve_summary(curves)
    assert summary.fitted_rounds.iloc[0]==total
    assert summary.MAE_deterioration.iloc[0]>=0
    for invalid in ([0],[total+1],[1,1],[2,1],[1.5]):
        with pytest.raises(ValueError): list(model.predict_at_iterations(rows.iloc[60:],invalid))


def test_checkpoints_actual_count():
    assert checkpoints(1)==[1]
    assert checkpoints(3)==[1,2,3]
    points=checkpoints(2400)
    assert len(points)==200 and points[0]==1 and points[-1]==2400


def test_historical_parent_reproduced_and_immutable(tmp_path):
    from hashlib import sha256
    from pokeml.pipeline.workflow import WorkflowConfig,run_workflow
    rows=pd.read_csv('datasets/pkdx_min.csv').groupby('generation',group_keys=False).head(12)
    source=tmp_path/'training.csv'; rows.to_csv(source,index=False)
    result=run_workflow(WorkflowConfig(training_csv=str(source),output_root=str(tmp_path),run_name='parent',
        sensitivity=False,parameters={'light_gbm':{'n_estimators':5,'num_leaves':3}},training_curves=False))
    parent=result.run_directory
    hashes={p:sha256(p.read_bytes()).hexdigest() for p in parent.rglob('*') if p.is_file()}
    output=tmp_path/'diagnostic'
    summary=diagnose_run(parent,output,5)
    assert len(summary)==8 and set(summary.generation)=={6,7,8,9}
    assert json.loads((output/'manifest.json').read_text())['endpoint_scores_verified']
    assert all(sha256(p.read_bytes()).hexdigest()==v for p,v in hashes.items())
    with pytest.raises(ValueError,match='outside'): diagnose_run(parent,parent/'new')
    with pytest.raises(FileExistsError): diagnose_run(parent,output)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='dataset changed'): diagnose_run(parent,tmp_path/'other')


def test_mismatched_endpoint_marks_diagnostic_failed(tmp_path, monkeypatch):
    import pokeml.evaluation.curves as module
    from pokeml.pipeline.workflow import WorkflowConfig,run_workflow
    rows=pd.read_csv('datasets/pkdx_min.csv').groupby('generation',group_keys=False).head(10)
    source=tmp_path/'training.csv'; rows.to_csv(source,index=False)
    result=run_workflow(WorkflowConfig(training_csv=str(source),output_root=str(tmp_path),run_name='parent',
        sensitivity=False,parameters={'light_gbm':{'n_estimators':5}},training_curves=False))
    original=module.collect_curves
    def changed(*args,**kwargs):
        table=original(*args,**kwargs)
        table.loc[table.split=='validation','MAE']+=100
        return table
    monkeypatch.setattr(module,'collect_curves',changed)
    output=tmp_path/'diagnostic'
    with pytest.raises(ValueError,match='do not reproduce'): diagnose_run(result.run_directory,output,5)
    assert json.loads((output/'manifest.json').read_text())['status']=='failed'
