"""Tests for repair: proposals for diagnostics defects, and recording/replaying them as a recipe."""
import pytest
import geopandas
import shapely.geometry

import watershed_workflow
import watershed_workflow.diagnostics as diag
import watershed_workflow.repair as repair
import watershed_workflow.sources.standard_names as names


def _hucs():
    """A = [0,1000]x[0,1000] drains to B = [1000,2000]x[0,1000], which drains out at x=2000."""
    polys = [shapely.geometry.box(0, 0, 1000, 1000), shapely.geometry.box(1000, 0, 2000, 1000)]
    return geopandas.GeoDataFrame({names.ID: ['A', 'B'], 'tohuc': ['B', 'OUTSIDE']},
                                  geometry=polys, crs='EPSG:5070')


def _rivers(lines, reachcodes=None, **cols):
    n = len(lines)
    df = {names.ID: [str(i) for i in range(n)], names.ORDER: [1] * n, names.BANKFULL_WIDTH: [8.] * n,
          'reachcode': reachcodes if reachcodes is not None else ['01020304000001'] * n}
    df.update(cols)
    geoms = [shapely.geometry.LineString(l) if l is not None else None for l in lines]
    return geopandas.GeoDataFrame(df, geometry=geoms, crs='EPSG:5070')


ONE_STREAM = [[(1100, 500), (2000, 500)], [(200, 500), (600, 520), (1100, 500)]]
KWARGS = dict(reach_segment_target_length=50., huc_segment_target_length=100., mesh=False)


def _report(*defects):
    return diag.DefectReport(defects=list(defects))


def test_empty_geometry_proposal_rewires_hydroseq():
    # 2 -> 1 (empty) -> 0
    rivers = _rivers(ONE_STREAM + [None, ], hydroseq=[1., 3., 2.], dnhydroseq=[0., 2., 1.],
                     uphydroseq=[2., 0., 3.])
    rivers.loc[1, 'hydroseq'], rivers.loc[2, 'hydroseq'] = 3., 2.
    report = diag.findDefects(_hucs(), rivers, **KWARGS)
    props = repair.proposeFixes(report, _hucs(), rivers)
    p = props[[d.kind for d in report.defects].index('empty-geometry')][0]
    assert p.action == 'dropEmptyReaches'
    _, out = p.apply(_hucs(), rivers)
    assert len(out) == 2
    assert list(out['dnhydroseq']) == [0., 1.]       # reach 1 now drains straight to reach 0
    assert list(out['uphydroseq']) == [3., 0.]
    assert len(rivers) == 3                           # input untouched


def test_offset_divide_proposal_for_a_reach_along_the_divide():
    # reach 2 runs along the A|B divide inside A, crossing it four times; reach 3 continues it
    lines = ONE_STREAM + [[(990, 100), (1005, 200), (990, 300), (1005, 400), (990, 450)],
                          [(990, 900), (990, 460)]]
    rivers = _rivers(lines)
    defect = diag.Defect('simplify', 'multiple-crossing', 'test', reaches=['2'])
    [props] = repair.proposeFixes(_report(defect), _hucs(), rivers)
    best = props[0]
    assert best.action == 'offsetDivide'
    assert (best.params['keep'], best.params['other']) == ('A', 'B')
    assert sorted(best.params['reaches']) == ['2', '3']
    assert best.params['clearance']['2'] == pytest.approx(2.5 * 8.)
    assert best.preview['reach_crossings'] == 0
    assert best.preview['min_reach_to_divide'] > 19.5      # buffer arcs are polygonal
    assert not best.caveats
    assert set(best.geometry['role']) == {'area moved', 'divide before', 'reaches'}
    hucs, _ = best.apply(_hucs(), rivers)
    watershed_workflow.Watershed(hucs)
    # the reverse direction, if offered, comes second with a caveat
    assert all(p.caveats for p in props[1:])


def test_drop_disconnected_network_proposal():
    # a separate reach leaves B across the exterior boundary at y=0
    lines = ONE_STREAM + [[(1500, 100), (1500, 0)]]
    report = diag.findDefects(_hucs(), _rivers(lines, reachcodes=['01020304000001'] * 2 + ['09990000000001']),
                              **KWARGS)
    outlets = [i for i, d in enumerate(report.defects) if d.kind == 'multiple-outlets']
    assert len(outlets) == 1
    props = repair.proposeFixes(report, _hucs(),
                                _rivers(lines, reachcodes=['01020304000001'] * 2 + ['09990000000001']))
    [p] = props[outlets[0]]
    assert p.action == 'dropReaches' and p.params['reaches'] == ['2']
    assert p.preview['foreign'] is True and not p.caveats

    # the same network with the domain's reachcode is still offered, with a caveat
    rivers = _rivers(lines)
    [p] = repair.proposeFixes(diag.findDefects(_hucs(), rivers, **KWARGS), _hucs(), rivers)[outlets[0]]
    assert p.preview['foreign'] is False and p.caveats


def test_no_proposal_when_both_outlets_are_the_main_network():
    lines = [[(1500, 500), (2000, 500)], [(200, 300), (1000, 300), (1500, 500)],
             [(200, 700), (1000, 700), (1500, 500)]]
    report = diag.findDefects(_hucs(), _rivers(lines), **KWARGS)
    props = repair.proposeFixes(report, _hucs(), _rivers(lines))
    assert [d.kind for d in report.defects] == ['multiple-outlets']
    assert props == [[]]


def test_recipe_round_trip_and_rejections(tmp_path):
    hucs, rivers = _hucs(), _rivers(ONE_STREAM + [None, [(1500, 100), (1500, 0)]],
                                    reachcodes=['01020304000001'] * 3 + ['09990000000001'])
    recipe = repair.Recipe(metadata=dict(basin='test'))
    report = diag.findDefects(hucs, rivers, **KWARGS)
    props = repair.proposeFixes(report, hucs, rivers)
    drop_empty = props[[d.kind for d in report.defects].index('empty-geometry')][0]
    drop_other = props[[d.kind for d in report.defects].index('multiple-outlets')][0]
    h1, r1 = drop_empty.apply(hucs, rivers)
    recipe.record(drop_empty, 'approved', by='tester')
    recipe.record(drop_other, 'rejected', by='tester', note='keep it')

    f = str(tmp_path / 'recipe.json')
    recipe.save(f)
    loaded = repair.Recipe.load(f)
    assert loaded.metadata == dict(basin='test')
    assert [s['decision'] for s in loaded.steps] == ['approved', 'rejected']
    assert loaded.steps[0]['defect']['kind'] == 'empty-geometry'

    h2, r2 = repair.applyRecipe(hucs, rivers, f)      # rejected step not applied
    assert list(r2[names.ID]) == list(r1[names.ID]) == ['0', '1', '3']

    # a rejected proposal is not offered again
    report = diag.findDefects(h2, r2, **KWARGS)
    props = repair.proposeFixes(report, h2, r2, recipe=loaded)
    assert [d.kind for d in report.defects] == ['multiple-outlets']
    assert props == [[]]


def test_record_requires_a_decision_and_serializable_params():
    p = repair.Proposal('dropReaches', dict(reaches=['1']), 'drop 1')
    with pytest.raises(ValueError):
        repair.Recipe().record(p, 'maybe')
    with pytest.raises(TypeError):
        repair.Recipe().record(repair.Proposal('dropReaches', dict(reaches={1, 2}), 'x'), 'approved')
