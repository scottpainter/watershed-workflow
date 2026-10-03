"""Tests for diagnostics.findDefects: reporting only the defects that the
Watershed Workflow pipeline does not fix itself."""
import pytest
import geopandas
import shapely.geometry

import watershed_workflow.diagnostics
import watershed_workflow.sources.standard_names as names


def _hucs():
    """A = [0,1000]x[0,1000] drains to B = [1000,2000]x[0,1000], which drains out at x=2000."""
    polys = [shapely.geometry.box(0, 0, 1000, 1000), shapely.geometry.box(1000, 0, 2000, 1000)]
    return geopandas.GeoDataFrame({names.ID: ['A', 'B'], 'tohuc': ['B', 'OUTSIDE']},
                                  geometry=polys, crs='EPSG:5070')


def _rivers(lines):
    geoms = [shapely.geometry.LineString(l) if l is not None else None for l in lines]
    return geopandas.GeoDataFrame({names.ID: [str(i) for i in range(len(lines))],
                                   names.ORDER: [1] * len(lines),
                                   names.BANKFULL_WIDTH: [8.] * len(lines)},
                                  geometry=geoms, crs='EPSG:5070')


# one stream through A into B and out of the domain; WW cuts it at the divide
ONE_STREAM = [[(1100, 500), (2000, 500)], [(200, 500), (600, 520), (1100, 500)]]

KWARGS = dict(reach_segment_target_length=50., huc_segment_target_length=100.)


def _find(rivers, **kwargs):
    return watershed_workflow.diagnostics.findDefects(_hucs(), rivers, **KWARGS, **kwargs)


def test_clean_case_has_no_defects_and_meshes():
    report = _find(_rivers(ONE_STREAM), min_cell_area=1.)
    assert report.ok, report.summary()
    assert report.mesh is not None
    assert len(report.to_dataframe()) == 0


def test_inputs_not_modified():
    hucs, rivers = _hucs(), _rivers(ONE_STREAM)
    h0, r0 = hucs.copy(), rivers.copy()
    watershed_workflow.diagnostics.findDefects(hucs, rivers, **KWARGS, mesh=False)
    assert all(a.equals(b) for a, b in zip(hucs.geometry, h0.geometry))
    assert all(a.equals(b) for a, b in zip(rivers.geometry, r0.geometry))


def test_empty_geometry_reported():
    report = _find(_rivers(ONE_STREAM + [None, ]), mesh=False)
    kinds = [d.kind for d in report.defects]
    assert kinds == ['empty-geometry']
    assert report.defects[0].reaches == ['2']


def test_multiple_outlets_reported():
    # two separate branches leave A and join in B
    lines = [[(1500, 500), (2000, 500)],
             [(200, 300), (1000, 300), (1500, 500)],
             [(200, 700), (1000, 700), (1500, 500)]]
    report = _find(_rivers(lines), mesh=False)
    outlets = [d for d in report.defects if d.kind == 'multiple-outlets']
    assert len(outlets) == 1
    assert outlets[0].hucs == ['A']
    assert len(outlets[0].details['outlets']) == 2


def test_small_cells_reported_as_sites():
    report = _find(_rivers(ONE_STREAM), min_cell_area=1.e9)
    small = [d for d in report.defects if d.kind == 'small-cells']
    assert len(small) >= 1
    assert sum(len(d.details['cells']) for d in small) == len(report.mesh.conn)
    df = report.to_dataframe()
    assert len(df) == len(report.defects) and not df.geometry.is_empty.any()


def test_mesh_skipped_after_fatal_defect(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('cannot cut')
    monkeypatch.setattr(watershed_workflow.hydro.hydrography, '_cutAndSnapInteriorCrossing', fail)
    report = _find(_rivers(ONE_STREAM), min_cell_area=1.)
    assert 'multiple-crossing' in [d.kind for d in report.defects]
    assert report.mesh is None
    assert any('mesh stage skipped' in n for n in report.notes)
