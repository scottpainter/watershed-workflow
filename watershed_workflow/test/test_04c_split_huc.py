"""Tests for hydro.watershed.splitHUC: splitting a HUC along a cut."""
import pytest
import geopandas
import shapely
import shapely.geometry

import watershed_workflow
import watershed_workflow.hydro.watershed as ww
import watershed_workflow.sources.standard_names as names


def _hucs():
    """A = [0,2000]x[0,1000] above B = [0,2000]x[-1000,0]."""
    polys = [shapely.geometry.box(0, 0, 2000, 1000), shapely.geometry.box(0, -1000, 2000, 0)]
    return geopandas.GeoDataFrame({names.ID: ['A', 'B'], 'huc12': ['A', 'B'], 'tohuc': ['B', 'OUT'],
                                   'name': ['Upper', 'Lower'], 'areasqkm': [2., 2.], 'tnmid': ['x', 'y']},
                                  geometry=polys)


CUT = [(1000, 0), (1100, 500), (1000, 1000)]


def test_split_into_a_new_huc():
    out, report = ww.splitHUC(_hucs(), 'A', CUT, (1500, 500), new_id='Aa', new_tohuc='B', name_suffix=' (east)')
    assert list(out[names.ID]) == ['A', 'B', 'Aa']
    a, b, aa = out.geometry
    assert aa.area == pytest.approx(report['piece_area']) == pytest.approx(0.95e6)
    assert a.area + aa.area == pytest.approx(2.e6)
    new = out.iloc[2]
    assert (new['huc12'], new['tohuc'], new['name']) == ('Aa', 'B', 'Upper (east)')
    assert new['areasqkm'] == pytest.approx(aa.area * 1.e-6) and out.iloc[0]['areasqkm'] == pytest.approx(a.area * 1.e-6)
    assert new['tnmid'] is None or new['tnmid'] != new['tnmid']
    assert report['cut_length'] == pytest.approx(2 * 509.902, abs=0.01)


def test_cut_end_on_a_shared_edge_is_noded_in_the_neighbor():
    out, _ = ww.splitHUC(_hucs(), 'A', CUT, (1500, 500), new_id='Aa')
    a, b, aa = out.geometry
    assert (1000., 0.) in set(b.exterior.coords)
    for p, q in ((a, b), (aa, b), (a, aa)):
        assert p.intersection(q).area == 0.
    watershed_workflow.Watershed(out)


def test_cut_ends_are_moved_onto_the_boundary():
    out, _ = ww.splitHUC(_hucs(), 'A', [(1000, 3), (1100, 500), (1000, 996)], (1500, 500), new_id='Aa')
    assert (1000., 0.) in set(out.geometry[0].exterior.coords)
    assert (1000., 1000.) in set(out.geometry[2].exterior.coords)


def test_merge_into_a_neighbor():
    hucs = _hucs()
    hucs.loc[1, 'geometry'] = shapely.geometry.box(1000, -1000, 2000, 0)      # B only under A's east half
    out, report = ww.splitHUC(hucs, 'A', [(1000, 0), (1000, 1000)], (1500, 500), merge_into='B')
    assert len(out) == 2
    assert out.geometry[1].area == pytest.approx(2.e6) == report['merged_area']
    assert out.geometry[0].area == pytest.approx(1.e6)


def test_errors():
    with pytest.raises(ValueError):
        ww.splitHUC(_hucs(), 'A', CUT, (1500, 500))
    with pytest.raises(RuntimeError):      # a cut that ends inside A
        ww.splitHUC(_hucs(), 'A', [(1000, 0), (1000, 400)], (1500, 500), new_id='Aa')
    with pytest.raises(RuntimeError):      # a cut that crosses A twice
        ww.splitHUC(_hucs(), 'A', [(500, 0), (1000, 1100), (1500, 0)], (1500, 500), new_id='Aa', end_tol=200)
    hucs = _hucs()
    hucs.loc[1, 'geometry'] = shapely.geometry.box(0, -1000, 900, 0)
    with pytest.raises(RuntimeError):      # piece does not touch B
        ww.splitHUC(hucs, 'A', [(1000, 0), (1000, 1000)], (1500, 500), merge_into='B')
