"""Tests for hydro.watershed.offsetDivideFromReaches: moving a HUC divide
away from reaches that run along (or across) it."""
import pytest
import geopandas
import shapely
import shapely.geometry

import watershed_workflow.hydro.watershed as ww
import watershed_workflow.sources.standard_names as names


def _hucs():
    """A = [0,100]x[0,100], B = [100,200]x[0,100], C = [0,200]x[100,150] on top of both."""
    polys = [shapely.geometry.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
             shapely.geometry.Polygon([(100, 0), (200, 0), (200, 100), (100, 100)]),
             shapely.geometry.Polygon([(0, 100), (100, 100), (200, 100), (200, 150), (0, 150)])]
    return geopandas.GeoDataFrame({names.ID: ['A', 'B', 'C']}, geometry=polys)


def _reaches(coords, **props):
    return geopandas.GeoDataFrame(props, geometry=[shapely.geometry.LineString(coords)])


# a reach of A running 2 m inside the A|B divide, with one excursion into B
ALONG_DIVIDE = [(98, 10), (98, 40), (101, 50), (98, 60), (98, 90)]


def test_divide_moves_off_the_reach():
    hucs = _hucs()
    reaches = _reaches(ALONG_DIVIDE)
    out, report = ww.offsetDivideFromReaches(hucs, reaches, 'A', 'B', 10.)
    a, b, c = out.geometry

    assert report['reach_crossings'] == 0
    assert report['min_reach_to_divide'] > 9.9
    assert a.area > hucs.geometry[0].area
    assert a.area + b.area == pytest.approx(20000.)
    assert all(isinstance(p, shapely.geometry.Polygon) and p.is_valid for p in (a, b))
    # the reach is now entirely in A
    assert a.buffer(1e-6).contains(reaches.geometry[0])


def test_neighbors_stay_coincident_and_domain_unchanged():
    hucs = _hucs()
    out, _ = ww.offsetDivideFromReaches(hucs, _reaches(ALONG_DIVIDE), 'A', 'B', 10.)
    a, b, c = out.geometry
    for p, q in ((a, b), (a, c), (b, c)):
        assert p.intersection(q).area < 1e-6
    assert shapely.unary_union([a, b, c]).area == pytest.approx(30000.)
    assert c.equals(hucs.geometry[2])                 # same shape (grid snapping may reorder vertices)


def test_clearance_from_a_function_of_the_reach():
    hucs = _hucs()
    reaches = _reaches(ALONG_DIVIDE, width=[4.])
    out, report = ww.offsetDivideFromReaches(hucs, reaches, 'A', 'B', lambda r: 2.5 * r['width'])
    assert report['min_reach_to_divide'] > 9.9


def test_untouched_when_reach_is_far_from_the_divide():
    hucs = _hucs()
    out, report = ww.offsetDivideFromReaches(hucs, _reaches([(50, 10), (50, 90)]), 'A', 'B', 10.)
    assert report['area_moved'] == pytest.approx(0.)
    assert out.geometry[1].equals(hucs.geometry[1])


def test_splitting_the_other_huc_raises():
    # a reach of A crossing all of B would cut B in two large pieces
    with pytest.raises(RuntimeError):
        ww.offsetDivideFromReaches(_hucs(), _reaches([(90, 50), (210, 50)]), 'A', 'B', 10.)
