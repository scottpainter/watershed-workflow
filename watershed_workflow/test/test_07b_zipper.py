"""Tests for angles._zipperSiblings: moving a sharp confluence upstream."""
import math

import pytest
import numpy as np
import geopandas
import shapely.geometry

import watershed_workflow.hydro.river
import watershed_workflow.hydro.angles as angles
import watershed_workflow.sources.standard_names as names


def _branch(deg, n=10, ds=10.):
    """Straight branch arriving at the origin from direction deg (from +y), upstream first."""
    t = math.radians(deg)
    return [(r * math.sin(t), r * math.cos(t)) for r in np.arange(n, 0, -1) * ds] + [(0., 0.)]


def _river(side_coords=None):
    """Parent P below the origin; siblings M (main: P's uphydroseq, larger
    drainage) at +5 degrees and S (side) at -5 degrees. S is listed first."""
    side = _branch(-5.) if side_coords is None else side_coords
    df = geopandas.GeoDataFrame({
        names.ID: ['P', 'S', 'M'],
        names.HYDROSEQ: [1., 3., 2.],
        names.DOWNSTREAM_HYDROSEQ: [0., 1., 1.],
        names.UPSTREAM_HYDROSEQ: [2., 0., 0.],
        names.DRAINAGE_AREA: [100., 20., 80.],
        names.ORDER: [3, 2, 2],
        'geometry': [shapely.geometry.LineString([(0., 0.), (0., -50.), (0., -100.)]),
                     shapely.geometry.LineString(side),
                     shapely.geometry.LineString(_branch(5.))]})
    return watershed_workflow.hydro.river.createRivers(df, method='hydroseq')[0]


def _sibling_angle(reach):
    return min(angles._getAngles([c.linestring for c in reach.children]))


def test_zipper_opens_confluence_to_min_angle():
    river = _river()
    assert _sibling_angle(river) == pytest.approx(10., abs=0.01)
    assert angles._zipperSiblings(list(river.children), 25.)

    # one step only reaches ~19.8 degrees; two are needed to reach ~29.3
    merged = river.children[0]
    assert len(river.children) == 1
    assert len(merged.linestring.coords) == 3
    assert _sibling_angle(merged) >= 25.
    assert river.isContinuous()


def test_zipper_merged_reach_takes_main_branch_identity():
    river = _river()
    angles._zipperSiblings(list(river.children), 25.)
    merged = river.children[0]

    assert merged[names.ID].startswith('M')
    assert merged[names.DRAINAGE_AREA] == pytest.approx(100.)
    assert merged[names.ORDER] == 3                  # two order-2 branches meet
    assert river[names.HYDROSEQ] < merged[names.HYDROSEQ] < min(c[names.HYDROSEQ] for c in merged.children)
    assert river[names.UPSTREAM_HYDROSEQ] == merged[names.HYDROSEQ]
    assert all(c[names.DOWNSTREAM_HYDROSEQ] == merged[names.HYDROSEQ] for c in merged.children)
    assert river.isHydroseqConsistent()


def test_zipper_without_points_to_spare_changes_nothing():
    # a side branch with a single segment has nothing to give
    river = _river(side_coords=_branch(-5., n=1))
    before = [(n[names.ID], list(n.linestring.coords)) for n in river]
    assert not angles._zipperSiblings(list(river.children), 25.)
    assert [(n[names.ID], list(n.linestring.coords)) for n in river] == before


def test_zipper_already_open_changes_nothing():
    river = _river()
    assert not angles._zipperSiblings(list(river.children), 5.)
    assert len(river.children) == 2
