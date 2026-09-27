"""Tests for snapHUCsToCorridors: HUC boundaries running alongside a river
corridor are made to follow the corridor's own vertices."""
import pytest
import numpy as np
import geopandas
import shapely.geometry

import watershed_workflow.hydro.watershed
from watershed_workflow.mesh.river_mesh import snapHUCsToCorridors


def _two_boxes(divide_ys):
    """A = [0,100]x[0,100], B = [100,200]x[0,100]; the shared divide x=100
    has interior vertices at divide_ys."""
    divide = [(100., 0.)] + [(100., y) for y in divide_ys] + [(100., 100.)]
    a = shapely.geometry.Polygon([(0., 0.)] + divide + [(0., 100.)])
    b = shapely.geometry.Polygon([(100., 0.), (200., 0.), (200., 100.), (100., 100.)]
                                 + [(100., y) for y in reversed(divide_ys)])
    return watershed_workflow.hydro.watershed.Watershed(geopandas.GeoDataFrame(geometry=[a, b]))


def _corridor():
    """A 4 m wide corridor centered on x=104, from y=10 to y=90, vertices every 20 m."""
    ys = [10., 30., 50., 70., 90.]
    return shapely.geometry.Polygon([(102., y) for y in ys] + [(106., y) for y in reversed(ys)])


def _divide(hucs):
    handles = [h for spine in hucs.intersections.values() for h in spine.values()]
    assert len(handles) == 1
    return handles[0]


def _coords(ls):
    return sorted((round(x, 6), round(y, 6)) for x, y in ls.coords)


def test_vertices_snap_to_corridor():
    hucs = _two_boxes([25., 50., 75.])
    h = _divide(hucs)
    report = snapHUCsToCorridors(hucs, [_corridor(), ], tol=5.)
    assert report == [(h, 3, 0)]
    # endpoints unchanged, interior vertices moved to the nearest corridor vertex
    assert _coords(hucs.linestrings[h]) == sorted([(100., 0.), (102., 30.), (102., 50.),
                                                   (102., 70.), (100., 100.)])


def test_corridor_vertices_inserted_between_snaps():
    hucs = _two_boxes([15., 85.])
    h = _divide(hucs)
    report = snapHUCsToCorridors(hucs, [_corridor(), ], tol=5.)
    assert report == [(h, 2, 3)]
    assert _coords(hucs.linestrings[h]) == sorted([(100., 0.), (102., 10.), (102., 30.), (102., 50.),
                                                   (102., 70.), (102., 90.), (100., 100.)])


def test_far_boundaries_unchanged():
    hucs = _two_boxes([25., 50., 75.])
    before = {h: list(ls.coords) for h, ls in hucs.linestrings.items()}
    assert snapHUCsToCorridors(hucs, [_corridor(), ], tol=1.) == []
    assert {h: list(ls.coords) for h, ls in hucs.linestrings.items()} == before


def test_snapped_boundary_stays_outside_corridor_interior():
    hucs = _two_boxes([15., 85.])
    snapHUCsToCorridors(hucs, [_corridor(), ], tol=5.)
    interior = _corridor().buffer(-1.e-6)
    for ls in hucs.linestrings.values():
        assert ls.intersection(interior).length < 1.e-9
