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


def test_exterior_boundary_never_snapped():
    # one HUC whose exterior edge x=110 runs 2 m from a corridor at x in [104, 108]
    edge = [(110., y) for y in (0., 25., 50., 75., 100.)]
    poly = shapely.geometry.Polygon([(0., 0.)] + edge + [(0., 100.)])
    hucs = watershed_workflow.hydro.watershed.Watershed(geopandas.GeoDataFrame(geometry=[poly]))
    ys = [10., 30., 50., 70., 90.]
    corridor = shapely.geometry.Polygon([(104., y) for y in ys] + [(108., y) for y in reversed(ys)])
    before = {h: list(ls.coords) for h, ls in hucs.linestrings.items()}
    assert snapHUCsToCorridors(hucs, [corridor, ], tol=5.) == []
    assert {h: list(ls.coords) for h, ls in hucs.linestrings.items()} == before


#
# snapCorridorsToExterior / matchRiverVertices
#
from watershed_workflow.mesh.river_mesh import snapCorridorsToExterior, matchRiverVertices


def _box_domain():
    """One HUC [0,100]x[0,100]; its exterior boundary is the only linestring."""
    poly = shapely.geometry.Polygon([(0., 0.), (100., 0.), (100., 100.), (0., 100.)])
    return watershed_workflow.hydro.watershed.Watershed(geopandas.GeoDataFrame(geometry=[poly]))


def test_corridor_vertex_near_exterior_moves_onto_it():
    hucs = _box_domain()
    # corridor leaving through the bottom edge; its vertex (30, 2) is 2 m inside
    coords = np.array([[40., 0.], [30., 2.], [30., 40.], [38., 40.]])
    corridors = [shapely.geometry.Polygon(coords)]
    corridors, moves = snapCorridorsToExterior(hucs, coords, corridors, tol=5.)

    assert len(moves) == 1 and moves[0]['index'] == 1
    assert np.allclose(coords[1], (30., 0.))                       # projected onto the edge
    assert np.allclose(coords[0], (40., 0.))                       # already on the boundary: unchanged
    assert np.allclose(coords[2:], [[30., 40.], [38., 40.]])       # far from the boundary: unchanged
    assert (30., 0.) in [tuple(c) for c in corridors[0].exterior.coords]
    boundary = shapely.ops.unary_union(list(hucs.linestrings.values()))
    assert any(np.allclose(c[0:2], (30., 0.)) for ls in hucs.linestrings.values() for c in ls.coords)
    assert boundary.length == pytest.approx(400.)                  # the domain boundary did not move


def test_corridor_vertex_snaps_to_nearby_boundary_vertex():
    hucs = _box_domain()
    # (99.4, 0.5) projects to (99.4, 0), which is within 1 m of the existing
    # boundary vertex (100, 0), so the corridor vertex goes onto that vertex
    coords = np.array([[99.4, 0.5], [60., 20.], [60., 40.]])
    snapCorridorsToExterior(hucs, coords, [shapely.geometry.Polygon(coords)], tol=5.)
    assert np.allclose(coords[0], (100., 0.))


def test_match_river_vertices_handles_dropped_and_reordered_vertices():
    river_coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    elems = [[0, 1, 2, 3]]
    # the triangulation dropped (0, 0) and put the others in a different order
    tri_coords = np.array([[5., 5.], [1., 1.], [0., 1.], [1., 0.]])
    coords, new_elems, n = matchRiverVertices(tri_coords, river_coords, elems, [])
    assert n == 1
    assert len(coords) == 5
    assert np.allclose(coords[new_elems[0]], river_coords)
