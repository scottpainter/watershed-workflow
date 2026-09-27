"""Tests for findHUCOutlets / updateToHUCs: each HUC's outlet and
downstream HUC (tohuc) derived from the geometry."""
import pytest
import pandas
import geopandas
import shapely.geometry

import watershed_workflow
import watershed_workflow.hydro.watershed
import watershed_workflow.hydro.river
import watershed_workflow.hydro.hydrography as hydro
import watershed_workflow.sources.standard_names as names


def _hucs(tohuc):
    """Two boxes: A = [0,10]x[-5,5] drains to B = [10,20]x[-5,5], which drains out at x=20."""
    polys = [shapely.geometry.Polygon([(0, -5), (10, -5), (10, 5), (0, 5)]),
             shapely.geometry.Polygon([(10, -5), (20, -5), (20, 5), (10, 5)])]
    df = geopandas.GeoDataFrame({names.ID: ['A', 'B'], 'tohuc': tohuc, 'geometry': polys})
    return watershed_workflow.hydro.watershed.Watershed(df)


def _rivers(lines):
    df = geopandas.GeoDataFrame({names.ID: [str(i) for i in range(len(lines))],
                                 'geometry': [shapely.geometry.LineString(l) for l in lines]})
    return watershed_workflow.hydro.river.createRivers(df, method='geometry')


# A single stream, already cut at the A|B divide (as after cutAndSnapCrossings)
ONE_OUTLET = [[(10, 0), (20, 0)], [(2, 0), (10, 0)]]


def test_consistent_tohuc_unchanged():
    hucs = _hucs(['B', 'OUTSIDE'])
    hydro.updateToHUCs(hucs, _rivers(ONE_OUTLET))
    assert list(hucs.df['tohuc']) == ['B', 'OUTSIDE']
    assert list(hucs.df['tohuc_wbd']) == ['B', 'OUTSIDE']
    assert hucs.df[names.OUTLET][0].equals(shapely.geometry.Point(10, 0))
    assert hucs.df[names.OUTLET][1].equals(shapely.geometry.Point(20, 0))
    assert hucs.exterior_outlet.equals(shapely.geometry.Point(20, 0))


def test_inconsistent_tohuc_corrected():
    # attribute says A drains out of the domain, but the geometry says A -> B
    hucs = _hucs(['ELSEWHERE', 'OUTSIDE'])
    hydro.updateToHUCs(hucs, _rivers(ONE_OUTLET))
    assert list(hucs.df['tohuc']) == ['B', 'OUTSIDE']
    assert list(hucs.df['tohuc_wbd']) == ['ELSEWHERE', 'OUTSIDE']


def test_domain_outlet_naming_internal_huc_is_cleared():
    # B leaves the domain, so it cannot drain into A
    hucs = _hucs(['B', 'A'])
    hydro.updateToHUCs(hucs, _rivers(ONE_OUTLET))
    assert hucs.df['tohuc'][0] == 'B'
    assert pandas.isna(hucs.df['tohuc'][1])


def test_branches_meeting_on_divide_are_one_outlet():
    # two branches from A meet exactly on the divide at (10, 0)
    lines = [[(10, 0), (20, 0)], [(2, 3), (10, 0)], [(2, -3), (10, 0)]]
    hucs = _hucs(['B', 'OUTSIDE'])
    outlets = hydro.findHUCOutlets(hucs, _rivers(lines))
    assert len(outlets[0]) == 1
    assert len(outlets[0][0].reaches) == 2
    assert outlets[0][0].downstream == 1
    hydro.updateToHUCs(hucs, _rivers(lines))
    assert list(hucs.df['tohuc']) == ['B', 'OUTSIDE']


def test_two_outlets_raise():
    # A drains into B at two separate places, (10, 3) and (10, -3)
    lines = [[(15, 0), (20, 0)],
             [(10, 3), (15, 0)], [(10, -3), (15, 0)],
             [(2, 3), (10, 3)], [(2, -3), (10, -3)]]
    hucs = _hucs(['B', 'OUTSIDE'])
    with pytest.raises(hydro.MultipleOutletsError) as err:
        hydro.updateToHUCs(hucs, _rivers(lines))
    assert list(err.value.outlets.keys()) == ['A']
    assert len(err.value.outlets['A']) == 2
    assert 'HUC A' in str(err.value)
    # nothing was modified
    assert 'tohuc_wbd' not in hucs.df.columns


def test_simplify_updates_tohuc():
    # uncut stream crossing the divide: simplify() cuts it, then updates tohuc
    polys = [shapely.geometry.Polygon([(0, -500), (1000, -500), (1000, 500), (0, 500)]),
             shapely.geometry.Polygon([(1000, -500), (2000, -500), (2000, 500), (1000, 500)])]
    hucs = watershed_workflow.hydro.watershed.Watershed(
        geopandas.GeoDataFrame({names.ID: ['A', 'B'], 'tohuc': ['ELSEWHERE', 'OUTSIDE'],
                                'geometry': polys}))
    rivers = _rivers([[(200, 10), (800, -10), (1400, 10), (1990, 0)]])
    watershed_workflow.simplify(hucs, rivers, 100.0)
    assert list(hucs.df['tohuc']) == ['B', 'OUTSIDE']
    assert list(hucs.df['tohuc_wbd']) == ['ELSEWHERE', 'OUTSIDE']
