"""Tests for river_mesh.computeWidthByOrder / widthByOrderFunction."""
import pytest
import geopandas
import shapely.geometry

import watershed_workflow.hydro.river
import watershed_workflow.sources.standard_names as names
from watershed_workflow.mesh.river_mesh import computeWidthByOrder, widthByOrderFunction


def _reaches():
    lines = [[(1, 0), (0, 0)], [(2, 1), (1, 0)], [(2, -1), (1, 0)]]
    return geopandas.GeoDataFrame({names.ORDER: [2, 1, 1],
                                   names.BANKFULL_WIDTH: [10., 3., 5.]},
                                  geometry=[shapely.geometry.LineString(l) for l in lines])


def test_width_by_order_from_dataframe():
    assert computeWidthByOrder(_reaches()) == {1: 4., 2: 10.}
    assert computeWidthByOrder(_reaches(), statistic='max') == {1: 5., 2: 10.}


def test_width_by_order_from_rivers():
    rivers = watershed_workflow.hydro.river.createRivers(_reaches(), method='geometry')
    assert computeWidthByOrder(rivers) == {1: 4., 2: 10.}


def test_width_function_uses_nearest_order():
    width = widthByOrderFunction({1: 4., 2: 10., 4: 30.})
    assert width({names.ORDER: 1}) == 4.
    assert width({names.ORDER: 4}) == 30.
    assert width({names.ORDER: 6}) == 30.      # above the table
    assert width({names.ORDER: 3}) in (10., 30.)  # between two orders: either neighbor
