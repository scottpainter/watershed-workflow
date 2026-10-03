"""creates river mesh using quad, pentagon and hexagon elements"""
from typing import Callable, List, Tuple, Dict, Optional

import numpy as np
import scipy.spatial
import pandas as pd
import logging

from matplotlib import pyplot as plt
import matplotlib.axes

import geopandas as gpd

import shapely.geometry
import shapely.ops

import watershed_workflow.utils.geometry
import watershed_workflow.utils.tinytree
import watershed_workflow.hydro.angles
import watershed_workflow.plot.plot
from watershed_workflow.hydro.river import River
from watershed_workflow.hydro.watershed import Watershed
import watershed_workflow.sources.standard_names as names

__all__ = [
    'createRiversMesh',
    'computeWidthByOrder',
    'widthByOrderFunction',
    'snapHUCsToCorridors',
    'snapCorridorsToExterior',
    'matchRiverVertices',
]


def _isNonoverlapping(points: np.ndarray, elems: List[List[int]], tol: float = 1) -> bool:
    """Check if a set of polygon shapes are nonoverlapping.

    Parameters
    ----------
    points : np.ndarray
        Array of coordinate points.
    elems : List[List[int]]
        List of element connectivity, each element is a list of point indices.
    tol : float, optional
        Tolerance for area comparison, by default 1.

    Returns
    -------
    bool
        True if shapes are nonoverlapping within tolerance.
    """
    shps = [shapely.geometry.Polygon(points[e]) for e in elems]
    total_area = shapely.unary_union(shps).area
    summed_area = sum(shp.area for shp in shps)
    logging.info(f'  is nonoverlapping?  total_area = {total_area}, summed_area = {summed_area}')
    return abs(total_area - summed_area) < tol


def _computeExpectedNumCoords(river: River) -> int:
    """Compute the number of expected coordinates for river mesh.

    Parameters
    ----------
    river : River
        River network to compute coordinates for.

    Returns
    -------
    int
        Expected number of coordinates in the river mesh.
    """
    # two outlet points
    n = 2

    # internal points
    n += sum(2 * (len(reach.linestring.coords) - 2) for reach in river)

    # endpoints
    n += sum(len(reach.children) + 1 for reach in river)
    return n


def _computeExpectedNumElems(river: River) -> int:
    """Compute the number of expected elements for river mesh.

    Parameters
    ----------
    river : River
        River network to compute elements for.

    Returns
    -------
    int
        Expected number of elements in the river mesh.
    """
    return sum(len(reach.linestring.coords) - 1 for reach in river)


def _plotRiver(river: River,
               coords: np.ndarray,
               ax: matplotlib.axes.Axes,
               intersections : Optional[gpd.GeoDataFrame] = None) -> None:
    """Plot the river and elements for a debugging plot.

    Parameters
    ----------
    river : River
        River network to plot.
    coords : np.ndarray
        Array of mesh coordinates.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """
    elems = gpd.GeoDataFrame(geometry=[
        shapely.geometry.Polygon(coords[elem]) for reach in river for elem in reach[names.ELEMS]
    ], crs=river.df.crs)


    if intersections is not None:
        int_mp = intersections.buffer(2000).union_all()
        river.df[river.df.intersects(int_mp)].plot(ax=ax, color='b')#, marker='+')

        elems = elems[elems.intersects(int_mp)]
        logging.info(f' ... plotting {len(elems)} elements near {len(intersections)} intersections')
    else:
        river.df.plot(ax=ax, color='b')

    elems.boundary.plot(ax=ax, color='g')#, marker='x')
    # watershed_workflow.plot.plot.linestringsWithCoords(elems.boundary, color='g', marker='x', ax=ax)


def computeWidthByOrder(rivers : gpd.GeoDataFrame | List[River],
                        order_col : str = names.ORDER,
                        width_col : str = names.BANKFULL_WIDTH,
                        statistic : str = 'mean') -> Dict[int, float]:
    """Corridor width for each stream order, from the reaches' own channel widths.

    Summarizes width_col (bankfull width by default) over all reaches of each
    stream order, so that a width-by-order corridor is consistent with the
    channel data rather than with a hand-chosen table.  Use with
    widthByOrderFunction() to build the river_width argument of
    tessalateRiverAligned().

    Parameters
    ----------
    rivers : gpd.GeoDataFrame or List[River]
        Reaches with order_col and width_col properties.
    order_col, width_col : str, optional
        Property names of the stream order and width.
    statistic : str, optional
        A pandas groupby aggregation, e.g. 'mean' (default) or 'median'.

    Returns
    -------
    Dict[int, float]
        Width for each stream order present.
    """
    if isinstance(rivers, gpd.GeoDataFrame):
        df = rivers
    else:
        df = pd.DataFrame([{order_col : reach[order_col], width_col : reach[width_col]}
                           for river in rivers for reach in river])
    widths = df.groupby(order_col)[width_col].agg(statistic)
    return { int(order) : float(width) for order, width in widths.items() }


def widthByOrderFunction(width_by_order : Dict[int, float],
                         order_col : str = names.ORDER) -> Callable[[River], float]:
    """A river_width callable for createRiversMesh() / tessalateRiverAligned().

    Returns width_by_order[reach[order_col]]; a reach whose order is not in
    the table gets the width of the nearest order that is.
    """
    orders = sorted(width_by_order.keys())

    def riverWidth(reach : River) -> float:
        order = reach[order_col]
        if order in width_by_order:
            return width_by_order[order]
        return width_by_order[min(orders, key=lambda o: abs(o - order))]
    return riverWidth


def createRiversMesh(hucs : Watershed,
                     rivers : List[River],
                     computeWidth : Callable[[River], float],
                     ax : Optional[matplotlib.axes.Axes] = None,
                     plot : bool = False,
                     ) -> \
                     Tuple[np.ndarray,
                           List[List[int]],
                           List[shapely.geometry.Polygon],
                           List[shapely.geometry.Point],
                           gpd.GeoDataFrame | None,
                           ]:
    """Create meshes for each river and merge them.

    Parameters
    ----------
    hucs : Watershed
        Split HUCs object for mesh adjustment.
    rivers : List[River]
        List of river networks to mesh.
    computeWidth : Callable[[River], float]
        Function to compute the river width for each reach (given as a River object).
        This callable can either return a constant value, or dynamically fetch a value 
        based on stream order, properties, or a user-defined rule.
    ax : matplotlib.axes.Axes, optional
        Axes for debugging plots, by default None.

    Returns
    -------
    Tuple[np.ndarray, List[List[int]], List[shapely.geometry.Polygon], List[shapely.geometry.Point], gpd.GeoDataFrame | None]
        Tuple containing coordinates, elements, corridors, hole points, and intersections dataframe.
    """
    elems: List[List[int]] = []
    coords: List[np.ndarray] = []
    corridors: List[shapely.geometry.Polygon] = []
    hole_points: List[shapely.geometry.Point] = []
    coords_gid_start = 0
    elems_gid_start = 0

    for i,river in enumerate(rivers):
        # create the mesh
        lcoords, lelems = createRiverMesh(river, computeWidth, elems_gid_start)
        logging.info(f' ... created a mesh with {len(lelems)} elements for river {i}')
        elems_gid_start += len(lelems)

        # adjust the HUC linestrings to include the small cross-stream
        # segment
        adjustHUCsToRiverMesh(hucs, river, lcoords)

        if ax is not None and plot:
            logging.info('Plotting the river mesh')
            _plotRiver(river, lcoords, ax)

        # hole point is the centroid of the outlet element
        hole_points.append(lcoords[lelems[-1]].mean(axis=0))

        # corridor is the trace of the outside
        corridors.append(shapely.geometry.Polygon(lcoords))

        # shift to get a global ordering
        if coords_gid_start != 0:
            lelems = [[j + coords_gid_start for j in e] for e in lelems]
        for reach in river:
            reach[names.ELEMS][:] = [[j + coords_gid_start for j in e] for e in reach[names.ELEMS]]
        coords_gid_start += len(lcoords)

        elems.extend(lelems)
        coords.append(lcoords)

    if ax is not None:
        #hucs.plotAsLinestrings(color='k', marker='x', ax=ax)
        hucs.plotAsLinestrings(color='k', ax=ax)

    all_coords = np.concatenate(coords)

    if not _isNonoverlapping(all_coords, elems):
        logging.warning(
            f'Found at least one intersection overlapping elements in the river mesh... searching for the first intersection now'
        )

        # find overlaps
        # -- create reach polygons
        reach_polys = [
            shapely.unary_union([
                shapely.geometry.Polygon([all_coords[e] for e in elem])
                for elem in reach[names.ELEMS]
            ]) for river in rivers for reach in river
        ]
        reach_ids = [reach[names.ID] for river in rivers for reach in river]

        # -- find pairwise intersections
        intersection_i = []
        intersection_j = []
        intersection_p = []
        for i in range(0, len(reach_polys) - 1):
            for j in range(i + 1, len(reach_polys)):
                if reach_polys[i].intersection(reach_polys[j]).area > 0:
                    intersection_i.append(reach_ids[i])
                    intersection_j.append(reach_ids[j])
                    intersection_p.append(reach_polys[i].intersection(reach_polys[j]))

        intersections_df = gpd.GeoDataFrame(data={
            'i': intersection_i,
            'j': intersection_j,
        },
                                            geometry=intersection_p,
                                            crs=hucs.crs)
        logging.info(f' ... found {len(intersections_df)} intersections:')
        logging.info(f'  at: {intersections_df.centroid}')
        if ax is not None:
            _plotRiver(river, lcoords, ax, intersections_df)
            intersections_df.plot(color='r', marker='x', ax=ax)

    else:
        intersections_df = None

    return all_coords, elems, corridors, hole_points, intersections_df


def createRiverMesh(river: River,
                    computeWidth: Callable[[River, ], float],
                    elems_gid_start: int = 0,
                    check_convexity: bool = True):
    """Returns list of elems and river corridor polygons for a given list of river trees

    Parameters
    ----------
    river : River
        River tree along which river mesh is to be created.
    computeWidth : Callable[[River, ], float]
        Function that computes the width for a given reach.
    elems_gid_start : int, optional
        Starting global ID for elements, by default 0.
    check_convexity : bool, optional
        If True, check each element for convexity and attempt to fix
        non-convex tip elements after projection.  Set to False to skip
        this pass entirely, which is useful when deliberately testing
        degenerate width configurations that would otherwise raise inside
        the convexity fixer.  Default is True.

    Returns
    -------
    corrs: list(shapely.geometry.Polygon)
        List of river corridor polygons, one per river, storing the
        coordinates used in elems.
    elems: list(list)
        List of river elements, each element a list of indices into
        corr.coords.

    """
    coords = np.nan * np.ones((_computeExpectedNumCoords(river), 2), 'd')
    river.df[names.ELEMS] = pd.Series([[list() for i in range(len(ls.coords) - 1)]
                                       for ls in river.df.geometry],
                                      index=river.df.index)

    # project the starting point
    # k tracks the index of the point/coordinate
    k = 0

    debug: Tuple[int | None, int | None] = None, None  # reach index, coordinate index
    if debug[0] != None and debug[1] != None:
        node = river.getNode(debug[0])
        if node is not None:
            logging.info(
                f"Debugging reach {debug[0]}, coordinate {debug[1]}, at {node.linestring.coords[debug[1]]}"
            )

    for touch, reach in river.prePostInBetweenOrder():
        halfwidth = computeWidth(reach) / 2.
        reach_elems = reach[names.ELEMS]

        if debug[0] == reach.index:
            logging.info(f'PRE: reach = {reach.index}, touch = {touch}, elems = {reach_elems}')

        if touch == 0:
            # add paddler's right downstream point
            if reach.parent is None:
                # A simple projection orthogonal to the downstream segment
                # TODO -- follow the HUC boundary?
                coords[k] = projectOne(reach.linestring.coords[-2], reach.linestring.coords[-1],
                                       halfwidth)
                if reach.index == debug[0] and (-1 == debug[1]
                                                or len(reach.linestring.coords) - 1 == debug[1]):
                    logging.info(
                        f" -- adding coord {k} = {coords[k]} as {reach.index} outlet right")
                reach_elems[-1].append(k)
                k += 1

            # add paddler's right internal points by two-touches projection
            for i in reversed(range(1, len(reach.linestring.coords) - 1)):
                coords[k] = projectTwoClampedMiter(reach.linestring.coords[i - 1], reach.linestring.coords[i],
                                       reach.linestring.coords[i + 1], halfwidth, halfwidth,
                                       reach.index == debug[0] and i == debug[1])
                if reach.index == debug[0] and i == debug[1]:
                    logging.info(
                        f" -- adding coord {k} = {coords[k]} as {reach.index} internal right")
                reach_elems[i].append(k)
                reach_elems[i - 1].append(k)
                k += 1

            # add the upstream point
            if len(reach.children) == 0:
                # add an upstream triangle tip at stream midpoint
                coords[k] = reach.linestring.coords[0]
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} leaf tip")
                reach_elems[0].append(k)
                k += 1

            elif len(reach.children) == 1:
                # add an upstream, paddler's right point based on inline junction of two reaches
                child_halfwidth = computeWidth(reach.children[0]) / 2.
                coords[k] = projectTwoClampedMiter(reach.children[0].linestring.coords[-2],
                                       reach.linestring.coords[0], reach.linestring.coords[1],
                                       child_halfwidth, halfwidth, reach.index == debug[0]
                                       and 0 == debug[1])
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(
                        f" -- adding coord {k} = {coords[k]} as {reach.index} inline child upstream right"
                    )
                reach_elems[0].append(k)
                child_elems = reach.children[0][names.ELEMS]
                child_elems[-1].append(k)
                k += 1

            else:
                # add an upstream, paddler's right point based on junction of multiple reaches
                coords[k] = projectJunction(reach, touch, computeWidth, reach.index == debug[0]
                                            and 0 == debug[1])
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(
                        f" -- adding coord {k} = {coords[k]} as {reach.index} junction child upstream right"
                    )
                reach_elems[0].append(k)
                reach.children[0][names.ELEMS][-1].append(k)
                k += 1

        if touch == len(reach.children):
            if len(reach.children) == 0:
                pass  # no second point

            elif len(reach.children) == 1:
                # add an upstream, paddler's left pont based on inline junction of two reaches
                child_halfwidth = computeWidth(reach.children[-1]) / 2.
                coords[k] = projectTwoClampedMiter(reach.linestring.coords[1], reach.linestring.coords[0],
                                       reach.children[-1].linestring.coords[-2], halfwidth,
                                       child_halfwidth, reach.index == debug[0] and 0 == debug[1])
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(
                        f" -- adding coord {k} = {coords[k]} as {reach.index} inline child upstream left"
                    )
                reach_elems[0].append(k)
                reach.children[-1][names.ELEMS][-1].append(k)
                k += 1

            else:
                # add an upstream, paddler's left point based on junction of multiple reaches
                coords[k] = projectJunction(reach, touch, computeWidth, reach.index == debug[0]
                                            and 0 == debug[1])
                if reach.index == debug[0] and 0 == debug[1]:
                    logging.info(
                        f" -- adding coord {k} = {coords[k]} as {reach.index} junction child upstream left"
                    )
                reach_elems[0].append(k)
                reach.children[-1][names.ELEMS][-1].append(k)
                k += 1

            # add a paddler's left internal point
            for i in range(1, len(reach.linestring.coords) - 1):
                coords[k] = projectTwoClampedMiter(reach.linestring.coords[i + 1], reach.linestring.coords[i],
                                       reach.linestring.coords[i - 1], halfwidth, halfwidth,
                                       (reach.index == debug[0] and i == debug[1]))
                if reach.index == debug[0] and i == debug[1]:
                    logging.info(
                        f" -- adding coord {k} = {coords[k]} as {reach.index} internal left")
                reach_elems[i - 1].append(k)
                reach_elems[i].append(k)
                k += 1

            # add paddler's left downstream point
            if reach.parent is None:
                # A simple projection orthogonal to the downstream segment
                # TODO -- follow the HUC boundary?
                coords[k] = projectOne(reach.linestring.coords[-2], reach.linestring.coords[-1],
                                       -halfwidth)
                if reach.index == debug[0] and (-1 == debug[1]
                                                or len(reach.linestring.coords) - 1 == debug[1]):
                    logging.info(f" -- adding coord {k} = {coords[k]} as {reach.index} outlet left")
                reach_elems[-1].append(k)
                k += 1

        if touch != 0 and touch != len(reach.children):
            # add a mid-tributary junction point
            coords[k] = projectJunction(reach, touch, computeWidth, reach.index == debug[0]
                                        and 0 == debug[1])
            if reach.index == debug[0] and 0 == debug[1]:
                logging.info(
                    f" -- adding coord {k} = {coords[k]} as {reach.index} junction midpoint")
            reach_elems[0].append(k)
            reach.children[touch - 1][names.ELEMS][-1].append(k)
            reach.children[touch][names.ELEMS][-1].append(k)
            k += 1

        if debug[0] == reach.index:
            logging.info(f'POST: reach = {reach.index}, touch = {touch}, elems = {reach_elems}')

    assert k == len(coords)

    # clamp middle junction points so they cannot overshoot the flanking points
    # For a reach with 2+ children the first element is a pentagon (or wider).
    # The flanking points (paddler's right and left at the junction) are already
    # placed; the middle point(s) must not be farther from the junction centre p
    # than either flanking point, or they can break convexity.
    for reach in river:
        if len(reach.children) < 2:
            continue
        p = np.asarray(reach.linestring.coords[0])
        pentagon = reach[names.ELEMS][0]
        # pentagon order: [right_dn, right_up, mid_1, ..., left_up, left_dn]
        # flanking upstream points are index 1 (right) and index -2 (left)
        right_up = coords[pentagon[1]]
        left_up  = coords[pentagon[-2]]
        cap_d = min(np.linalg.norm(right_up - p), np.linalg.norm(left_up - p))
        # middle points are everything between index 1 and index -2 (exclusive)
        for idx in pentagon[2:-2]:
            m = coords[idx]
            d = np.linalg.norm(m - p)
            if d > cap_d:
                coords[idx] = p + cap_d * (m - p) / d

    # another pass to check for convexity
    if check_convexity:
        for reach in river:
            for k, elem in enumerate(reach[names.ELEMS]):
                e_coords = coords[elem]
                if not watershed_workflow.utils.geometry.isConvex(e_coords):
                    if k != 0:
                        fig, ax = plt.subplots(1, 1)

                        reaches = [reach, ]
                        if reach.parent is not None:
                            reaches.append(reach.parent)
                        reaches = reaches + list(reach.children)
                        for r in reaches:
                            ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'b-x')

                        for k2, e2 in enumerate(reach.parent[names.ELEMS]):
                            e_coords2 = coords[e2]
                            poly2 = shapely.geometry.Polygon(e_coords2)
                            ax.plot(poly2.exterior.xy[0], poly2.exterior.xy[1], '-x', color='purple')

                        for lcv_reach in reach.parent.children:
                            for k2, e2 in enumerate(lcv_reach[names.ELEMS]):
                                e_coords2 = coords[e2]
                                poly2 = shapely.geometry.Polygon(e_coords2)
                                ax.plot(poly2.exterior.xy[0], poly2.exterior.xy[1], '-x', color='grey')

                        poly = shapely.geometry.Polygon(e_coords)
                        ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'g-x')

                        ls = reach.linestring
                        ax.plot(ls.xy[0], ls.xy[1], 'r-x')
                        ax.set_aspect('equal', adjustable='box')
                        plt.show()
                        raise RuntimeError(f'Convexity in non-0th ({k})th element of reach {reach.index} with ID {reach[names.ID]}')

                    new_e_coords = fixConvexity(reach, e_coords, computeWidth)
                    for c_index, coord in zip(elem, new_e_coords):
                        coords[c_index] = coord

    # gather elems
    elems = [e for reach in river.postOrder() for e in reach[names.ELEMS]]
    assert len(elems) == _computeExpectedNumElems(river)

    # assign GID to each elem start
    # note this must be done in the same order as above elems
    if names.ELEMS_GID_START not in river.df.columns:
        river.df[names.ELEMS_GID_START] = -np.ones(len(river.df), 'i')

    for reach in river.postOrder():
        reach[names.ELEMS_GID_START] = elems_gid_start
        elems_gid_start += len(reach[names.ELEMS])

    return coords, elems


def adjustHUCsToRiverMesh(hucs: Watershed, river: River, coords: np.ndarray) -> None:
    """Adjust HUC segments that touch reach endpoints to match the corridor coordinates.

    Parameters
    ----------
    hucs : Watershed
        Split HUCs object to adjust.
    river : River
        River network with mesh coordinates.
    coords : np.ndarray
        Array of mesh coordinates.
    """

    # downstream the river outlet
    remerge, touches = watershed_workflow.hydro.angles._getOutletLinestrings(hucs, river)

    if len(touches) > 0:
        assert len(touches) == 3
        logging.info(f"Adjusting HUC to match reaches at outlet")
        # touches[1] is paddler's left
        left_old_coords = touches[1][1].coords
        left_new_coord = coords[river[names.ELEMS][-1][-1]]
        left_new_ls = shapely.geometry.LineString(left_old_coords[:-1] + [left_new_coord, ])

        right_old_coords = touches[2][1].coords
        right_new_coord = coords[river[names.ELEMS][-1][0]]
        right_new_ls = shapely.geometry.LineString(right_old_coords[:-1] + [right_new_coord, ])

        if remerge:
            new_ls = shapely.geometry.LineString(
                list(reversed(left_new_ls.coords)) + list(right_new_ls.coords[1:]))
            hucs.linestrings[touches[1][0]] = new_ls
        else:
            hucs.linestrings[touches[1][0]] = watershed_workflow.utils.geometry.reverseLineString(
                left_new_ls) if touches[1][2] else left_new_ls
            hucs.linestrings[touches[2][0]] = watershed_workflow.utils.geometry.reverseLineString(
                right_new_ls) if touches[2][2] else right_new_ls

    # adjust all upstream endpoints
    for reach in river:
        touches = watershed_workflow.hydro.angles._getUpstreamLinestrings(hucs, reach)
        if len(touches) > len(reach.children) + 1:
            logging.info(
                f"Adjusting HUC to match reaches at reach {reach.index} and coordinate {reach.linestring.coords[0]}"
            )
            # yes, there are junctions involved...
            point_i = 1
            touch_i = 1
            while touch_i < len(touches):
                if touches[touch_i][0] >= 0:
                    # make sure touches before and after are reaches
                    #
                    # This will fail if there are two successive HUC
                    # strings.  I'm not sure that should ever happen,
                    # but if it does, we would have to choose which
                    # point to put on the reach junction element
                    # coordinate, and wierd stuff would probably
                    # happen in triangulation anyway.
                    assert touches[touch_i-1][0] is None or touches[touch_i-1][0] < 0, \
                        f"Neighboring touch at reach {reach.index}, ID {reach[names.ID]} coords {reach.linestring.coords[0]} is wierd"

                    assert touches[(touch_i+1)%len(touches)][0] is None or touches[(touch_i+1)%len(touches)][0] < 0, \
                        f"Neighboring touch at reach {reach.index}, ID {reach[names.ID]} coords {reach.linestring.coords[0]} is wierd"

                    # it is a HUC, insert the point
                    new_coord = coords[reach[names.ELEMS][0][point_i]]
                    old_coords = touches[touch_i][1].coords
                    new_ls = shapely.geometry.LineString(old_coords[:-1] + [new_coord, ])
                    hucs.linestrings[touches[touch_i][0]] = new_ls
                else:
                    point_i += 1

                touch_i += 1


def snapHUCsToCorridors(hucs: Watershed,
                        corridors: List[shapely.geometry.Polygon],
                        tol: float,
                        max_gap_vertices: int = 10) -> List[Tuple[int, int, int]]:
    """Make HUC boundaries that run alongside a river corridor follow its edge.

    adjustHUCsToRiverMesh() moves the ends of HUC linestrings that touch a
    river onto corridor coordinates, but a HUC boundary that runs close to
    and roughly parallel with a corridor keeps its own, independently
    resampled vertices.  The thin gap between the two then fills with tiny
    triangles during triangulation, as the refinement tries to satisfy its
    minimum angle, and those small cells limit the simulation time step.

    Each interior vertex of a HUC linestring lying within tol of a corridor
    outline is moved onto the nearest vertex of that outline.  Between two
    consecutive snapped vertices on the same outline, the outline vertices
    in between (the shorter way round, if there are at most
    max_gap_vertices) are inserted, so that the boundary follows the
    corridor edge exactly.  Only existing corridor vertices are used, so no
    vertex is placed part-way along a quad edge.  Linestring endpoints (HUC
    junctions, including those adjustHUCsToRiverMesh() placed on corridor
    corners) are never moved.  A linestring whose snapped version would
    self-intersect is left unchanged, with a warning.

    Only interior HUC boundaries are snapped, never the domain's exterior
    boundary.  An exterior boundary made to coincide with a corridor edge
    leaves nothing between them to triangulate, so the corridor vertices
    there belong to no triangle; Triangle then drops them from its output
    and renumbers every later vertex, which scrambles the river elements.

    Must be called after createRiversMesh() and before triangulation.
    Modifies hucs.linestrings in place.

    Parameters
    ----------
    hucs : Watershed
        Split HUCs object, as modified by createRiversMesh().
    corridors : List[shapely.geometry.Polygon]
        River corridor polygons returned by createRiversMesh().
    tol : float
        Distance within which a HUC boundary vertex is snapped to the
        corridor, in the units of the CRS.
    max_gap_vertices : int, optional
        Largest number of corridor vertices inserted between two
        consecutive snapped vertices.

    Returns
    -------
    List[Tuple[int, int, int]]
        For each modified linestring: (handle, vertices snapped, corridor
        vertices inserted).
    """
    rings = []
    for corridor in corridors:
        for ring in [corridor.exterior, ] + list(corridor.interiors):
            rings.append(np.array(ring.coords)[:-1, 0:2])
    ring_linestrings = [shapely.geometry.LineString(np.vstack([ring, ring[0:1]])) for ring in rings]
    ring_trees = [shapely.STRtree([shapely.geometry.Point(p) for p in ring]) for ring in rings]
    all_rings = shapely.ops.unary_union(ring_linestrings)

    exterior_handles = set(h for spine in hucs.boundaries.values() for h in spine.values())

    report = []
    for handle, ls in list(hucs.linestrings.items()):
        if handle in exterior_handles or all_rings.distance(ls) > tol:
            continue

        # snap each interior vertex near a corridor to the closest corridor vertex
        coords = [tuple(c[0:2]) for c in ls.coords]
        snapped : List[Tuple[Tuple[float, float], Optional[int], Optional[int]]] = []
        n_snapped = 0
        for i, c in enumerate(coords):
            p = shapely.geometry.Point(c)
            if 0 < i < len(coords) - 1 and all_rings.distance(p) < tol:
                k = int(np.argmin([rls.distance(p) for rls in ring_linestrings]))
                j = int(ring_trees[k].nearest(p))
                snapped.append((tuple(rings[k][j]), k, j))
                n_snapped += 1
            else:
                snapped.append((c, None, None))
        if n_snapped == 0:
            continue

        # follow the corridor outline between consecutive snapped vertices
        new_coords = [snapped[0][0], ]
        n_inserted = 0
        for (pa, ka, ja), (pb, kb, jb) in zip(snapped[:-1], snapped[1:]):
            if ka is not None and ka == kb and ja != jb:
                n = len(rings[ka])
                forward = (jb - ja) % n
                backward = (ja - jb) % n
                step, count = (1, forward) if forward <= backward else (-1, backward)
                if count - 1 <= max_gap_vertices:
                    for t in range(1, count):
                        new_coords.append(tuple(rings[ka][(ja + step*t) % n]))
                        n_inserted += 1
            new_coords.append(pb)

        new_coords = [new_coords[0], ] + [c for c_prev, c in zip(new_coords[:-1], new_coords[1:])
                                          if not watershed_workflow.utils.geometry.isClose(c, c_prev)]
        new_ls = shapely.geometry.LineString(new_coords)
        if len(new_coords) < 2 or not new_ls.is_simple:
            logging.warning(f'snapHUCsToCorridors: snapping HUC linestring {handle} would make it '
                            'self-intersect; leaving it unchanged')
            continue
        hucs.linestrings[handle] = new_ls
        report.append((handle, n_snapped, n_inserted))

    logging.info(f'snapHUCsToCorridors: snapped {sum(r[1] for r in report)} vertices on '
                 f'{len(report)} HUC linestrings to river corridors (tol = {tol})')
    return report



def snapCorridorsToExterior(hucs: Watershed,
                            coords: np.ndarray,
                            corridors: List[shapely.geometry.Polygon],
                            tol: float,
                            vertex_tol: float = 1.0) -> Tuple[List[shapely.geometry.Polygon], List[Dict]]:
    """Move river-corridor vertices that lie just inside the domain boundary onto it.

    Where the domain's exterior boundary runs within a few metres of a
    corridor edge (typically right at the domain outlet, next to the
    corridor corner that adjustHUCsToRiverMesh() put on the boundary), the
    sliver of land between them becomes one or more tiny cells.  Each
    corridor vertex within tol of an exterior HUC linestring (but not
    already on it) is moved onto that linestring: onto an existing boundary
    vertex if within vertex_tol of one (the triangulation rejects distinct
    vertices closer than that), otherwise onto its projection, which is
    inserted into the linestring.  The exterior boundary itself does not
    move.

    Afterwards a corridor vertex may have no land triangle next to it;
    tessalateRiverAligned() handles that by matching river vertices to the
    triangulation's output by coordinate.

    Must be called after createRiversMesh() and before triangulation.

    Parameters
    ----------
    hucs : Watershed
        Split HUCs object; its exterior linestrings are modified in place.
    coords : np.ndarray
        River corridor coordinates returned by createRiversMesh(); modified
        in place.
    corridors : List[shapely.geometry.Polygon]
        River corridor polygons returned by createRiversMesh().
    tol : float
        Distance within which a corridor vertex is moved, in CRS units.
    vertex_tol : float, optional
        Snap to an existing boundary vertex rather than inserting a new one
        if within this distance.

    Returns
    -------
    corridors : List[shapely.geometry.Polygon]
        Corridor polygons rebuilt with the moved vertices.
    moves : List[dict]
        One entry per moved vertex: index, old, new, distance, handle.
    """
    exterior_handles = [h for spine in hucs.boundaries.values() for h in spine.values()]
    exterior = shapely.ops.unary_union([hucs.linestrings[h] for h in exterior_handles])

    moves = []
    for i in range(len(coords)):
        p = shapely.geometry.Point(coords[i][0:2])
        if exterior.distance(p) >= tol:
            continue
        handle = min(exterior_handles, key=lambda h: hucs.linestrings[h].distance(p))
        ls = hucs.linestrings[handle]
        d = ls.distance(p)
        if d < 1.e-6:
            continue

        ls_coords = [tuple(c[0:2]) for c in ls.coords]
        s = ls.project(p)
        q = ls.interpolate(s)
        nearest = min(range(len(ls_coords)), key=lambda k: shapely.geometry.Point(ls_coords[k]).distance(q))
        if shapely.geometry.Point(ls_coords[nearest]).distance(q) < vertex_tol:
            new_c = ls_coords[nearest]
        else:
            new_c = (q.x, q.y)
            arclen = 0.
            for k in range(len(ls_coords) - 1):
                seg_len = shapely.geometry.Point(ls_coords[k]).distance(shapely.geometry.Point(ls_coords[k+1]))
                if arclen + seg_len >= s:
                    ls_coords.insert(k + 1, new_c)
                    break
                arclen += seg_len
            hucs.linestrings[handle] = shapely.geometry.LineString(ls_coords)

        moves.append(dict(index=i, old=tuple(coords[i][0:2]), new=new_c, distance=d, handle=handle))
        coords[i][0] = new_c[0]
        coords[i][1] = new_c[1]

    if len(moves) > 0:
        old_to_new = { tuple(np.round(m['old'], 6)) : m['new'] for m in moves }
        def _update(ring):
            return [old_to_new.get(tuple(np.round(c[0:2], 6)), tuple(c[0:2])) for c in ring.coords]
        corridors = [shapely.geometry.Polygon(_update(c.exterior), [_update(r) for r in c.interiors])
                     for c in corridors]
    logging.info(f'snapCorridorsToExterior: moved {len(moves)} corridor vertices onto the '
                 f'domain boundary (tol = {tol})')
    return corridors, moves


def matchRiverVertices(tri_coords: np.ndarray,
                       coords: np.ndarray,
                       elems: List[List[int]],
                       rivers: List[River],
                       tol: float = 1.e-3) -> Tuple[np.ndarray, List[List[int]], int]:
    """Re-index the river elements into the triangulation's vertices.

    The river elements index the corridor coordinates returned by
    createRiversMesh().  The triangulation lists those vertices first, but
    Triangle (as called by meshpy) drops any input vertex that is in no
    triangle -- e.g. a corridor corner with no land next to it on the
    domain boundary -- and renumbers every later vertex.  This finds each
    corridor vertex in the triangulation by coordinate, appends any that
    were dropped, and re-indexes the river elements (both elems and each
    reach's names.ELEMS) accordingly.

    Returns
    -------
    coords : np.ndarray
        The triangulation's vertices plus any appended corridor vertices.
    elems : List[List[int]]
        River elements indexing coords.
    n_appended : int
        Number of corridor vertices the triangulation had dropped.
    """
    tri_coords = np.asarray(tri_coords)
    coords = np.asarray(coords)
    tree = scipy.spatial.cKDTree(tri_coords[:, 0:2])
    dist, index = tree.query(coords[:, 0:2])
    index = np.where(dist < tol, index, -1)

    missing = np.where(index < 0)[0]
    if len(missing) > 0:
        extra = np.zeros((len(missing), tri_coords.shape[1]))
        n = min(tri_coords.shape[1], coords.shape[1])
        extra[:, 0:n] = coords[missing, 0:n]
        index[missing] = len(tri_coords) + np.arange(len(missing))
        tri_coords = np.vstack([tri_coords, extra])
        logging.info(f'matchRiverVertices: re-added {len(missing)} corridor vertices dropped by the triangulation')

    new_elems = [[int(index[v]) for v in e] for e in elems]
    for river in rivers:
        for reach in river:
            reach[names.ELEMS][:] = [[int(index[v]) for v in e] for e in reach[names.ELEMS]]
    return tri_coords, new_elems, len(missing)

def computeLine(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float, float]:
    """Compute line coefficients (Ax + By + C = 0) for a line defined by two points.

    Parameters
    ----------
    p1 : np.ndarray
        First point coordinates.
    p2 : np.ndarray
        Second point coordinates.

    Returns
    -------
    Tuple[float, float, float]
        Line coefficients (A, B, C) such that Ax + By + C = 0.
    """
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = p2[0] * p1[1] - p1[0] * p2[1]
    return A, B, C


def translateLinePerpendicular(line: Tuple[float, float, float],
                               distance: float) -> Tuple[float, float, float]:
    """Translate a line by a specified distance in the direction perpendicular to the line.

    Parameters
    ----------
    line : Tuple[float, float, float]
        Tuple of line coefficients (A, B, C).
    distance : float
        Scalar distance to translate the line.

    Returns
    -------
    Tuple[float, float, float]
        Tuple of new line coefficients (A, B, C).
    """
    A, B, C = line
    # Normalize A and B to get the unit normal vector
    normal_length = np.hypot(A, B)
    if normal_length == 0:
        raise ValueError("Invalid line coefficients.")

    # Translate C by the perpendicular distance
    C_new = C - distance*normal_length
    return A, B, C_new


def findIntersection(line1: Tuple[float, float, float],
                     line2: Tuple[float, float, float],
                     debug: bool = False) -> np.ndarray | None:
    """Find the intersection point of two lines given by coefficients.

    Parameters
    ----------
    line1 : Tuple[float, float, float]
        First line coefficients (A, B, C).
    line2 : Tuple[float, float, float]
        Second line coefficients (A, B, C).
    debug : bool, optional
        Whether to print debug information, by default False.

    Returns
    -------
    np.ndarray | None
        Intersection point coordinates, or None if lines are parallel.
    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2

    # shift the intercept to near the origin
    determinant = A1*B2 - A2*B1
    eps = 1.e-8 * max(abs(A1), abs(A2), abs(B1), abs(B2), abs(C1), abs(C2))
    if debug:
        logging.info(f"  Parallel?  det = {determinant} relative {eps}")
    if abs(determinant) < eps:
        return None  # Lines are parallel or coincident

    x = (B1*C2 - B2*C1) / determinant
    y = (A2*C1 - A1*C2) / determinant
    return np.array([x,y])


def projectOne(p_up: np.ndarray, p: np.ndarray, width: float) -> np.ndarray:
    """Find a point p_out that is width away from p and such that p_up --> p is right-perpendicular to p --> p_out.

    Parameters
    ----------
    p_up : np.ndarray
        Upstream point coordinates.
    p : np.ndarray
        Reference point coordinates.
    width : float
        Distance to project perpendicular to the line.

    Returns
    -------
    np.ndarray
        Projected point coordinates.
    """
    c_up = np.array(p_up)
    c = np.array(p)

    dp = (c - c_up)
    dp /= np.linalg.norm(dp)
    perp = np.array([dp[1], -dp[0]])
    return p + width*perp


def _projectTwoMiter(p_up: np.ndarray,
               p: np.ndarray,
               p_dn: np.ndarray,
               width1: float,
               width2: float,
               debug: bool = False) -> np.ndarray:
    """Find a point that is perpendicular to the linestring p_up --> p --> p_dn.

    Projects by intersecting two lines, one parallel to p_up --> p and width1 away,
    and one parallel to p --> p_dn and width2 away.

    Parameters
    ----------
    p_up : np.ndarray
        Upstream point coordinates.
    p : np.ndarray
        Middle point coordinates.
    p_dn : np.ndarray
        Downstream point coordinates.
    width1 : float
        Width for upstream segment.
    width2 : float
        Width for downstream segment.
    debug : bool, optional
        Whether to print debug information, by default False.

    Returns
    -------
    np.ndarray
        Projected point coordinates.
    """
    l1 = computeLine(p_up, p)
    if debug:
        logging.info(f"line defined by: {p_up} --> {p} = {l1}")
    l1 = translateLinePerpendicular(l1, width1)
    if debug:
        logging.info(f"  translated by {width1} = {l1}")

    l2 = computeLine(p, p_dn)
    if debug:
        logging.info(f"line defined by: {p} --> {p_dn} = {l2}")
    l2 = translateLinePerpendicular(l2, width2)
    if debug:
        logging.info(f"  translated by {width2} = {l2}")

    intersection = findIntersection(l1, l2, debug)
    if intersection is None:
        new_p = projectOne(p_up, p, (width1+width2) / 2.)
        if debug:
            logging.info(f"parallel! results in intersection = {new_p}")
        return new_p

    else:
        if debug:
            logging.info(f"results in intersection = {intersection}")
        return intersection


def projectTwoClampedMiter(p_up: np.ndarray,
                           p: np.ndarray,
                           p_dn: np.ndarray,
                           width1: float,
                           width2: float,
                           debug: bool = False) -> np.ndarray:
    """Find a bank point using the miter approach, clamped to avoid overshoot.

    Computes the miter intersection via _projectTwoMiter(), then applies one
    of two strategies depending on the bend angle theta at p:

    - theta > 90 deg (nearly-straight reach, p_up-p-p_dn angle close to
      180): the miter direction becomes unreliable; fall back to the
      bisector direction with the average width, similarly capped.
    - theta <= 90 deg (sharp bend): the miter direction is reliable; cap
      ``|m - p|`` to 3/4 * min(``|p_up - p|``, ``|p_dn - p|``) to prevent overshoot.

    In both cases the fallback also triggers when ``|m - p|`` already exceeds
    the cap (e.g. parallel segments where _projectTwoMiter returns a
    perpendicular).

    Parameters
    ----------
    p_up : np.ndarray
        Upstream point coordinates.
    p : np.ndarray
        Middle point coordinates.
    p_dn : np.ndarray
        Downstream point coordinates.
    width1 : float
        Width for upstream segment.
    width2 : float
        Width for downstream segment.
    debug : bool, optional
        Whether to log debug information, by default False.

    Returns
    -------
    np.ndarray
        Projected bank point coordinates.
    """
    p_up = np.asarray(p_up, dtype=float)
    p = np.asarray(p, dtype=float)
    p_dn = np.asarray(p_dn, dtype=float)

    cap_d = 0.75 * min(np.linalg.norm(p_up - p), np.linalg.norm(p_dn - p))
    avg_width = (width1 + width2) / 2.0

    # bend angle theta at p: angle between incoming and outgoing unit tangents
    v_up = (p_up - p) / np.linalg.norm(p_up - p)
    v_dn = (p_dn - p) / np.linalg.norm(p_dn - p)
    cos_theta = np.clip(np.dot(v_up, v_dn), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))  # 180 = straight, 0 = U-turn

    m = _projectTwoMiter(p_up, p, p_dn, width1, width2, debug)
    d = np.linalg.norm(m - p)

    if d > cap_d:
        if theta > 90.0:
            # nearly-straight bend + overshoot: miter direction unreliable, use bisector
            if debug:
                logging.info(f"bisector fallback: theta={theta:.1f}, d={d:.4f}, cap_d={cap_d:.4f}")
            m = projectTwoBisector(p_up, p, p_dn, avg_width, debug)
            d = np.linalg.norm(m - p)
            if d > cap_d:
                m = p + cap_d * (m - p) / d
        else:
            # sharp bend (theta <= 90) + overshoot: miter direction is fine, just clamp length
            if debug:
                logging.info(f"clamping miter: theta={theta:.1f}, d={d:.4f}, cap_d={cap_d:.4f}")
            m = p + cap_d * (m - p) / d
    return m


def projectTwoBisector(p_up: np.ndarray,
                       p: np.ndarray,
                       p_dn: np.ndarray,
                       width: float,
                       debug: bool = False) -> np.ndarray:
    """Find a point that is width away from p along the bisector of p_up --> p --> p_dn.

    Unlike projectTwoMiter(), which uses a miter approach and accepts two widths,
    this uses the angle bisector of the two segments and a single width.  The
    result is always exactly width away from p, so it cannot overshoot
    regardless of the bend angle or width ratio between adjacent reaches.

    When the two segments are antiparallel (straight line, zero bend), the
    bisector is degenerate and the function falls back to a simple perpendicular
    projection via projectOne() using the upstream segment.

    Parameters
    ----------
    p_up : np.ndarray
        Upstream point coordinates.
    p : np.ndarray
        Middle point coordinates.
    p_dn : np.ndarray
        Downstream point coordinates.
    width : float
        Distance from p to the projected bank point, measured along the bisector.
    debug : bool, optional
        Whether to log debug information, by default False.

    Returns
    -------
    np.ndarray
        Projected bank point coordinates.
    """
    p_up = np.asarray(p_up, dtype=float)
    p = np.asarray(p, dtype=float)
    p_dn = np.asarray(p_dn, dtype=float)

    # unit tangent of upstream segment pointing into p
    d1 = p - p_up
    d1 /= np.linalg.norm(d1)

    # unit tangent of downstream segment pointing away from p
    d2 = p_dn - p
    d2 /= np.linalg.norm(d2)

    # right-hand perpendiculars of each segment at p
    perp1 = np.array([d1[1], -d1[0]])
    perp2 = np.array([d2[1], -d2[0]])

    # bisector of the two perpendiculars
    bisector = perp1 + perp2
    bisector_norm = np.linalg.norm(bisector)

    if bisector_norm < 1e-10:
        # antiparallel segments (straight reach): fall back to upstream perpendicular
        if debug:
            logging.info(f"bisector degenerate (straight reach), falling back to projectOne")
        return projectOne(p_up, p, width)

    bisector /= bisector_norm

    result = p + width * bisector
    if debug:
        logging.info(f"bisector = {bisector}, result = {result}")
    return result


def projectJunction(reach: River,
                    child_idx: int,
                    computeWidth: Callable[[River], float],
                    debug: bool = False) -> np.ndarray:
    """Find points around the junction between reach and its children.

    Parameters
    ----------
    reach : River
        Parent reach containing the junction.
    child_idx : int
        Index of the child reach at the junction.
    computeWidth : Callable[[River], float]
        Function to compute river width.
    debug : bool, optional
        Whether to print debug information, by default False.

    Returns
    -------
    np.ndarray
        Junction point coordinates.
    """
    if child_idx == 0:
        return projectTwoClampedMiter(reach.children[0].linestring.coords[-2], reach.linestring.coords[0],
                          reach.linestring.coords[1],
                          computeWidth(reach.children[0]) / 2.,
                          computeWidth(reach) / 2., debug)
    elif child_idx == len(reach.children):
        return projectTwoClampedMiter(reach.linestring.coords[1], reach.linestring.coords[0],
                          reach.children[-1].linestring.coords[-2],
                          computeWidth(reach) / 2.,
                          computeWidth(reach.children[-1]) / 2., debug)
    else:
        return projectTwoClampedMiter(reach.children[child_idx].linestring.coords[-2],
                          reach.linestring.coords[0],
                          reach.children[child_idx - 1].linestring.coords[-2],
                          computeWidth(reach.children[child_idx]) / 2.,
                          computeWidth(reach.children[child_idx - 1]) / 2., debug)


def fixConvexity(reach: River, e_coords: np.ndarray, computeWidth: Callable[[River],
                                                                            float]) -> np.ndarray:
    """Snap element coordinates onto the convex hull while respecting upstream stream width.

    Parameters
    ----------
    reach : River
        River reach containing the element.
    e_coords : np.ndarray
        Element coordinates to fix.
    computeWidth : Callable[[River], float]
        Function to compute river width.

    Returns
    -------
    np.ndarray
        Fixed element coordinates.
    """
    e_poly = shapely.geometry.Polygon(e_coords)
    e_poly_hull = e_poly.convex_hull

    # snap to convex hull
    fix_points = []
    for i, coord in enumerate(e_coords):
        closest_p = watershed_workflow.utils.geometry.findNearestPoint(shapely.geometry.Point(coord),
                                                              e_poly_hull.boundary)

        if not watershed_workflow.utils.geometry.isClose(coord, closest_p, 1.e-4):
            fix_points.append(i)

            if i == 1 or i == len(e_coords) - 2:
                # intersect the shifted line with the convex hull to find the new point
                if i == 1:
                    child = reach.children[0]
                    sign = 1
                elif i == len(e_coords) - 2:
                    child = reach.children[-1]
                    sign = -1

                p0 = child.linestring.coords[-1]
                p1 = child.linestring.coords[-2]
                halfwidth = computeWidth(child) / 2.
                p0p = projectOne(p1, p0, sign * halfwidth)
                p1p = projectOne(p0, p1, -sign * halfwidth)
                ls = shapely.geometry.LineString([p0p, p1p])
                if not ls.intersects(e_poly_hull.boundary):
                    fig, ax = plt.subplots(1, 1)

                    reaches = [reach, ]
                    if reach.parent is not None:
                        reaches.append(reach.parent)
                    reaches = reaches + list(reach.children)
                    for r in reaches:
                        ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'b-x')

                    poly = shapely.geometry.Polygon(e_coords)
                    ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'g-x')

                    ax.plot(ls.xy[0], ls.xy[1], 'r-x')
                    ax.plot(e_poly_hull.boundary.xy[0], e_poly_hull.boundary.xy[1], 'k-x')
                    ax.set_aspect('equal', adjustable='box')
                    plt.show()
                    assert False, "No intersection point with convex hull at reach {}?".format(reach.properties['comid'])

                new_c_p = ls.intersection(e_poly_hull.boundary)

                if isinstance(new_c_p, shapely.geometry.MultiPoint):
                    # two intersections...
                    fig, ax = plt.subplots(1, 1)

                    reaches = [reach, ]
                    if reach.parent is not None:
                        reaches.append(reach.parent)
                    reaches = reaches + list(reach.children)
                    for r in reaches:
                        ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'b-x')

                    poly = shapely.geometry.Polygon(e_coords)
                    ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'g-x')

                    ax.plot(ls.xy[0], ls.xy[1], 'r-x')
                    ax.plot(e_poly_hull.boundary.xy[0], e_poly_hull.boundary.xy[1], 'k-x')
                    ax.set_aspect('equal', adjustable='box')
                    plt.show()
                    assert False, "Dual intersection points with convex hull for reach {}?".format(reach.properties['comid'])

                assert isinstance(new_c_p, shapely.geometry.Point)
                new_c = new_c_p.coords[0]

            elif i > 1 and i < len(e_coords) - 2:
                # snap it to the nearest point?
                new_c = closest_p

            else:
                # a 0th or last point should never be non-convex?
                fig, ax = plt.subplots(1, 1)

                reaches = [reach, reach.parent] + list(reach.children)
                for r in reaches:
                    ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'b-x')

                poly = shapely.geometry.Polygon(e_coords)
                ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'g-x')
                ax.scatter([coord[0], ], [coord[1], ], color='g', marker='o')
                ax.plot(e_poly_hull.boundary.xy[0], e_poly_hull.boundary.xy[1], 'k-x')
                ax.set_aspect('equal', adjustable='box')
                plt.show()

                assert False, "Do not know how to deal with non-convexity that doesn't start at the outermost children."

            e_coords[i] = new_c

    if not watershed_workflow.utils.geometry.isConvex(e_coords):
        # a 0th or last point should never be non-convex?
        fig, ax = plt.subplots(1, 1)

        reaches = [reach,]
        if reach.parent is not None:
            reaches.append(reach.parent)
        reaches = reaches + list(reach.children)

        for r in reaches:
            ax.plot(r.linestring.xy[0], r.linestring.xy[1], 'b-x')

        poly = shapely.geometry.Polygon(e_coords)
        ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], 'g-x')

        for i in fix_points:
            ax.scatter([e_coords[i, 0], ], [e_coords[i, 1], ], color='g', marker='o')

        ax.plot(e_poly_hull.boundary.xy[0], e_poly_hull.boundary.xy[1], 'k-x')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        assert False, "Cannot fix nonconvexity for reach {}?".format(reach.properties['comid'])

    return e_coords

