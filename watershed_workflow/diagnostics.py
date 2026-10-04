"""Find the topology defects that Watershed Workflow will not fix on its own.

Many inconsistencies between a HUC layer and a river network are repaired
by the standard pipeline itself: simplify() snaps HUC triple junctions onto
nearby confluences and reach endpoints onto HUC boundaries, updates each
HUC's downstream HUC and outlet, and tessalateRiverAligned() snaps HUC
boundaries onto river corridors.  Rather than flag every such inconsistency,
findDefects() runs that pipeline and reports only what it rejects or leaves
broken -- the defects that have to be fixed in the input data (or by a
targeted repair, e.g. hydro.watershed.offsetDivideFromReaches()).

The pipeline normally stops at the first failure.  To find every defect in
one pass, the steps of simplify() are run individually here and failures
are isolated per reach: a failing reach is recorded and the scan continues.
The step list must be kept in sync with watershed_workflow.simplify().

Example
-------
>>> report = watershed_workflow.diagnostics.findDefects(hucs_gdf, rivers_gdf, 60., 150.,
...                                                     snap_triple_junctions_tol=50.,
...                                                     min_cell_area=50., refine_min_angle=32.)
>>> print(report.summary())
>>> report.to_dataframe().explore()
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import collections
import dataclasses
import logging

import numpy as np
import geopandas as gpd
import shapely
import shapely.geometry
import shapely.ops
from matplotlib import pyplot as plt

import watershed_workflow
import watershed_workflow.hydro.angles
import watershed_workflow.hydro.hydrography
import watershed_workflow.hydro.resampling
import watershed_workflow.hydro.river
import watershed_workflow.hydro.watershed
import watershed_workflow.mesh.river_mesh
import watershed_workflow.sources.standard_names as names

__all__ = ['Defect', 'DefectReport', 'findDefects']

#: Defect kinds after which meshing is not attempted: the network or the HUCs
#: are not in a state the mesh stage can meaningfully build from.
FATAL_KINDS = ('createRivers', 'watershed', 'exterior-crossing', 'multiple-crossing',
               'continuity', 'snapReachEndpoints', 'sharp-angles')


@dataclasses.dataclass
class Defect:
    """One defect that Watershed Workflow will not fix.

    stage : str
        Pipeline stage that found it: 'rivers', 'hucs', 'simplify',
        'outlets' or 'mesh'.
    kind : str
        Short identifier, e.g. 'empty-geometry', 'multiple-crossing',
        'multiple-outlets', 'corridor-overlap', 'small-cells'.
    message : str
        Human-readable description.
    location : shapely.geometry.Point, optional
        Where it is, in the CRS of the inputs.
    reaches, hucs : list
        IDs of the reaches and HUCs involved.
    details : dict
        Kind-specific data (e.g. outlet locations, cell areas).
    """
    stage : str
    kind : str
    message : str
    location : Optional[shapely.geometry.Point] = None
    reaches : List[Any] = dataclasses.field(default_factory=list)
    hucs : List[Any] = dataclasses.field(default_factory=list)
    details : Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DefectReport:
    """Result of findDefects().

    defects : List[Defect]
        What Watershed Workflow will not fix.
    notes : List[str]
        What it handled itself, and stages that were skipped.
    mesh : Mesh2D, optional
        The mesh, if the mesh stage ran and succeeded.
    params : dict
        The findDefects() settings that later steps need (e.g. repair
        proposals keep new HUC corners outside snap_triple_junctions_tol).
    """
    defects : List[Defect] = dataclasses.field(default_factory=list)
    notes : List[str] = dataclasses.field(default_factory=list)
    mesh : Any = None
    params : Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if no defects were found."""
        return len(self.defects) == 0

    def add(self, stage : str, kind : str, message : str, **kwargs) -> None:
        self.defects.append(Defect(stage, kind, message, **kwargs))
        logging.warning(f'findDefects [{stage}/{kind}] {message}')

    def note(self, message : str) -> None:
        self.notes.append(message)
        logging.info(f'findDefects: {message}')

    def summary(self) -> str:
        counts = collections.Counter(d.kind for d in self.defects)
        lines = [f'{len(self.defects)} defect(s) Watershed Workflow will not fix'
                 + (f': {dict(counts)}' if counts else '')]
        lines += [f'  [{d.stage}/{d.kind}] {d.message}' for d in self.defects]
        if self.notes:
            lines.append(f'{len(self.notes)} note(s):')
            lines += [f'  {n}' for n in self.notes]
        return '\n'.join(lines)

    def to_dataframe(self, crs : Any = None) -> gpd.GeoDataFrame:
        """Defects as a GeoDataFrame of points (empty geometry if no location)."""
        rows = [dict(stage=d.stage, kind=d.kind, message=d.message,
                     reaches=[str(r) for r in d.reaches], hucs=[str(h) for h in d.hucs]) for d in self.defects]
        geoms = [d.location if d.location is not None else shapely.geometry.Point() for d in self.defects]
        return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def _reachId(reach) -> Any:
    return reach[names.ID] if names.ID in reach else reach.index


def _buildRivers(report : DefectReport, rivers_gdf : gpd.GeoDataFrame) -> list:
    empty = rivers_gdf.geometry.isna() | rivers_gdf.geometry.is_empty
    if empty.any():
        ids = list(rivers_gdf.loc[empty, names.ID]) if names.ID in rivers_gdf else list(rivers_gdf.index[empty])
        report.add('rivers', 'empty-geometry',
                   f'{int(empty.sum())} reaches have empty geometry; createRivers() cannot build a network '
                   'with them (they are dropped for the rest of this scan)', reaches=ids)
        rivers_gdf = rivers_gdf[~empty]

    try:
        rivers = watershed_workflow.hydro.river.createRivers(rivers_gdf.copy(), method='geometry')
    except Exception as err:
        report.add('rivers', 'createRivers', f'createRivers(method="geometry") failed: {type(err).__name__}: {err}')
        return []

    if names.HYDROSEQ in rivers_gdf and names.DOWNSTREAM_HYDROSEQ in rivers_gdf:
        try:
            rivers_h = watershed_workflow.hydro.river.createRivers(rivers_gdf.copy(), method='hydroseq')
        except Exception as err:
            report.add('rivers', 'hydroseq-vs-geometry',
                       f'createRivers(method="hydroseq") failed: {type(err).__name__}: {err}')
        else:
            def parents(trees):
                return { _reachId(n) : (_reachId(n.parent) if n.parent is not None else None)
                         for t in trees for n in t }
            p_g, p_h = parents(rivers), parents(rivers_h)
            bad = sorted((k for k in p_g if p_g[k] != p_h.get(k)), key=str)
            if bad:
                report.add('rivers', 'hydroseq-vs-geometry',
                           f'{len(bad)} reaches have a different downstream reach by hydroseq than by '
                           f'geometry ({len(rivers_h)} trees by hydroseq, {len(rivers)} by geometry)',
                           reaches=bad)

    rivers.sort(key=lambda t: -len(t))
    for t in rivers[1:]:
        report.note(f'separate river network rooted at reach {_reachId(t)} ({len(t)} reaches); if it '
                    'leaves the domain it will show up as an extra outlet')
    return rivers


def _simplifyIsolated(report : DefectReport,
                      hucs : watershed_workflow.hydro.watershed.Watershed,
                      rivers : list,
                      reach_segment_target_length : float,
                      huc_segment_target_length : Optional[float],
                      river_close_distance : float,
                      river_far_distance : float,
                      min_angle : float,
                      junction_min_angle : float,
                      snap_triple_junctions_tol : Optional[float]) -> None:
    """The steps of watershed_workflow.simplify(), failures isolated per reach."""
    L = reach_segment_target_length
    hydro = watershed_workflow.hydro.hydrography

    watershed_workflow.hydro.river.simplify(rivers, 1.e-4 * L)
    watershed_workflow.hydro.watershed.simplify(hucs, 1.e-4 * L)
    for river in rivers:
        watershed_workflow.hydro.river.pruneByLineStringLength(river, L)
    for river in rivers:
        watershed_workflow.hydro.river.mergeShortReaches(river, 0.75 * L)

    if snap_triple_junctions_tol is None:
        snap_triple_junctions_tol = 3 * L
    hydro.snapHUCsJunctions(hucs, rivers, snap_triple_junctions_tol)
    for river in rivers:
        hydro.snapReachEndpoints(hucs, river, L)
        if not river.isContinuous():
            report.add('simplify', 'snapReachEndpoints',
                       f'river {_reachId(river)} is not continuous after snapReachEndpoints()',
                       reaches=[_reachId(river)])

    for river in rivers:
        for node in [river, ] + list(river.leaf_nodes):
            try:
                hydro._cutAndSnapExteriorCrossing(hucs, node, L)
            except Exception as err:
                report.add('simplify', 'exterior-crossing',
                           f'reach {_reachId(node)}: cutting it at the domain boundary failed '
                           f'({type(err).__name__}: {err})',
                           location=shapely.geometry.Point(node.linestring.coords[-1]),
                           reaches=[_reachId(node)])

    for river in rivers:
        for reach in river:
            try:
                hydro._cutAndSnapInteriorCrossing(hucs, reach, L)
            except Exception as err:
                n_cross = 0
                for spine in hucs.intersections.values():
                    for handle in spine.values():
                        it = reach.linestring.intersection(hucs.linestrings[handle])
                        n_cross = max(n_cross, 0 if it.is_empty else len(getattr(it, 'geoms', [it, ])))
                report.add('simplify', 'multiple-crossing',
                           f'reach {_reachId(reach)}: its HUC-boundary crossings cannot be cut and snapped '
                           f'(it crosses one boundary segment {n_cross} times); a reach running along a '
                           'divide can be fixed with hydro.watershed.offsetDivideFromReaches()',
                           location=reach.linestring.interpolate(0.5, normalized=True),
                           reaches=[_reachId(reach)], details=dict(crossings=n_cross))

    for river in rivers:
        if not river.isContinuous():
            bad = [_reachId(n) for n in river if not n.isLocallyContinuous()]
            report.add('simplify', 'continuity', f'river {_reachId(river)} is discontinuous after cutting '
                       f'at HUC boundaries, at reaches {bad}', reaches=bad)

    if huc_segment_target_length is not None:
        dfunc = (river_close_distance, L, river_far_distance, huc_segment_target_length)
        river_mls = shapely.ops.unary_union([river.to_mls() for river in rivers])
        watershed_workflow.hydro.resampling.resampleWatershed(hucs, dfunc, river_mls)
    else:
        watershed_workflow.hydro.resampling.resampleWatershed(hucs, L)
    watershed_workflow.hydro.resampling.resampleRivers(rivers, L)

    try:
        watershed_workflow.hydro.angles.smoothSharpAngles(hucs, rivers, min_angle, junction_min_angle)
    except Exception as err:
        report.add('simplify', 'sharp-angles', f'smoothSharpAngles() failed: {type(err).__name__}: {err}')


def _checkOutlets(report : DefectReport,
                  hucs : watershed_workflow.hydro.watershed.Watershed,
                  rivers : list,
                  huc_id_col : str) -> None:
    if 'tohuc' not in hucs.df.columns:
        report.note("HUCs have no 'tohuc' column; the outlet check found each HUC's outlets only")
        outlets = watershed_workflow.hydro.hydrography.findHUCOutlets(hucs, rivers)
        ids = list(hucs.df[huc_id_col]) if huc_id_col in hucs.df else list(range(len(outlets)))
        multiple = { ids[i] : o for i, o in enumerate(outlets) if len(o) > 1 }
    else:
        ids = list(hucs.df[huc_id_col])
        old = list(hucs.df['tohuc'])
        try:
            watershed_workflow.hydro.hydrography.updateToHUCs(hucs, rivers, id_col=huc_id_col)
            multiple = {}
        except watershed_workflow.hydro.hydrography.MultipleOutletsError as err:
            multiple = err.outlets
        else:
            for i, h in enumerate(ids):
                if hucs.df['tohuc'].iloc[i] != old[i]:
                    report.note(f'Watershed Workflow updates tohuc of HUC {h}: {old[i]} -> {hucs.df["tohuc"].iloc[i]}')

    for huc_id, outlets in multiple.items():
        locs = [dict(point=o.point, into=('domain exterior' if o.downstream is None else ids[o.downstream]),
                     reaches=[_reachId(r) for r in o.reaches]) for o in outlets]
        report.add('outlets', 'multiple-outlets',
                   f'HUC {huc_id} has {len(outlets)} outlet locations: '
                   + '; '.join(f"into {l['into']} via reach(es) {l['reaches']} at "
                               f"({l['point'].x:.1f}, {l['point'].y:.1f})" for l in locs),
                   location=outlets[0].point, hucs=[huc_id],
                   reaches=[r for l in locs for r in l['reaches']], details=dict(outlets=locs))


def _checkMesh(report : DefectReport,
               hucs : watershed_workflow.hydro.watershed.Watershed,
               rivers : list,
               river_width : Callable,
               min_cell_area : Optional[float],
               cluster_distance : float,
               tessalate_kwargs : dict) -> None:
    try:
        res = watershed_workflow.tessalateRiverAligned(hucs, rivers, river_width=river_width, **tessalate_kwargs)
    except Exception as err:
        report.add('mesh', 'tessalateRiverAligned', f'tessalateRiverAligned() failed: {type(err).__name__}: {err}')
        return

    if isinstance(res, tuple) and len(res) == 3 and res[2] is not None and isinstance(res[2], gpd.GeoDataFrame):
        for _, row in res[2].iterrows():
            report.add('mesh', 'corridor-overlap', f'the corridors of reaches {row["i"]} and {row["j"]} '
                       f'overlap ({row.geometry.area:.1f} in area)',
                       location=row.geometry.representative_point(), reaches=[row['i'], row['j']])
        return

    m2 = res[0] if isinstance(res, tuple) else res
    report.mesh = m2
    coords = np.asarray(m2.coords)[:, 0:2]
    areas = np.array([shapely.geometry.Polygon(coords[c]).area for c in m2.conn])
    report.note(f'mesh built: {len(areas)} cells, smallest {areas.min():.1f}')

    if min_cell_area is None:
        return
    small = np.where(areas < min_cell_area)[0]
    if len(small) == 0:
        report.note(f'no cell is smaller than {min_cell_area}')
        return

    # group small cells into sites
    centroids = np.array([coords[m2.conn[c]].mean(axis=0) for c in small])
    site = -np.ones(len(small), dtype=int)
    n_sites = 0
    for i in range(len(small)):
        if site[i] >= 0:
            continue
        site[i] = n_sites
        stack = [i, ]
        while stack:
            j = stack.pop()
            near = np.where((np.hypot(*(centroids - centroids[j]).T) < cluster_distance) & (site < 0))[0]
            site[near] = n_sites
            stack.extend(near)
        n_sites += 1

    hucs_lines = shapely.ops.unary_union(list(hucs.linestrings.values()))
    for s in range(n_sites):
        members = small[site == s]
        loc = shapely.geometry.Point(centroids[site == s].mean(axis=0))
        n_river = sum(len(m2.conn[c]) > 3 for c in members)
        report.add('mesh', 'small-cells',
                   f'{len(members)} cell(s) smaller than {min_cell_area} (smallest {areas[members].min():.2f}), '
                   f'{loc.distance(hucs_lines):.1f} from the nearest HUC boundary; typically a HUC boundary '
                   'running very close to a river corridor',
                   location=loc,
                   details=dict(cells=[int(c) for c in members], areas=[float(a) for a in areas[members]],
                                river_cells=int(n_river)))


def _watershedDefect(report : DefectReport, hucs : gpd.GeoDataFrame, huc_id_col : str, err : Exception) -> None:
    """Report why Watershed() could not be built from the HUC polygons."""
    if all(hasattr(err, a) for a in ('i_p1', 'i_p2', 'inter')):
        # intersectAndSplit(): two HUCs whose shared boundary is not one line
        ids = list(hucs[huc_id_col]) if huc_id_col in hucs else list(hucs.index)
        a, b = ids[err.i_p1], ids[err.i_p2]
        parts = list(getattr(err.inter, 'geoms', [err.inter, ]))
        report.add('hucs', 'watershed',
                   f'HUCs {a} and {b} share {len(parts)} separate stretches of boundary; Watershed() needs '
                   'each pair of HUCs to share at most one', location=err.inter.representative_point(),
                   hucs=[a, b], details=dict(stretches=len(parts)))
    else:
        report.add('hucs', 'watershed', f'Watershed() failed on the HUC polygons: {type(err).__name__}: {err}')


def findDefects(hucs : gpd.GeoDataFrame,
                rivers : gpd.GeoDataFrame,
                reach_segment_target_length : float,
                huc_segment_target_length : Optional[float] = None,
                river_close_distance : float = 100.0,
                river_far_distance : float = 500.0,
                min_angle : float = 20.,
                junction_min_angle : float = 20.,
                snap_triple_junctions_tol : Optional[float] = None,
                huc_id_col : str = names.ID,
                mesh : bool = True,
                river_width : Optional[Callable] = None,
                min_cell_area : Optional[float] = None,
                cluster_distance : float = 30.,
                **tessalate_kwargs) -> DefectReport:
    """Run the Watershed Workflow pipeline on HUCs and rivers and report what it cannot fix.

    The stages and the defects each can report:

    - rivers: 'empty-geometry' (reaches createRivers() cannot handle),
      'createRivers', 'hydroseq-vs-geometry' (hydroseq links disagree with
      the geometry);
    - hucs: 'watershed' (the HUC polygons cannot be split into a Watershed);
    - simplify: 'multiple-crossing' (a reach crosses one HUC boundary
      several times, e.g. a stream digitized along a divide),
      'exterior-crossing', 'continuity', 'snapReachEndpoints',
      'sharp-angles';
    - outlets: 'multiple-outlets' (a HUC that the network leaves at more
      than one place);
    - mesh (only if mesh=True and no fatal defect above): 'corridor-overlap',
      'tessalateRiverAligned', and 'small-cells' (cells smaller than
      min_cell_area, grouped into sites cluster_distance apart).

    Parameters
    ----------
    hucs, rivers : gpd.GeoDataFrame
        The inputs, as they would be given to Watershed() and createRivers().
        They are not modified.
    reach_segment_target_length, huc_segment_target_length,
    river_close_distance, river_far_distance, min_angle,
    junction_min_angle, snap_triple_junctions_tol
        As for watershed_workflow.simplify().
    huc_id_col : str, optional
        Column of hucs identifying each HUC.
    mesh : bool, optional
        Whether to build the mesh too.
    river_width : Callable, optional
        As for tessalateRiverAligned(); by default the mean bankfull width of
        each stream order (mesh.river_mesh.computeWidthByOrder()).
    min_cell_area : float, optional
        Report cells smaller than this.  Not checked if None.
    cluster_distance : float, optional
        Small cells closer than this are reported as one site.
    tessalate_kwargs
        Passed to tessalateRiverAligned(), e.g. refine_min_angle.

    Returns
    -------
    DefectReport
    """
    report = DefectReport()
    if snap_triple_junctions_tol is None:
        snap_triple_junctions_tol = 3 * reach_segment_target_length
    report.params = dict(reach_segment_target_length=reach_segment_target_length,
                         huc_segment_target_length=huc_segment_target_length,
                         snap_triple_junctions_tol=snap_triple_junctions_tol, huc_id_col=huc_id_col)

    river_list = _buildRivers(report, rivers)
    if len(river_list) == 0:
        report.note('no river network could be built; later stages skipped')
        return report

    show = plt.show
    plt.show = lambda *args, **kwargs: None    # pipeline functions plot diagnostics before raising
    try:
        try:
            ws = watershed_workflow.Watershed(hucs.copy(deep=True))
        except Exception as err:
            _watershedDefect(report, hucs, huc_id_col, err)
            return report

        _simplifyIsolated(report, ws, river_list, reach_segment_target_length, huc_segment_target_length,
                          river_close_distance, river_far_distance, min_angle, junction_min_angle,
                          snap_triple_junctions_tol)
        _checkOutlets(report, ws, river_list, huc_id_col)

        if mesh:
            fatal = [d for d in report.defects if d.kind in FATAL_KINDS]
            if fatal:
                report.note(f'mesh stage skipped: fix the {len(fatal)} defect(s) of kind '
                            f'{sorted(set(d.kind for d in fatal))} first')
            else:
                if river_width is None:
                    rivers_ok = rivers[~(rivers.geometry.isna() | rivers.geometry.is_empty)]
                    river_width = watershed_workflow.mesh.river_mesh.widthByOrderFunction(
                        watershed_workflow.mesh.river_mesh.computeWidthByOrder(rivers_ok))
                _checkMesh(report, ws, river_list, river_width, min_cell_area, cluster_distance,
                           tessalate_kwargs)
    finally:
        plt.show = show
        plt.close('all')
    return report
