"""Build the river-aligned 2D mesh from repaired HUCs + rivers and report its quality.

usage: build_mesh2d.py --hucs H.gpkg --rivers R.gpkg [--out DIR] [--reach-L 60] [--huc-L 150]
                       [--snap-tj 50] [--refine-min-angle 32] [--min-cell-area 50] [--elevate]

Planform (always): createRivers(hydroseq) -> Watershed -> simplify() -> tessalateRiverAligned()
with corridor widths = mean bankfull width by stream order.  Writes to DIR:
  mesh2d_cells.gpkg         one polygon per cell with area [m2] and n_vertices (QGIS/notebook)
  mesh2d_hucs_simplified.gpkg, mesh2d_rivers_simplified.gpkg   inputs as simplified, before meshing
  mesh2d_stats.json         cell counts by type, smallest cells, cells under --min-cell-area
  mesh2d.pkl                the Mesh2D (if it pickles)
--elevate (slow, network): 3DEP 10 m DEM (py3dep static tiles), elevate, river profiles from
the DEM, burn-in, pit filling, HUC/outlet/river-corridor/stream-order regions, saved in
mesh2d.pkl.
"""
import argparse, json, logging, os, pickle, sys, time, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

import watershed_workflow
import watershed_workflow.mesh
import watershed_workflow.hydro.river as wr
import watershed_workflow.mesh.river_mesh as rm
import watershed_workflow.sources.standard_names as names

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--hucs', required=True); p.add_argument('--rivers', required=True)
p.add_argument('--out', default='.')
p.add_argument('--reach-L', type=float, default=60.); p.add_argument('--huc-L', type=float, default=150.)
p.add_argument('--snap-tj', type=float, default=50.); p.add_argument('--refine-min-angle', type=float, default=32.)
p.add_argument('--min-cell-area', type=float, default=50.)
p.add_argument('--elevate', action='store_true')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
out = lambda f: os.path.join(a.out, f)
logging.basicConfig(level=logging.WARNING, format='%(message)s')
t0 = time.time()

H, R = gpd.read_file(a.hucs), gpd.read_file(a.rivers)
width = rm.widthByOrderFunction(rm.computeWidthByOrder(R))
rivers = wr.createRivers(R.copy(), method='hydroseq')
hucs = watershed_workflow.Watershed(H.copy(deep=True))
watershed_workflow.simplify(hucs, rivers, a.reach_L, a.huc_L, snap_triple_junctions_tol=a.snap_tj)
# createRiversMesh adjusts HUC linestrings in place, after which the Watershed's polygons
# cannot be rebuilt: save the simplified inputs (and a copy for region labelling) now
hucs_premesh = hucs.deepcopy()
hucs.to_dataframe().to_file(out('mesh2d_hucs_simplified.gpkg'))
rdf = gpd.GeoDataFrame(pd.concat([r.to_dataframe() for r in rivers]).reset_index(drop=True), crs=R.crs)
rdf[[c for c in rdf.columns if c == 'geometry' or rdf[c].dtype != object or c in (names.ID, 'name', 'reachcode')]] \
    .to_file(out('mesh2d_rivers_simplified.gpkg'))

m2, areas, dists = watershed_workflow.tessalateRiverAligned(hucs, rivers, river_width=width,
                                                            refine_min_angle=a.refine_min_angle, diagnostics=True)
coords = np.asarray(m2.coords)[:, 0:2]
polys = [shapely.Polygon(coords[c]) for c in m2.conn]
cells = gpd.GeoDataFrame(dict(area=[g.area for g in polys], n_vertices=[len(c) for c in m2.conn]),
                         geometry=polys, crs=H.crs)
cells.to_file(out('mesh2d_cells.gpkg'))
small = cells[cells.area < a.min_cell_area].sort_values('area')
stats = dict(cells=len(cells), triangles=int((cells.n_vertices == 3).sum()), quads=int((cells.n_vertices == 4).sum()),
             pentagons_plus=int((cells.n_vertices > 4).sum()), area_km2=float(cells.area.sum() / 1e6),
             smallest_m2=[round(float(x), 2) for x in cells.area.nsmallest(5)],
             min_cell_area_criterion=a.min_cell_area, cells_below_criterion=len(small),
             cells_below_criterion_locations=[[round(g.centroid.x, 1), round(g.centroid.y, 1), round(float(ar), 2)]
                                              for g, ar in zip(small.geometry[:50], small.area[:50])],
             settings=dict(reach_L=a.reach_L, huc_L=a.huc_L, snap_triple_junctions_tol=a.snap_tj,
                           refine_min_angle=a.refine_min_angle))
print(f'mesh: {stats["cells"]} cells ({stats["triangles"]} tri, {stats["quads"]} quad, {stats["pentagons_plus"]} 5+), '
      f'{stats["area_km2"]:.2f} km2, smallest {stats["smallest_m2"][0]} m2, '
      f'{len(small)} cells < {a.min_cell_area} m2 ({time.time() - t0:.0f} s)')

if a.elevate:
    import py3dep
    dem = py3dep.get_dem(hucs_premesh.exterior.buffer(500).bounds, 10, crs=H.crs)   # static COG tiles, EPSG:5070
    watershed_workflow.elevate(m2, dem, method='linear')
    watershed_workflow.mesh.setProfileByDEM(rivers, dem)

    def burnIn(reach):          # [m], 1.22 ft * DA[mi2]^0.317 (WW Coweeta example)
        return 0.3048 * 1.22 * (reach[names.DRAINAGE_AREA] * 0.386102)**0.317
    watershed_workflow.mesh.conditionRiverMeshes(m2, rivers, network_burn_in_depth=burnIn)
    main = max(rivers, key=len)
    outlet = watershed_workflow.mesh.Edge(main['elems'][-1][0], main['elems'][-1][-1])
    m2, _ = watershed_workflow.mesh.conditionMesh(m2, preserved_pits=[c for c, cc in enumerate(m2.conn) if len(cc) > 3],
                                                  forced_outlet_edges=[outlet], epsilon=0.01)
    try:
        watershed_workflow.mesh.addWatershedAndOutletRegions(m2, hucs, outlet_width=300., exterior_outlet=True)
    except Exception as err:
        print(f'HUC regions from the meshed HUCs failed ({err}); using the simplified (pre-mesh) polygons')
        watershed_workflow.mesh.addWatershedAndOutletRegions(m2, hucs_premesh, outlet_width=300., exterior_outlet=True)
    watershed_workflow.mesh.addRiverCorridorRegions(m2, rivers)
    watershed_workflow.mesh.addStreamOrderRegions(m2, rivers)
    z = np.asarray(m2.coords)[:, 2]
    stats['elevation_m'] = [float(z.min()), float(z.max())]
    stats['labeled_sets'] = [(ls.setid, ls.entity, len(ls.ent_ids), ls.name) for ls in m2.labeled_sets]
    print(f'elevated: z {z.min():.1f}-{z.max():.1f} m, {len(m2.labeled_sets)} labeled sets ({time.time() - t0:.0f} s)')

json.dump(stats, open(out('mesh2d_stats.json'), 'w'), indent=1)
try:
    pickle.dump(m2, open(out('mesh2d.pkl'), 'wb'))
except Exception as err:
    print(f'mesh2d.pkl not written: {err}')
print('wrote mesh2d_cells.gpkg, mesh2d_stats.json' + (', mesh2d.pkl' if os.path.exists(out('mesh2d.pkl')) else ''))
