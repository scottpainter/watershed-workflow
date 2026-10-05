"""Download HUCs (WBD) and NHDPlus flowlines for a HUC, build the river network and prune it.

usage: fetch_and_prune.py --huc 04100006 [--level 12] [--prune-km2 2.0] [--keep-braids]
                          [--remove-diversions] [--out DIR]

Writes to DIR (default: current directory):
  hucs_raw.gpkg            HUCs at --level inside --huc (WBD, EPSG:5070)
  reaches_raw.gpkg         NHDPlus MR v2.1 flowlines with VAAs and bankfull properties
  catchments_raw.gpkg      their NHDPlus catchment polygons
  rivers_pruned.gpkg       network after pruning (the input to the repair loop)
  recipe.json              an empty repair recipe whose metadata records these settings

Run from a working directory: HyRiver caches downloads in ./cache.
"""
import argparse, json, logging, os, time, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import pandas as pd
import geopandas as gpd

import watershed_workflow
import watershed_workflow.sources as sources
import watershed_workflow.hydro.river as wr
import watershed_workflow.repair as repair
import watershed_workflow.sources.standard_names as names

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--huc', required=True, help='HUC code defining the domain, e.g. 04100006')
p.add_argument('--level', type=int, default=12, help='HUC level of the sub-watersheds (default 12)')
p.add_argument('--prune-km2', type=float, required=True,
               help='ABSOLUTE contributing-area threshold [km2]: reaches draining less are removed')
p.add_argument('--keep-braids', action='store_true', help='keep braided divergences (default: remove them)')
p.add_argument('--remove-diversions', action='store_true', help='also remove diversions')
p.add_argument('--out', default='.')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
out = lambda f: os.path.join(a.out, f)
logging.basicConfig(level=logging.WARNING, format='%(message)s')
crs = watershed_workflow.crs.default_crs

t = time.time()
wbd = sources.ManagerWBD()
wbd.setLevel(a.level)
hucs = wbd.getShapesByID([a.huc], out_crs=crs)
hucs.to_file(out('hucs_raw.gpkg'))
print(f'WBD: {len(hucs)} HUC{a.level}s, {hucs.area.sum() / 1e6:.1f} km2 ({time.time() - t:.0f} s)')

t = time.time()
nhd = sources.ManagerNHD('NHDPlus MR v2.1', catchments=True)
reaches = nhd.getShapesByGeometry(hucs.union_all(), crs, out_crs=crs)
geom_cols = [c for c in reaches.columns if c != 'geometry' and isinstance(reaches[c], gpd.GeoSeries)]
if names.CATCHMENT in reaches:
    cas = gpd.GeoDataFrame(reaches[[names.ID]], geometry=gpd.GeoSeries(reaches[names.CATCHMENT], crs=reaches.crs))
    cas[~cas.geometry.isna()].to_file(out('catchments_raw.gpkg'))
reaches = reaches.drop(columns=geom_cols)
reaches.to_file(out('reaches_raw.gpkg'))
print(f'NHDPlus MR v2.1: {len(reaches)} flowlines ({time.time() - t:.0f} s)')

rivers = wr.createRivers(reaches.copy(), method='hydroseq')
print(f'network: {len(rivers)} trees by hydroseq, sizes {sorted((len(r) for r in rivers), reverse=True)[:6]}')
rivers = watershed_workflow.reduceRivers(rivers, prune_by_area=a.prune_km2,
                                         remove_braided_divergences=not a.keep_braids,
                                         remove_diversions=a.remove_diversions)
for r in rivers:
    r.resetDataFrame()                       # drops rows of merged/removed reaches
# the index is WW-internal ('new_preorder_index'); createRivers() rejects a file that carries it
pruned = gpd.GeoDataFrame(pd.concat([r.df for r in rivers]).reset_index(drop=True), crs=reaches.crs)
pruned = pruned[[c for c in pruned.columns if not isinstance(pruned[c], gpd.GeoSeries) or c == 'geometry']]
pruned.to_file(out('rivers_pruned.gpkg'))
print(f'pruned at {a.prune_km2} km2 (braids removed: {not a.keep_braids}, diversions removed: '
      f'{a.remove_diversions}): {len(rivers)} trees, {len(pruned)} reaches')

repair.Recipe(metadata=dict(
    domain=f'HUC {a.huc}, HUC{a.level} sub-watersheds',
    hucs=f'WBD via ManagerWBD().setLevel({a.level}).getShapesByID(["{a.huc}"]) -> hucs_raw.gpkg',
    rivers=(f"NHDPlus MR v2.1 via ManagerNHD(catchments=True) -> reaches_raw.gpkg; createRivers(method='hydroseq'); "
            f'reduceRivers(prune_by_area={a.prune_km2} km2, remove_braided_divergences={not a.keep_braids}, '
            f'remove_diversions={a.remove_diversions}) -> rivers_pruned.gpkg'),
    replay='repair.applyRecipe(hucs_raw, rivers_pruned, recipe.json)',
)).save(out('recipe.json'))
print('wrote', ', '.join(f for f in ('hucs_raw.gpkg', 'reaches_raw.gpkg', 'catchments_raw.gpkg',
                                       'rivers_pruned.gpkg', 'recipe.json')))
