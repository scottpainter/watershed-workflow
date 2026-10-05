"""Scan HUCs + rivers for the defects Watershed Workflow will not fix, and list proposed fixes.

usage: scan.py --hucs H.gpkg --rivers R.gpkg --tag s0 [--recipe recipe.json] [--mesh]
               [--min-cell-area 50] [--reach-L 60] [--huc-L 150] [--snap-tj 50] [--out DIR]

Prints every defect with its numbered proposals (preview and caveats) and writes
  scan_<tag>.pkl            (report, proposals) for apply_step.py / plot_defect.py
  scan_<tag>_defects.gpkg   one point per defect, for QGIS or a notebook
Nothing is changed.  With --recipe, proposals already rejected there are left out.
--mesh also builds the mesh (minutes) and reports cells smaller than --min-cell-area.
"""
import argparse, logging, os, pickle, sys, time, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import geopandas as gpd
import watershed_workflow.diagnostics as diag
import watershed_workflow.repair as repair

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--hucs', required=True); p.add_argument('--rivers', required=True)
p.add_argument('--tag', required=True); p.add_argument('--out', default='.')
p.add_argument('--recipe'); p.add_argument('--mesh', action='store_true')
p.add_argument('--min-cell-area', type=float, default=50.)
p.add_argument('--reach-L', type=float, default=60.); p.add_argument('--huc-L', type=float, default=150.)
p.add_argument('--snap-tj', type=float, default=50.); p.add_argument('--refine-min-angle', type=float, default=32.)
a = p.parse_args()
logging.basicConfig(level=logging.ERROR, format='%(message)s')

t = time.time()
H, R = gpd.read_file(a.hucs), gpd.read_file(a.rivers)
rep = diag.findDefects(H, R, a.reach_L, a.huc_L, snap_triple_junctions_tol=a.snap_tj, mesh=a.mesh,
                       min_cell_area=a.min_cell_area, refine_min_angle=a.refine_min_angle)
recipe = repair.Recipe.load(a.recipe) if a.recipe else None
props = repair.proposeFixes(rep, H, R, recipe=recipe)

print(rep.summary())
for i, (d, ps) in enumerate(zip(rep.defects, props)):
    print(f'\n#{i} [{d.stage}/{d.kind}] {d.message}')
    if not ps:
        print('   (no proposal: discuss options with the user; check how WW handles it first)')
    for j, pr in enumerate(ps):
        pv = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in pr.preview.items()
              if k != 'detached_fragments'}
        print(f'   {i}.{j} -> {pr.description}\n         preview {pv}' + (f'\n         caveats {pr.caveats}' if pr.caveats else ''))
print(f'\n({time.time() - t:.0f} s)')

rep.mesh = None
pickle.dump((rep, props), open(os.path.join(a.out, f'scan_{a.tag}.pkl'), 'wb'))
if rep.defects:
    rep.to_dataframe(crs=H.crs).to_file(os.path.join(a.out, f'scan_{a.tag}_defects.gpkg'))
