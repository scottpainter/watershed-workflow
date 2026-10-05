"""Apply ONE user-approved proposal (or record a rejection) and save it in the recipe.

usage: apply_step.py --scan s0 --defect 2 --proposal 0 --hucs H.gpkg --rivers R.gpkg
                     --recipe recipe.json --by NAME [--note TEXT] [--reject] [--out DIR]

H.gpkg / R.gpkg are the CURRENT state (the files the scan was made from); they are
overwritten with the result.  Only run this after the user has approved this proposal.
With --reject nothing is applied; the rejection is recorded so it is not offered again.
"""
import argparse, logging, os, pickle, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import geopandas as gpd
import watershed_workflow.repair as repair

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--scan', required=True); p.add_argument('--defect', type=int, required=True)
p.add_argument('--proposal', type=int, required=True)
p.add_argument('--hucs', required=True); p.add_argument('--rivers', required=True)
p.add_argument('--recipe', required=True); p.add_argument('--by', required=True)
p.add_argument('--note'); p.add_argument('--reject', action='store_true'); p.add_argument('--out', default='.')
a = p.parse_args()
logging.basicConfig(level=logging.ERROR)

rep, props = pickle.load(open(os.path.join(a.out, f'scan_{a.scan}.pkl'), 'rb'))
prop = props[a.defect][a.proposal]
recipe = repair.Recipe.load(a.recipe)
if a.reject:
    recipe.record(prop, 'rejected', by=a.by, note=a.note)
    print('recorded rejection:', prop.description)
else:
    H, R = gpd.read_file(a.hucs), gpd.read_file(a.rivers)
    H, R = prop.apply(H, R)
    H.to_file(a.hucs); R.to_file(a.rivers)
    recipe.record(prop, 'approved', by=a.by, note=a.note)
    print('applied:', prop.description)
    print(f'state: {len(H)} HUCs, {len(R)} reaches')
recipe.save(a.recipe)
print(f'recipe: {len(recipe.approved)} approved, {len(recipe.steps) - len(recipe.approved)} rejected steps')
