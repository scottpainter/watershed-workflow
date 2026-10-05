# Handing off to a Jupyter notebook

Many users run the WW workflow in a notebook (e.g. `examples/coweeta_stream_aligned_mesh.ipynb`).
The repair has to come first; give them cells like these to continue from the corrected data.
Use the same WW branch in the notebook kernel as for the repair.

## Load and check the corrected data

```python
import geopandas as gpd
import watershed_workflow
import watershed_workflow.hydro.river as wr
import watershed_workflow.diagnostics as diag
import watershed_workflow.mesh.river_mesh as rm

hucs_df = gpd.read_file('<name>_hucs_corrected.gpkg')
rivers_df = gpd.read_file('<name>_rivers_corrected.gpkg')

# should report 0 defects with the settings the repair was verified with
report = diag.findDefects(hucs_df, rivers_df, 60., 150., snap_triple_junctions_tol=50., mesh=False)
print(report.summary())
```

## Continue the standard workflow

```python
rivers = wr.createRivers(rivers_df, method='hydroseq')
watershed = watershed_workflow.Watershed(hucs_df)

# keep these settings, or rescan with yours: the repairs were checked against them
watershed_workflow.simplify(watershed, rivers, 60., 150., snap_triple_junctions_tol=50.)
# simplify() also updates tohuc/outlets from the geometry (raises MultipleOutletsError if a HUC
# has more than one outlet -- it should not after the repair)

widths = rm.widthByOrderFunction(rm.computeWidthByOrder(rivers_df))   # mean bankfull width by order
m2, areas, dists = watershed_workflow.tessalateRiverAligned(watershed, rivers, river_width=widths,
                                                            refine_min_angle=32., diagnostics=True)
```
Then elevation, conditioning, regions and properties as in the example notebook. Notes for the
notebook user:
- Do not call `watershed.to_dataframe()` / polygons after `tessalateRiverAligned()`; save them
  before (`watershed.deepcopy()`).
- If the notebook changes `simplify` settings (segment lengths, snapping tolerances), rerun
  `findDefects` with those settings -- defects depend on them.

## Running the repair loop in a notebook instead

The same loop works interactively:

```python
import watershed_workflow.repair as repair
recipe = repair.Recipe.load('recipe.json')
report = diag.findDefects(hucs_df, rivers_df, 60., 150., snap_triple_junctions_tol=50., mesh=False)
proposals = repair.proposeFixes(report, hucs_df, rivers_df, recipe=recipe)
for i, (d, ps) in enumerate(zip(report.defects, proposals)):
    print(i, d.kind, d.message)
    for j, p in enumerate(ps):
        print('   ', j, p.description, p.preview, p.caveats)

# after reviewing defect i (plot p.geometry over the HUCs/rivers) and choosing proposal j:
p = proposals[i][j]
hucs_df, rivers_df = p.apply(hucs_df, rivers_df)
recipe.record(p, 'approved', by='me', note='why')
recipe.save('recipe.json')
# then rescan before the next defect
```
Replay later from the original inputs: `repair.applyRecipe(hucs_raw, rivers_raw, 'recipe.json')`.
