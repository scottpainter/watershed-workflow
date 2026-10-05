"""Plot a defect and its proposals, for the user to review.

usage: plot_defect.py --scan s0 --defect 2 --hucs H.gpkg --rivers R.gpkg [--proposals 0 1]
                      [--width 1500] [--out DIR]

Writes defect_<scan>_<defect>.png: first column, the HUCs (labelled with the last 4 digits
of their IDs, computed from the data -- never place labels by hand) and reaches around the
defect, with the reaches named in the defect in red; then one column per proposal with its
geometry (area moved / cut / piece / dropped reaches) overlaid.  Top row: overview; bottom
row: zoom of --zoom m around the defect location.
"""
import argparse, os, pickle, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely
import textwrap

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--scan', required=True); p.add_argument('--defect', type=int, required=True)
p.add_argument('--hucs', required=True); p.add_argument('--rivers', required=True)
p.add_argument('--proposals', type=int, nargs='*', default=None)
p.add_argument('--width', type=float, default=None, help='half-width of the overview [m]')
p.add_argument('--zoom', type=float, default=300., help='half-width of the zoomed row [m]')
p.add_argument('--id-col', default='ID'); p.add_argument('--out', default='.')
a = p.parse_args()

rep, props = pickle.load(open(os.path.join(a.out, f'scan_{a.scan}.pkl'), 'rb'))
d = rep.defects[a.defect]
ps = props[a.defect] if a.proposals is None else [props[a.defect][i] for i in a.proposals]
H, R = gpd.read_file(a.hucs), gpd.read_file(a.rivers)
ids = [str(r) for r in d.reaches]
named = R[R[a.id_col].astype(str).isin(ids) | R[a.id_col].astype(str).isin([i.rstrip('ab') for i in ids])]

focus = [g for g in [d.location] if g is not None] + list(named.geometry)
for pr in ps:
    if pr.geometry is not None:
        focus += list(pr.geometry.geometry)
if not focus:
    focus = list(H[H[a.id_col].isin(d.hucs)].geometry)
c = shapely.unary_union(focus).centroid
w = a.width or max(500., 0.6 * max(shapely.unary_union(focus).bounds[2] - shapely.unary_union(focus).bounds[0],
                                    shapely.unary_union(focus).bounds[3] - shapely.unary_union(focus).bounds[1]))
box = shapely.box(c.x - w, c.y - w, c.x + w, c.y + w)

n = 1 + len(ps)
zc = d.location if d.location is not None else c
fig, axs = plt.subplots(2, n, figsize=(7.5 * n, 15), squeeze=False)
colors = {'area moved': ('orange', 0.6), 'piece': ('violet', 0.35), 'cut': ('green', 1.0),
          'reaches dropped': ('red', 1.0), 'divide before': ('black', 1.0), 'confluence': ('black', 1.0)}
for row, (cx, cy, hw) in enumerate(((c.x, c.y, w), (zc.x, zc.y, a.zoom))):
    view = shapely.box(cx - hw, cy - hw, cx + hw, cy + hw)
    for k in range(n):
        ax = axs[row][k]
        sub = H[H.intersects(view)]
        sub.plot(ax=ax, column=a.id_col, cmap='Pastel1', alpha=0.5, edgecolor='k', linewidth=1.0)
        for _, r in sub.iterrows():
            q = r.geometry.intersection(view).representative_point()
            ax.annotate(str(r[a.id_col])[-4:], (q.x, q.y), fontsize=12, weight='bold', ha='center')
        R[R.intersects(view)].plot(ax=ax, color='tab:blue', linewidth=0.9)
        named.plot(ax=ax, color='tab:red', linewidth=2.5)
        if d.location is not None:
            ax.plot(d.location.x, d.location.y, 'o', mfc='none', mec='k', ms=12, mew=2)
        if k == 0:
            title = f'#{a.defect} [{d.kind}] {", ".join(d.hucs)}: {d.message}'
        else:
            pr = ps[k - 1]
            if pr.geometry is not None:
                for _, g in pr.geometry.iterrows():
                    col, al = colors.get(g['role'], ('magenta', 0.8))
                    gpd.GeoSeries([g.geometry], crs=H.crs).plot(ax=ax, color=col, alpha=al, linewidth=2)
            j = k - 1 if a.proposals is None else a.proposals[k - 1]
            title = f'proposal {a.defect}.{j}: {pr.description}'
        if row == 0:
            ax.set_title('\n'.join(textwrap.wrap(title[:400], 80)), fontsize=9)
        else:
            ax.set_title(f'zoom ({2 * hw:.0f} m across)', fontsize=9)
        ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fn = os.path.join(a.out, f'defect_{a.scan}_{a.defect}.png')
plt.savefig(fn, dpi=100)
print('wrote', fn)
