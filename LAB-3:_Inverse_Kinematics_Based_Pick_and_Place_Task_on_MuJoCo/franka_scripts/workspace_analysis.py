#!/usr/bin/env python3
"""Task/tool-down/dexterous workspace analysis for Franka Panda in Lab 3."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from kinematics import PandaKinematics, DEFAULT_SCENE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', default=str(DEFAULT_SCENE))
    p.add_argument('--fast', action='store_true', help='coarse grid for a quick check')
    p.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'results'))
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    r = PandaKinematics(a.scene)
    # Panda scene coordinates: scan around the table in front of the base.
    xs = np.arange(0.20, 0.751, 0.05 if a.fast else 0.035)
    ys = np.arange(-0.50, 0.501, 0.05 if a.fast else 0.035)
    table_z = r.table_top_z if r.table_top_z is not None else 0.80
    zs = np.array([table_z + v for v in (0.035, 0.08, 0.14, 0.22, 0.32)])
    yaws = np.linspace(0, 2*np.pi, 4 if a.fast else 8, endpoint=False)
    status = np.zeros((len(xs), len(ys), len(zs)), np.int8) # 0 unreachable, 1 position, 2 some yaw, 3 all yaw
    iterations = np.zeros_like(status, dtype=np.int16)
    for ix, x in enumerate(xs):
        print(f'Workspace scan x row {ix+1}/{len(xs)}', end='\r', flush=True)
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                target = np.array([x, y, z])
                ok, q, it, _ = r.ik(target, rotation=None)
                if not ok: continue
                status[ix, iy, iz] = 1; iterations[ix, iy, iz] = it
                successes = 0; warm = q
                for yaw in yaws:
                    ok_y, q_y, it_y, _ = r.ik(target, r.target_rotation(yaw), q_start=warm)
                    if ok_y:
                        successes += 1; warm = q_y
                if successes > 0: status[ix, iy, iz] = 2
                if successes == len(yaws): status[ix, iy, iz] = 3
    print('\nScan complete.')
    task = np.argwhere(status >= 1); tool = np.argwhere(status >= 2); dex = np.argwhere(status == 3)
    def pts(idx): return np.c_[xs[idx[:,0]], ys[idx[:,1]], zs[idx[:,2]]] if len(idx) else np.empty((0,3))
    task_pts, tool_pts, dex_pts = pts(task), pts(tool), pts(dex)
    print(f'Task workspace points: {len(task_pts)}')
    print(f'Tool-down workspace points (>=1 yaw): {len(tool_pts)}')
    print(f'Dexterous workspace points (all {len(yaws)} yaws): {len(dex_pts)}')
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(131, projection='3d')
    for cloud, label, color, size, alpha in [(task_pts,'Task workspace','tab:blue',3,.10),(tool_pts,'Tool-down','tab:orange',5,.3),(dex_pts,'Dexterous','tab:green',7,.8)]:
        if len(cloud): ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],s=size,alpha=alpha,label=label,c=color)
    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Franka Panda workspace'); ax.legend(fontsize=7)
    # Show two horizontal slices near the work surface and hover height.
    for ax_idx, offset in enumerate((0.04, 0.14), start=2):
        ax2 = fig.add_subplot(1,3,ax_idx); z = table_z + offset; iz = int(np.argmin(abs(zs-z)))
        for code, label, color in [(1,'Position reachable','tab:blue'),(2,'Some yaw','tab:orange'),(3,'All yaw','tab:green')]:
            ii = np.argwhere(status[:,:,iz] == code)
            if len(ii): ax2.scatter(xs[ii[:,0]],ys[ii[:,1]],s=22,c=color,label=label)
        if r.table_bounds:
            x0,x1,y0,y1=r.table_bounds; ax2.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='black',lw=1.5,label='Current table'))
        ax2.set(xlabel='X (m)',ylabel='Y (m)',title=f'Workspace slice z={zs[iz]:.3f} m'); ax2.axis('equal'); ax2.grid(True,alpha=.25); ax2.legend(fontsize=7)
    fig.tight_layout(); png=out/'workspace_analysis_franka.png'; npz=out/'workspace_analysis_franka.npz'
    fig.savefig(png,dpi=250,bbox_inches='tight'); plt.close(fig)
    np.savez_compressed(npz,x=xs,y=ys,z=zs,status=status,task_workspace=task_pts,tool_down_workspace=tool_pts,dexterous_workspace=dex_pts,table_top_z=table_z,yaws=yaws)
    print('Saved:',png); print('Saved:',npz)

if __name__ == '__main__': main()
