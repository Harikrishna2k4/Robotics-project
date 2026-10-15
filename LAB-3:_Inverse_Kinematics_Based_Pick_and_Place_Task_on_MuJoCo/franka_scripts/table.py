#!/usr/bin/env python3
"""Grid-based table-height/feasible-area study for Franka Panda Lab 3.

This is a kinematic screening tool. It checks vertical TCP paths, yaw IK,
joint continuity, Jacobian conditioning, and a conservative skeleton clearance.
It is not a full mesh-collision proof.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from kinematics import PandaKinematics, DEFAULT_SCENE


def main():
    p=argparse.ArgumentParser(); p.add_argument('--scene',default=str(DEFAULT_SCENE)); p.add_argument('--fast',action='store_true'); p.add_argument('--output-dir',default=str(Path(__file__).resolve().parent/'results')); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); r=PandaKinematics(a.scene)
    current_z=r.table_top_z if r.table_top_z is not None else 0.80
    heights=np.array([current_z-0.08,current_z-0.04,current_z,current_z+0.04,current_z+0.08])
    step=0.07 if a.fast else 0.05; xs=np.arange(.22,.751,step); ys=np.arange(-.50,.501,step)
    yaws=np.linspace(0,2*np.pi,2 if a.fast else 4,endpoint=False); npath=3 if a.fast else 5
    grasp_offset=.025+.01 # cube half-height + a small TCP/fingertip offset; tune to your TCP definition
    approach_lift=.12; max_jump=.8; sigma_threshold=.025; clearance_threshold=.035
    status=np.zeros((len(heights),len(xs),len(ys)),np.int8); worst_sigma=np.zeros_like(status,dtype=float); worst_clear=np.zeros_like(status,dtype=float); max_jumps=np.zeros_like(status,dtype=float)
    for ih,h in enumerate(heights):
        print(f'Checking table height {h:.3f} m ({ih+1}/{len(heights)})')
        for ix,x in enumerate(xs):
            for iy,y in enumerate(ys):
                good=True; any_ik=False; cell_sigma=np.inf; cell_clear=np.inf; cell_jump=0.0
                levels=np.linspace(h+grasp_offset,h+grasp_offset+approach_lift,npath)
                for yaw in yaws:
                    warm=r.q_home.copy(); prev=None
                    for z in levels:
                        ok,q,it,J=r.ik([x,y,z],r.target_rotation(yaw),q_start=warm)
                        if not ok: good=False; break
                        any_ik=True; warm=q
                        jump=float(np.max(np.abs(q-prev))) if prev is not None else 0.0; prev=q.copy(); cell_jump=max(cell_jump,jump)
                        r.set_q(q); sv=np.linalg.svd(r.jacobian(),compute_uv=False); sig=float(sv.min()); cell_sigma=min(cell_sigma,sig)
                        clear=r.sampled_link_clearance(h, (float(xs.min()),float(xs.max()))); cell_clear=min(cell_clear,clear)
                        if jump>max_jump or sig<sigma_threshold or clear<clearance_threshold: good=False
                    if not good: break
                status[ih,ix,iy]=3 if good else (2 if any_ik else 0)
                worst_sigma[ih,ix,iy]=0 if not np.isfinite(cell_sigma) else cell_sigma
                worst_clear[ih,ix,iy]=0 if not np.isfinite(cell_clear) else cell_clear
                max_jumps[ih,ix,iy]=cell_jump
    # Select height with greatest number of good grid cells; tie-break by higher clearance and conditioning.
    counts=np.sum(status==3,axis=(1,2)); quality=np.array([np.mean(worst_clear[i][status[i]==3]) if counts[i] else -np.inf for i in range(len(heights))])
    best=int(max(range(len(heights)),key=lambda i:(counts[i],quality[i])))
    fig,axes=plt.subplots(1,len(heights),figsize=(4*len(heights),5),squeeze=False); axes=axes.ravel()
    for i,h in enumerate(heights):
        ax=axes[i]
        for code,label,color in [(0,'No complete IK path','lightgray'),(2,'Some IK, fails margin','tab:orange'),(3,'Feasible path','tab:green')]:
            ids=np.argwhere(status[i]==code)
            if len(ids): ax.scatter(xs[ids[:,0]],ys[ids[:,1]],s=18,c=color,label=label)
        ax.set_title(f'h={h:.3f} m\nfeasible={counts[i]} cells'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.axis('equal'); ax.grid(alpha=.2)
        if i==best and counts[i]:
            ids=np.argwhere(status[i]==3); xlo,xhi=xs[ids[:,0]].min(),xs[ids[:,0]].max(); ylo,yhi=ys[ids[:,1]].min(),ys[ids[:,1]].max()
            ax.add_patch(Rectangle((xlo,ylo),xhi-xlo,yhi-ylo,fill=False,ec='red',lw=2,label='Feasible bounds'))
        ax.legend(fontsize=6,loc='best')
    fig.suptitle('Franka Panda table-height feasibility (kinematic screening)'); fig.tight_layout()
    png=out/'table_design_franka.png'; npz=out/'table_design_franka.npz'; fig.savefig(png,dpi=250,bbox_inches='tight'); plt.close(fig)
    np.savez_compressed(npz,heights=heights,x=xs,y=ys,status=status,worst_sigma=worst_sigma,worst_clearance=worst_clear,max_joint_jump=max_jumps,selected_height=heights[best])
    print('\nSummary by candidate table top:')
    for i,h in enumerate(heights): print(f'  z={h:.3f} m: {counts[i]} / {len(xs)*len(ys)} cells pass all tested yaw/path checks')
    print(f'Selected candidate by grid count: z={heights[best]:.3f} m')
    print('IMPORTANT: compare against the current scene table height and physical requirements; this is not a collision-certified design.')
    print('Saved:',png); print('Saved:',npz)

if __name__=='__main__': main()
