#!/usr/bin/env python3
"""Batch-run the existing Franka pickandplace.py and save per-episode output/logs.

This runner deliberately uses the project's already-working pickandplace.py
instead of duplicating its controller. It records episode success and wall time
from its console output. Detailed IK iteration/clearance metrics require those
metrics to be emitted by pickandplace.py itself.
"""
import argparse, csv, json, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
DEFAULT_SCENE=PROJECT/'robot_descriptions'/'franka'/'pickandplace_scene.xml'
PICKPLACE=HERE/'pickandplace.py'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=25)
    p.add_argument('--scene',default=str(DEFAULT_SCENE))
    p.add_argument('--out',default=None)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--speed',type=float,default=1.0)
    a=p.parse_args()
    if a.episodes<1: p.error('--episodes must be >= 1')
    if not PICKPLACE.exists(): raise FileNotFoundError(f'Expected existing script: {PICKPLACE}')
    out=Path(a.out) if a.out else PROJECT/'results'/f"franka_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(PICKPLACE),'--scene',str(Path(a.scene).resolve()),'--episodes',str(a.episodes)]
    if a.viewer: cmd.append('--viewer')
    # Existing pickandplace.py may not have a speed argument; don't pass one unless viewer timing is configured there.
    started=time.perf_counter()
    print('Running:', ' '.join(cmd)); print('Output directory:',out)
    proc=subprocess.run(cmd,cwd=PROJECT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    elapsed=time.perf_counter()-started
    raw=out/'batch_console.log'; raw.write_text(proc.stdout,encoding='utf-8')
    # Parse episode headers and final result lines; keep complete console text for traceability.
    lines=proc.stdout.splitlines(); episodes=[]; current=None
    header=re.compile(r'^Episode\s+(\d+)\s*:',re.I)
    for line in lines:
        m=header.search(line.strip())
        if m:
            if current is not None: episodes.append(current)
            current={'episode':int(m.group(1)),'seed':int(m.group(1)),'success':False,'failure_reason':'unknown','console_lines':[]}
        if current is not None:
            current['console_lines'].append(line)
            if 'Result: SUCCESS' in line or re.search(r'\bSUCCESS\b',line):
                current['success']=True; current['failure_reason']=''
            if 'Result:' in line and 'SUCCESS' not in line:
                current['failure_reason']=line.split('Result:',1)[-1].strip()
            if 'grasp failed' in line.lower(): current['failure_reason']='grasp_failed'
            if 'IK failed' in line or 'IK infeasible' in line: current['failure_reason']='ik_infeasible'
    if current is not None: episodes.append(current)
    # Include a fallback record if the source script's output doesn't expose per-episode headers.
    if not episodes:
        success=bool(re.search(r'Summary:\s*\d+\s*/\s*\d+\s+successful',proc.stdout,re.I))
        episodes=[{'episode':i,'seed':i,'success':False,'failure_reason':'see batch_console.log','console_lines':[]} for i in range(a.episodes)]
        if success:
            # Preserve only the aggregate; do not invent which seeds passed.
            for e in episodes: e['failure_reason']='per-episode result not printed by pickandplace.py'
    records=[]
    for e in episodes:
        records.append({k:v for k,v in e.items() if k!='console_lines'})
        (out/f"episode_{e['episode']:03d}.log").write_text('\n'.join(e['console_lines'])+'\n',encoding='utf-8')
    with (out/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['episode','seed','success','failure_reason']); w.writeheader(); w.writerows(records)
    summary={'requested_episodes':a.episodes,'parsed_episodes':len(records),'successes':sum(bool(e['success']) for e in records),'success_rate':sum(bool(e['success']) for e in records)/len(records) if records else 0.0,'wall_time_s':round(elapsed,3),'return_code':proc.returncode,'command':cmd,'note':'Per-episode IK iterations and clearance are not inferred; see batch_console.log. Add instrumentation to pickandplace.py for those metrics.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'summary.md').write_text('# Franka batch evaluation\n\n'+ '\n'.join(f'- **{k}:** {v}' for k,v in summary.items())+'\n',encoding='utf-8')
    print(proc.stdout)
    print('\nBatch summary:',json.dumps(summary,indent=2))
    print('Saved logs and summary to',out)
    raise SystemExit(proc.returncode)

if __name__=='__main__': main()
