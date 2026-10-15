# Pick-and-place batch evaluation

- Episodes per method: **30** (seeds 0..29, identical for every method)
- Methods: mink, dls, qp
- Date: 2026-10-15 15:58:36

## Metrics

| Metric | mink | dls | qp |
|---|---|---|---|
| Success rate | 28/30 (93%) | 29/30 (97%) | 29/30 (97%) |
| Success 95% CI (Wilson) | [79, 98]% | [83, 99]% | [83, 99]% |
| Mean IK time per episode (s) | 0.978 +/- 0.595 | 0.697 +/- 0.245 | 0.746 +/- 0.311 |
| Mean IK time per call (ms) | 69.8 | 49.8 | 53.3 |
| IK iterations to converge (mean +/- std, max) | 4.4 +/- 0.2, 12 | 18.1 +/- 3.2, 217 | 11.9 +/- 1.8, 101 |
| Position-only fallback calls (fraction) | 0.23 | 0.25 | 0.24 |
| Min clearance, worst episode (mm) | 6.1 | 2.4 | 5.5 |
| Min clearance, mean over episodes (mm) | 10.2 | 9.6 | 10.0 |
| Episodes with collision | 0 | 0 | 0 |
| Mean final placement error, successes (mm) | 14.0 | 9.7 | 13.5 |

## Constraint violations

Joint speed limit inside IK: 1.5 rad/s; joint range from the model. A violation is an IK iteration whose step exceeds the speed limit or leaves the joint range.

| Quantity | mink | dls | qp |
|---|---|---|---|
| IK iterations over speed limit (episodes affected) | 10299 (30) | 0 (0) | 3 (3) |
| IK iterations outside joint range (episodes affected) | 0 (0) | 0 (0) | 0 (0) |
| Peak joint speed requested by IK (rad/s) | 193.1 | 1.5 | 1.5 |
| Simulation steps over speed limit | 0 | 5 | 11 |
| Peak simulated joint speed (rad/s) | 1.02 | 1.06 | 1.10 |
| Max simulated joint-limit excess (rad) | 0.0000 | 0.0004 | 0.0000 |

## Failure analysis

**mink**

| Failure | Count | Seeds | Evidence |
|---|---|---|---|
| grasp_slip@carry | 2 | 12, 15 | `failures/mink_seed012.png`, `failures/mink_seed015.png` |

**dls**

| Failure | Count | Seeds | Evidence |
|---|---|---|---|
| grasp_slip@carry | 1 | 28 | `failures/dls_seed028.png` |

**qp**

| Failure | Count | Seeds | Evidence |
|---|---|---|---|
| grasp_slip@carry | 1 | 13 | `failures/qp_seed013.png` |

### Suggested mitigations

- **grasp_slip**: Raise CUBE_FRICTION or grip force, lower CARRY_SPEED, grasp lower (GRASP_Z_OFFSET) or pick the yaw so the pads meet flat faces.

## Per-seed outcomes (same seed = same cube pose)

| Seed | Cube x,y,yaw | mink | dls | qp |
|---|---|---|---|---|
| 0 | 0.49, 0.11, -41 | ok | ok | ok |
| 1 | 0.44, 0.40, -32 | ok | ok | ok |
| 2 | 0.33, 0.13, 28 | ok | ok | ok |
| 3 | 0.26, 0.10, 27 | ok | ok | ok |
| 4 | 0.63, 0.21, 43 | ok | ok | ok |
| 5 | 0.57, 0.34, 1 | ok | ok | ok |
| 6 | 0.45, 0.14, -12 | ok | ok | ok |
| 7 | 0.49, 0.38, 25 | ok | ok | ok |
| 8 | 0.36, 0.41, -16 | ok | ok | ok |
| 9 | 0.59, 0.12, 9 | ok | ok | ok |
| 10 | 0.63, 0.09, 30 | ok | ok | ok |
| 11 | 0.28, 0.21, 9 | ok | ok | ok |
| 12 | 0.33, 0.40, -28 | FAIL grasp_slip@carry | ok | ok |
| 13 | 0.59, 0.36, 28 | ok | ok | FAIL grasp_slip@carry |
| 14 | 0.58, 0.15, 18 | ok | ok | ok |
| 15 | 0.52, 0.34, -14 | FAIL grasp_slip@carry | ok | ok |
| 16 | 0.46, 0.18, -37 | ok | ok | ok |
| 17 | 0.58, 0.07, 5 | ok | ok | ok |
| 18 | 0.39, 0.30, -20 | ok | ok | ok |
| 19 | 0.40, 0.39, -20 | ok | ok | ok |
| 20 | 0.34, 0.19, -34 | ok | ok | ok |
| 21 | 0.56, 0.25, 19 | ok | ok | ok |
| 22 | 0.38, 0.08, -37 | ok | ok | ok |
| 23 | 0.52, 0.27, -33 | ok | ok | ok |
| 24 | 0.36, 0.17, 7 | ok | ok | ok |
| 25 | 0.29, 0.00, -26 | ok | ok | ok |
| 26 | 0.43, 0.10, -39 | ok | ok | ok |
| 27 | 0.52, 0.13, -34 | ok | ok | ok |
| 28 | 0.59, 0.37, 24 | ok | FAIL grasp_slip@carry | ok |
| 29 | 0.24, 0.21, 2 | ok | ok | ok |

## Paired comparison (same seeds)

| A vs B | both ok | only A | only B | both fail |
|---|---|---|---|---|
| mink vs dls | 27 | 1 | 2 | 0 |
| mink vs qp | 27 | 1 | 2 | 0 |
| dls vs qp | 28 | 1 | 1 | 0 |
