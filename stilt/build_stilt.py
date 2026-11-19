"""Stilt-walker (Design C): 30 m four-legged, two-armed strider. Hard-surface build, rig, animations, bake, export.
Run:  blender.exe --background --python build_stilt.py -- [--no-bake] [--no-export] [--tex 4096] [--wear 0..1] [--name stilt_worn]
Everything is rebuilt from scratch on every run. Units are metres, +X forward, +Y left, +Z up.
"""
import bpy, bmesh, sys, os, json, time
import numpy as np
from math import sin, cos, pi, radians, degrees, sqrt, atan2, hypot, floor
from mathutils import Vector, Matrix, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = 'stilt'
NAME = ARGS[ARGS.index('--name') + 1] if '--name' in ARGS else OUT  # fork: output name (blend/glb/fbx/info/textures)
TEX_N = int(ARGS[ARGS.index('--tex') + 1]) if '--tex' in ARGS else 4096
T0 = time.time()


def V(*a):
    return Vector(a)


XAX, YAX, ZAX = V(1, 0, 0), V(0, 1, 0), V(0, 0, 1)
I4 = Matrix.Identity(4)
S_M = Matrix.Diagonal((1, -1, 1, 1))


def frame(origin, x=None, y=None, z=None):
    if x is None:
        x = y.cross(z)
    if y is None:
        y = z.cross(x)
    if z is None:
        z = x.cross(y)
    m = Matrix.Identity(4)
    for i, a in enumerate((x.normalized(), y.normalized(), z.normalized())):
        m[0][i], m[1][i], m[2][i] = a.x, a.y, a.z
    m[0][3], m[1][3], m[2][3] = origin.x, origin.y, origin.z
    return m


def T(*a):
    return Matrix.Translation(a[0] if len(a) == 1 else a)


def Rx(a):
    return Matrix.Rotation(a, 4, 'X')


def Ry(a):
    return Matrix.Rotation(a, 4, 'Y')


def Rz(a):
    return Matrix.Rotation(a, 4, 'Z')


def ss(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def spl(xs, ys, x):
    """Cubic Hermite through (xs, ys)."""
    n = len(xs)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = max(j for j in range(n - 1) if xs[j] <= x)
    h = xs[i + 1] - xs[i]
    t = (x - xs[i]) / h

    def m(j):
        if j == 0:
            return (ys[1] - ys[0]) / (xs[1] - xs[0])
        if j == n - 1:
            return (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return (ys[j + 1] - ys[j - 1]) / (xs[j + 1] - xs[j - 1])
    return ((2 * t ** 3 - 3 * t * t + 1) * ys[i] + (t ** 3 - 2 * t * t + t) * h * m(i)
            + (-2 * t ** 3 + 3 * t * t) * ys[i + 1] + (t ** 3 - t * t) * h * m(i + 1))


# ----------------------------------------------------------------------------------------------
# dimensions (see PARTS.md)
# ----------------------------------------------------------------------------------------------
PEL = V(0, 0, 21.1)
L1, L2, L3 = 8.55, 7.55, 5.05            # thigh, shin, tarsus
HIP_DROP = 1.35                       # abduction pin above the hip drum
FOOT_Z = 0.60
R_HIP, R_KNEE, R_ANK = 0.95, 0.75, 0.50
W_HIP, W_KNEE, W_ANK = 0.70, 0.60, 0.50
LEGS = {}
for _k, _s, _sd, _drum, _foot in (('FL', 1, 1, V(1.6, 1.9, 21.3), V(5.0, 2.5, FOOT_Z)), ('FR', 1, -1, V(1.6, -1.9, 21.3), V(5.0, -2.5, FOOT_Z)),
                                  ('RL', -1, 1, V(-2.0, 2.2, 20.9), V(-5.4, 4.6, FOOT_Z)), ('RR', -1, -1, V(-2.0, -2.2, 20.9), V(-5.4, -4.6, FOOT_Z))):
    LEGS[_k] = dict(s=_s, sd=_sd, piv=_drum + V(0, 0, HIP_DROP) - PEL, foot=_foot, mir=(_s * _sd == -1))
LU, LF = 7.0, 6.8                     # upper arm, forearm
SH_DROP = 1.0
R_SH, R_ELB, R_WR = 0.70, 0.50, 0.36
W_SH, W_ELB, W_WR = 0.50, 0.42, 0.36
ARMS = {}
for _sd, _sg in (('L', 1), ('R', -1)):
    _drum = V(2.6, 3.3 * _sg, 23.3)
    _du = V(-0.12, 0.085 * _sg, -1.0).normalized()
    _df = V(0.24, 0.085 * _sg, -1.0).normalized()
    ARMS[_sd] = dict(sd=_sg, piv=_drum + V(0, 0, SH_DROP) - PEL, wrist=_drum + _du * LU + _df * LF - PEL, mir=(_sg == -1))
FING = dict(a=(1.5, 1.2, 0.9), b=(1.5, 1.2, 0.9), c=(1.3, 1.1, 0.9))
FREST = dict(a=(10, 14, 12), b=(10, 14, 12), c=(8, 14, 12))          # rest curl per segment, degrees
PALM = 1.25
NECK = [V(2.9, 0, 24.4), V(3.2, 0, 25.45), V(3.7, 0, 26.45), V(4.35, 0, 27.35), V(5.05, 0, 28.0)]
HYAW, HPIV, SENS = V(5.8, 0, 28.5), V(5.8, 0, 28.85), V(8.0, 0, 28.9)
TOE_ANG = dict(a=38.0, b=-38.0, c=180.0)
TOE_R, TOE_DZ = 0.72, -0.33

# ----------------------------------------------------------------------------------------------
# solver: a pose state -> world matrix of every bone
# ----------------------------------------------------------------------------------------------
PARENT = {'root': None, 'pelvis': 'root'}
FKD = {}                 # fk bones: name -> local rest matrix relative to the parent


def fk(name, parent, loc, mir=False):
    PARENT[name] = parent
    FKD[name] = (S_M @ loc @ S_M) if mir else loc


for _k, _lg in LEGS.items():
    for _b, _p in (('hip', 'pelvis'), ('thigh', 'hip'), ('shin', 'thigh'), ('tarsus', 'shin'), ('foot', 'tarsus')):
        PARENT['%s_%s' % (_b, _k)] = _p if _p == 'pelvis' else '%s_%s' % (_p, _k)
    for _t, _a in TOE_ANG.items():
        _a = radians(_a)
        fk('toe_%s_%s' % (_t, _k), 'foot_' + _k, T(TOE_R * cos(_a), TOE_R * sin(_a), TOE_DZ) @ Rz(_a), _lg['mir'])
for _sd, _ar in ARMS.items():
    for _b, _p in (('shoulder', 'pelvis'), ('upperarm', 'shoulder'), ('forearm', 'upperarm'), ('hand', 'forearm')):
        PARENT['%s_%s' % (_b, _sd)] = _p if _p == 'pelvis' else '%s_%s' % (_p, _sd)
    _base = dict(a=T(-0.02, 0.40, -PALM) @ Rx(radians(7)) @ Rz(pi), b=T(-0.02, -0.40, -PALM) @ Rx(radians(-7)) @ Rz(pi),
                 c=T(0.32, 0.0, -0.92) @ Ry(radians(-30)))
    for _f in 'abc':
        _par = 'hand_' + _sd
        _loc = _base[_f]
        for _i in range(3):
            _n = 'finger_%s%d_%s' % (_f, _i + 1, _sd)
            fk(_n, _par, _loc @ Ry(radians(FREST[_f][_i])), _ar['mir'])
            _par, _loc = _n, T(0, 0, -FING[_f][_i])
fk('neck_01', 'pelvis', T(NECK[0] - PEL))
for _i in range(1, 5):
    fk('neck_%02d' % (_i + 1), 'neck_%02d' % _i, T(NECK[_i] - NECK[_i - 1]))
fk('head_yaw', 'neck_05', T(HYAW - NECK[4]))
fk('head', 'head_yaw', T(HPIV - HYAW))
fk('sensor', 'head', T(SENS - HPIV))
FK_ORDER = list(FKD)
WARN = set()
CUR = ['rest']


def two_bone(Tp, Tq, l1, l2, bend):
    d0 = hypot(Tp, Tq)
    d = min(max(d0, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    ux, uy = Tp / d0, Tq / d0
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = sqrt(max(l1 * l1 - a * a, 0.0))
    return (ux * a - bend * uy * h, uy * a + bend * ux * h), (ux * d, uy * d), d0 <= l1 + l2 - 1e-3


def leg2d(Tp, Tq, extra):
    """Three-segment zigzag: the knee sits ahead of the hip-foot line, the ankle behind it by 0.6 of that."""
    d0 = hypot(Tp, Tq)
    nx, ny = -Tq / d0, Tp / d0

    def ev(tau):
        Ap, Aq = Tp - L3 * sin(tau), Tq + L3 * cos(tau)
        K, A, ok = two_bone(Ap, Aq, L1, L2, 1)
        return K, A, 0.6 * (K[0] * nx + K[1] * ny) + (A[0] * nx + A[1] * ny) + L3 * extra, ok
    lo, hi = -1.2, 1.45
    if ev(lo)[2] > 0:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if ev(mid)[2] > 0:
                lo = mid
            else:
                hi = mid
        lo = 0.5 * (lo + hi)
    K, A, g, ok = ev(lo)
    return K, A, ok


def segf(top, bot, yl):
    z = (top - bot).normalized()
    return frame(top, x=yl.cross(z), y=yl, z=z)


def limb_plane(Mp, piv, target, s):
    A = Mp @ piv
    xp = (Mp.to_3x3() @ XAX).normalized()
    d = target - A
    dp = d - xp * d.dot(xp)
    up = (-dp).normalized()
    xs = xp * s
    yl = up.cross(xs).normalized()
    return A, xs, yl, up


def solve(st):
    W = {'root': I4.copy()}
    Mp = st['pel']
    W['pelvis'] = Mp
    for k, lg in LEGS.items():
        ft = st['feet'][k]
        F = ft['pos']
        A, xs, yl, up = limb_plane(Mp, lg['piv'], F, lg['s'])
        W['hip_' + k] = frame(A, x=xs, y=yl, z=up)
        D = A - up * HIP_DROP
        Tv = F - D
        K2, A2, ok = leg2d(Tv.dot(xs), Tv.dot(up), ft.get('fold', 0.0))
        if not ok:
            WARN.add('leg %s overstretched %.2f %s' % (k, Tv.length, CUR[0]))
        Kw, Aw = D + xs * K2[0] + up * K2[1], D + xs * A2[0] + up * A2[1]
        W['thigh_' + k] = segf(D, Kw, yl)
        W['shin_' + k] = segf(Kw, Aw, yl)
        W['tarsus_' + k] = segf(Aw, F, yl)
        s = lg['s']
        W['foot_' + k] = frame(F, x=V(s, 0, 0), y=V(0, s, 0), z=ZAX) @ Ry(ft.get('pitch', 0.0))
    for sd, ar in ARMS.items():
        a = st['arms'][sd]
        Wr = Mp @ a['wrist_p']
        if a.get('reach', 0.0) > 0:
            Wr = Wr.lerp(a['wrist_w'], a['reach'])
        A, xs, yl, up = limb_plane(Mp, ar['piv'], Wr, 1)
        W['shoulder_' + sd] = frame(A, x=xs, y=yl, z=up)
        D = A - up * SH_DROP
        Tv = Wr - D
        E2, W2, ok = two_bone(Tv.dot(xs), Tv.dot(up), LU, LF, -1)
        if not ok:
            WARN.add('arm %s overstretched %.2f %s' % (sd, Tv.length, CUR[0]))
        Ew, Ww = D + xs * E2[0] + up * E2[1], D + xs * W2[0] + up * W2[1]
        W['upperarm_' + sd] = segf(D, Ew, yl)
        Mf = segf(Ew, Ww, yl)
        W['forearm_' + sd] = Mf
        zf = Mf.to_3x3().col[2]
        zh = zf.lerp(ZAX, a.get('level', 0.0)).normalized()
        yh = (yl - zh * yl.dot(zh)).normalized()
        W['hand_' + sd] = frame(Ww, y=yh, z=zh) @ Ry(a.get('hp', 0.0)) @ Rz(a.get('hr', 0.0))
    for b in FK_ORDER:
        W[b] = W[PARENT[b]] @ FKD[b] @ st['fk'].get(b, I4)
    W['ik_foot_root'] = I4.copy()
    W['ik_hand_root'] = I4.copy()
    for k in LEGS:
        W['ik_foot_' + k] = W['foot_' + k]
    for sd in ARMS:
        W['ik_hand_' + sd] = W['hand_' + sd]
    return W


for _n in ('ik_foot_root', 'ik_hand_root'):
    PARENT[_n] = 'root'
for _k in LEGS:
    PARENT['ik_foot_' + _k] = 'ik_foot_root'
for _sd in ARMS:
    PARENT['ik_hand_' + _sd] = 'ik_hand_root'


# ----------------------------------------------------------------------------------------------
# pose states and the three actions
# ----------------------------------------------------------------------------------------------
def base():
    return dict(pel=T(PEL), feet={k: dict(pos=lg['foot'].copy(), pitch=0.0, fold=0.0) for k, lg in LEGS.items()},
                arms={sd: dict(wrist_p=ar['wrist'].copy(), level=0.0, hp=0.0, hr=0.0) for sd, ar in ARMS.items()}, fk={}, light=1.0)


def pelm(dx=0.0, dy=0.0, dz=0.0, pitch=0.0, roll=0.0, yaw=0.0):
    return T(PEL + V(dx, dy, dz)) @ Rz(yaw) @ Ry(pitch) @ Rx(roll)


def set_fingers(st, sd, curls):
    """curls: {'a': (d1, d2, d3)} extra curl in degrees on top of the rest pose."""
    for f, cs in curls.items():
        for i, c in enumerate(cs):
            st['fk']['finger_%s%d_%s' % (f, i + 1, sd)] = Ry(radians(c))


def set_neck(st, pitch, yaw1=0.0, head_yaw=0.0, head_pitch=0.0, yaw_all=0.0):
    for i in range(5):
        st['fk']['neck_%02d' % (i + 1)] = Rz((yaw1 if i == 0 else 0.0) + yaw_all) @ Ry(radians(pitch[i]))
    st['fk']['head_yaw'] = Rz(head_yaw)
    st['fk']['head'] = Ry(head_pitch)


# A natural four-beat walk, lateral sequence, timed like a giraffe's: the hind and fore leg of one side swing almost
# together (the fore leaves 0.14 of a cycle after the hind), then the other side. Each foot is in the air for 0.34 of the
# cycle, so the swings overlap and two or three feet carry the body, which never stops moving forward.
# Cycle length (5.5 s) and the high knee with a low foot are read off the H3 Max clips refs/motion/walk2_*.mp4.
WALK_N, STRIDE, SWING, LIFT = 132, 6.0, 0.34, 1.9
PHASE = dict(RL=0.0, FL=0.14, RR=0.5, FR=0.64)


def walk_state(f):
    t = ((f - 1) % WALK_N) / WALK_N
    st = base()
    lean = sin(2 * pi * (t + 0.01))                     # > 0 while the left pair is in the air: the weight goes right
    beat = 4 * pi * t                                   # everything in the body happens twice a cycle, once per side
    st['pel'] = pelm(0.18 * sin(beat + 0.4), -0.95 * lean, -0.80 - 0.14 * cos(beat - 0.9),
                     pitch=radians(1.1) * sin(beat + 1.9), roll=radians(2.2) * lean, yaw=radians(2.4) * sin(2 * pi * (t - 0.20)))
    for k, lg in LEGS.items():
        u = (t - PHASE[k]) % 1.0
        ft = st['feet'][k]
        if u < SWING:
            w = u / SWING
            e = w * w * (3 - 2 * w)
            b = sin(pi * w)
            ft['pos'] = lg['foot'] + V(-STRIDE / 2 + STRIDE * e, 0, LIFT * sin(pi * w ** 0.85) ** 0.9)
            ft['pitch'] = -radians(8) * b                  # the foot hangs with the tarsus, toes drooping below it
            ft['fold'] = radians(30) * sin(pi * w ** 0.85)
            for tt, sg in (('a', 1), ('b', 1), ('c', 0.6)):
                st['fk']['toe_%s_%s' % (tt, k)] = Ry(radians(14) * b * sg)
        else:
            w = (u - SWING) / (1 - SWING)
            ft['pos'] = lg['foot'] + V(STRIDE / 2 - STRIDE * w, 0, 0)
    for sd, ar in ARMS.items():                         # each arm swings against the front leg of its own side
        uf = (t - PHASE['F' + sd]) % 1.0
        sw = cos(2 * pi * (uf - SWING))
        a = st['arms'][sd]
        a['wrist_p'] = ar['wrist'] + V(-0.85 * sw, 0.0, 0.10 * abs(sw))
        c = 4.0 * sw
        set_fingers(st, sd, dict(a=(c, c, c), b=(c * 0.7, c, c), c=(c, c, c)))
    n = 1.5 * sin(beat + 2.6)                           # the neck rocks fore and aft with each side's step; the head stays level
    set_neck(st, (n, n, n * 0.7, n * 0.4, 0.0), yaw1=-radians(2.0) * sin(2 * pi * (t - 0.20)),
             head_pitch=radians(2.0 - 3.1 * n) - radians(1.1) * sin(beat + 1.9))
    return st


REACH_N = 312
REACH_TGT = V(7.0, 4.6, 5.0)


def reach_state(f):
    st = base()
    # timing and stance read off the H3 Max reach clip: a long slow lowering led by the front legs (the rear legs
    # stay tall, so the body tips forward like a giraffe drinking), the arm unfolds late, the hand holds open
    c = ss(14, 112, f) - ss(238, 304, f)                 # lower
    r = ss(64, 140, f) - ss(212, 266, f)                 # arm out
    o = ss(136, 164, f) - ss(184, 208, f)                # hand opens ...
    g = ss(184, 208, f) - ss(224, 262, f)                # ... and closes on the thing
    st['pel'] = pelm(1.0 * c, 0.25 * c, -4.6 * c, pitch=radians(22) * c, yaw=radians(5) * c)
    a = st['arms']['L']
    a['reach'], a['wrist_w'], a['level'] = r, REACH_TGT, r
    a['hr'] = radians(12) * r
    op = (-9, -7, -5)
    cl = (22, 30, 26)
    set_fingers(st, 'L', {k: tuple(op[i] * o + cl[i] * g for i in range(3)) for k in 'abc'})
    st['fk']['finger_c1_L'] = Ry(radians(-14 * o + 26 * g))
    b = st['arms']['R']
    b['wrist_p'] = ARMS['R']['wrist'] + V(-1.6 * c, -0.5 * c, 0.6 * c)
    set_fingers(st, 'R', {k: (8 * c, 10 * c, 8 * c) for k in 'abc'})
    set_neck(st, tuple(v * c for v in (8, 9, 9, 7, 5)), yaw1=radians(9) * r, head_yaw=radians(12) * r, head_pitch=radians(-6) * c)
    st['light'] = 1.0 + 0.9 * o + 0.5 * g
    return st


HEAD_N = 240


def head_state(f):
    st = base()
    wake = ss(18, 78, f)
    droop = 1.0 - wake
    look_l = ss(92, 128, f) - ss(150, 176, f)
    look_r = ss(160, 196, f) - ss(208, 236, f)
    yaw = radians(52) * look_l - radians(40) * look_r
    st['pel'] = pelm(0, 0, -0.35 * droop + 0.05 * sin(2 * pi * f / 96.0), pitch=radians(2.0) * droop)
    lift = -2.5 * ss(60, 100, f) * (1 - ss(200, 236, f))
    set_neck(st, tuple(v * droop + lift for v in (4, 5, 5, 4, 3)), yaw1=0.28 * yaw, head_yaw=0.72 * yaw,
             head_pitch=radians(14) * droop - radians(4) * (look_l + look_r))
    lv = 0.0
    for a, b, v in ((60, 63, 0.5), (63, 66, 0.1), (66, 70, 0.8), (70, 74, 0.3), (74, 90, 1.0)):
        if f >= a:
            lv = lv + (v - lv) * ss(a, b, f)
    st['light'] = lv * (1.0 + 0.5 * (look_l + look_r))
    for sd in ARMS:
        cu = 14 * droop
        set_fingers(st, sd, {k: (cu, cu, cu) for k in 'abc'})
        st['arms'][sd]['wrist_p'] = ARMS[sd]['wrist'] + V(-0.5 * droop, 0, 0.1 * droop)
    return st


ACTIONS = (('A_Stilt_Walk', walk_state, WALK_N + 1), ('A_Stilt_Reach', reach_state, REACH_N), ('A_Stilt_HeadTurn', head_state, HEAD_N))
REST_ST = base()
REST = solve(REST_ST)
SAMPLES = [REST]
CUR = ['rest']
for _an, _fn, _n in ACTIONS:
    for _f in range(1, _n + 1, 4):
        CUR[0] = '%s f%d' % (_an, _f)
        SAMPLES.append(solve(_fn(_f)))
print('SOLVED', len(SAMPLES), 'sample poses', sorted(WARN), round(time.time() - T0, 1), 's')

# ----------------------------------------------------------------------------------------------
# actuators: anchors are searched so one cylinder + rod can cover every sampled pose of every limb
# ----------------------------------------------------------------------------------------------
ACT_DEF = dict(
    hip=dict(top='hip', bot='thigh', limbs=list(LEGS), rc=0.15,
             tops=[(x, z) for x in (1.2, 1.5, 1.8, 2.1) for z in (-0.35, -0.7, -1.05)],
             bots=[(x, z) for x in (0.75, 0.9) for z in (-1.8, -2.4, -3.0, -3.6)]),
    knee=dict(top='thigh', bot='shin', limbs=list(LEGS), rc=0.13,
              tops=[(x, -(L1 - u)) for x in (-0.8, -0.95, -1.1, -1.25) for u in (2.0, 2.5, 3.0, 3.5)],
              bots=[(x, z) for x in (-0.7, -0.85, -1.0, -1.15) for z in (-1.1, -1.4, -1.8, -2.2)]),
    ankle=dict(top='shin', bot='tarsus', limbs=list(LEGS), rc=0.10,
               tops=[(x, -(L2 - u)) for x in (0.6, 0.75, 0.9) for u in (1.8, 2.3, 2.8, 3.3)],
               bots=[(x, z) for x in (0.5, 0.65, 0.8) for z in (-0.8, -1.1, -1.4)]),
    elbow=dict(top='upperarm', bot='forearm', limbs=list(ARMS), rc=0.08,
               tops=[(x, -(LU - u)) for x in (0.5, 0.65, 0.8) for u in (1.5, 2.0, 2.6)],
               bots=[(x, z) for x in (0.45, 0.6) for z in (-0.8, -1.1, -1.4)]),
)
ACT = {}
ANCH_FILE = os.path.join(HERE, OUT + '_anchors.json')
FROZEN = json.load(open(ANCH_FILE)) if os.path.exists(ANCH_FILE) and '--search-anchors' not in ARGS else {}
for _n, _d in ACT_DEF.items():
    best = None
    if _n in FROZEN:                 # frozen: the lugs, shell slots and cylinders are geometry, and geometry is baked
        _d = dict(_d, tops=[tuple(FROZEN[_n]['tp'])], bots=[tuple(FROZEN[_n]['bp'])])
    mats = [(W['%s_%s' % (_d['top'], lb)], W['%s_%s' % (_d['bot'], lb)]) for W in SAMPLES for lb in _d['limbs']]
    for tp in _d['tops']:
        pt = V(tp[0], 0, tp[1])
        tw = [m[0] @ pt for m in mats]
        for bp in _d['bots']:
            pb = V(bp[0], 0, bp[1])
            lo, hi, lev = 1e9, 0.0, 1e9
            for (mt, mb), a in zip(mats, tw):
                b = mb @ pb
                ln = (a - b).length
                lo, hi = min(lo, ln), max(hi, ln)
                j = mb.to_translation()
                lev = min(lev, ((j - a).cross(b - a)).length / max(ln, 1e-6))
            ok = hi <= 2 * lo - 0.75
            sc = (1 if ok else 0, lev if ok else -(hi / lo))
            if best is None or sc > best[0]:
                best = (sc, tp, bp, lo, hi, lev)
    sc, tp, bp, lo, hi, lev = best
    Lc = lo - 0.30
    Lr = hi - Lc + 0.28
    if _n in FROZEN:
        Lc, Lr = FROZEN[_n]['Lc'], FROZEN[_n]['Lr']
        sc = (1 if (lo >= Lc + 0.2 and hi <= Lc + Lr - 0.2) else 0, 0)
    ACT[_n] = dict(top=_d['top'], bot=_d['bot'], limbs=_d['limbs'], rc=_d['rc'], tp=tp, bp=bp, lo=lo, hi=hi, lever=lev,
                   Lc=Lc, Lr=Lr, ok=bool(sc[0]))
    print('ACT %-6s top %s bot %s stroke %.2f..%.2f lever %.2f cyl %.2f rod %.2f %s' % (_n, tp, bp, lo, hi, lev, Lc, Lr, 'OK' if sc[0] else 'NOT TELESCOPIC'))
    for lb in _d['limbs']:
        PARENT['act_%s_top_%s' % (_n, lb)] = '%s_%s' % (_d['top'], lb)
        PARENT['act_%s_bot_%s' % (_n, lb)] = '%s_%s' % (_d['bot'], lb)


def add_acts(W):
    for n, d in ACT.items():
        for lb in d['limbs']:
            mt, mb = W['%s_%s' % (d['top'], lb)], W['%s_%s' % (d['bot'], lb)]
            a, b = mt @ V(d['tp'][0], 0, d['tp'][1]), mb @ V(d['bp'][0], 0, d['bp'][1])
            z = (a - b).normalized()
            y = mt.to_3x3().col[1]
            W['act_%s_top_%s' % (n, lb)] = frame(a, y=y, z=z)
            W['act_%s_bot_%s' % (n, lb)] = frame(b, y=y, z=z)
    return W


if not FROZEN:
    json.dump({k: dict(tp=v['tp'], bp=v['bp'], Lc=round(v['Lc'], 3), Lr=round(v['Lr'], 3)) for k, v in ACT.items()}, open(ANCH_FILE, 'w'), indent=1)
add_acts(REST)
BONE_ORDER = []
_pend = dict(PARENT)
while _pend:
    for _n, _p in list(_pend.items()):
        if _p is None or _p in BONE_ORDER:
            BONE_ORDER.append(_n)
            del _pend[_n]
LOCREST = {b: (REST[PARENT[b]].inverted() @ REST[b]) for b in BONE_ORDER if PARENT[b]}
for _k in LEGS:
    print('REST leg %s knee %s ankle %s' % (_k, [round(v, 2) for v in REST['shin_' + _k].to_translation()], [round(v, 2) for v in REST['tarsus_' + _k].to_translation()]))
for _sd in ARMS:
    print('REST arm %s elbow %s wrist %s' % (_sd, [round(v, 2) for v in REST['forearm_' + _sd].to_translation()], [round(v, 2) for v in REST['hand_' + _sd].to_translation()]))

# ----------------------------------------------------------------------------------------------
# scene, materials (procedural here; baked to textures further down)
# ----------------------------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
SCN = bpy.context.scene
SCN.unit_settings.system = 'METRIC'
SCN.unit_settings.scale_length = 1.0
COL = bpy.data.collections.new('Stilt')
SCN.collection.children.link(COL)
MATS = {}
CER, CARB, MET, STL, CAB, RUB, EMIT = ('M_Stilt_Ceramic', 'M_Stilt_Carbon', 'M_Stilt_Metal', 'M_Stilt_Steel', 'M_Stilt_Cable',
                                       'M_Stilt_Rubber', 'M_Stilt_Emit_Sensor')
FRAME_SET = (CARB, MET, STL, CAB, RUB)
BLUE = (0.01, 0.10, 1.0)


def mat(name, base, metallic=0.0, rough=0.5, noise=0.0, nscale=1.5, emit=None, strength=0.0, bump=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = (*base, 1)
    b.inputs['Metallic'].default_value = metallic
    b.inputs['Roughness'].default_value = rough
    if emit:
        b.inputs['Emission Color'].default_value = (*emit, 1)
        b.inputs['Emission Strength'].default_value = strength
    if noise:
        tc = nt.nodes.new('ShaderNodeAttribute')
        tc.attribute_name = 'rest_pos'
        nz = nt.nodes.new('ShaderNodeTexNoise')
        nz.inputs['Scale'].default_value = nscale
        nz.inputs['Detail'].default_value = 6.0
        mr = nt.nodes.new('ShaderNodeMapRange')
        mr.inputs['To Min'].default_value = rough - noise
        mr.inputs['To Max'].default_value = rough + noise
        nt.links.new(tc.outputs['Vector'], nz.inputs['Vector'])
        nt.links.new(nz.outputs['Fac'], mr.inputs['Value'])
        nt.links.new(mr.outputs['Result'], b.inputs['Roughness'])
        if bump:
            n2 = nt.nodes.new('ShaderNodeTexNoise')
            n2.inputs['Scale'].default_value = nscale * 9
            n2.inputs['Detail'].default_value = 4.0
            bp = nt.nodes.new('ShaderNodeBump')
            bp.inputs['Strength'].default_value = bump
            bp.inputs['Distance'].default_value = 0.01
            nt.links.new(tc.outputs['Vector'], n2.inputs['Vector'])
            nt.links.new(n2.outputs['Fac'], bp.inputs['Height'])
            nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    m.diffuse_color = (*base, 1)
    MATS[name] = m
    return m


def ceramic(name):
    """Pale ceramic: large tonal blotches, faint vertical rain streaks, fine pitting."""
    m = mat(name, (0.60, 0.595, 0.57), 0.0, 0.42)
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    L = nt.links.new
    tc = nt.nodes.new('ShaderNodeAttribute')
    tc.attribute_name = 'rest_pos'
    big = nt.nodes.new('ShaderNodeTexNoise')
    big.inputs['Scale'].default_value = 0.45
    big.inputs['Detail'].default_value = 5.0
    tone = nt.nodes.new('ShaderNodeMapRange')
    tone.inputs['From Min'].default_value = 0.3
    tone.inputs['From Max'].default_value = 0.7
    tone.inputs['To Min'].default_value = 0.86
    tone.inputs['To Max'].default_value = 1.04
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (5.0, 5.0, 0.22)
    st = nt.nodes.new('ShaderNodeTexNoise')
    st.inputs['Scale'].default_value = 1.0
    st.inputs['Detail'].default_value = 7.0
    sr = nt.nodes.new('ShaderNodeMapRange')
    sr.inputs['From Min'].default_value = 0.56
    sr.inputs['From Max'].default_value = 0.78
    sr.inputs['To Min'].default_value = 0.0
    sr.inputs['To Max'].default_value = 0.45
    mul = nt.nodes.new('ShaderNodeMixRGB')
    mul.blend_type = 'MULTIPLY'
    mul.inputs['Fac'].default_value = 1.0
    mul.inputs['Color1'].default_value = (0.60, 0.595, 0.57, 1)
    grime = nt.nodes.new('ShaderNodeMixRGB')
    grime.inputs['Color2'].default_value = (0.30, 0.285, 0.25, 1)
    fine = nt.nodes.new('ShaderNodeTexNoise')
    fine.inputs['Scale'].default_value = 14.0
    fine.inputs['Detail'].default_value = 6.0
    rr = nt.nodes.new('ShaderNodeMapRange')
    rr.inputs['To Min'].default_value = 0.34
    rr.inputs['To Max'].default_value = 0.52
    bp = nt.nodes.new('ShaderNodeBump')
    bp.inputs['Strength'].default_value = 0.12
    bp.inputs['Distance'].default_value = 0.01
    L(tc.outputs['Vector'], big.inputs['Vector'])
    L(tc.outputs['Vector'], mp.inputs['Vector'])
    L(mp.outputs['Vector'], st.inputs['Vector'])
    L(tc.outputs['Vector'], fine.inputs['Vector'])
    L(big.outputs['Fac'], tone.inputs['Value'])
    L(tone.outputs['Result'], mul.inputs['Color2'])
    L(st.outputs['Fac'], sr.inputs['Value'])
    L(mul.outputs['Color'], grime.inputs['Color1'])
    L(sr.outputs['Result'], grime.inputs['Fac'])
    L(grime.outputs['Color'], b.inputs['Base Color'])
    L(fine.outputs['Fac'], rr.inputs['Value'])
    L(rr.outputs['Result'], b.inputs['Roughness'])
    L(fine.outputs['Fac'], bp.inputs['Height'])
    L(bp.outputs['Normal'], b.inputs['Normal'])
    return m


ceramic(CER)
mat(CARB, (0.020, 0.020, 0.023), 0.0, 0.46, noise=0.10, nscale=2.0, bump=0.05)
mat(MET, (0.085, 0.085, 0.095), 0.9, 0.50, noise=0.12, nscale=1.6)
mat(STL, (0.60, 0.60, 0.62), 1.0, 0.24, noise=0.06, nscale=3.0)
mat(CAB, (0.032, 0.030, 0.028), 0.0, 0.68, noise=0.10, nscale=30.0, bump=0.3)
mat(RUB, (0.018, 0.018, 0.018), 0.0, 0.90)
mat(EMIT, (0.01, 0.02, 0.05), 0.0, 0.3, emit=BLUE, strength=6.0)
BEV = {CER: 0.02, CARB: 0.018, MET: 0.014, STL: 0.006, CAB: 0.0, RUB: 0.01, EMIT: 0.0}


def industrial_wear(m, amount):
    """albertop27 fork: heavy-machinery wear layered over a material (owner: 'more roughness of large size machines').
    Rust patches, vertical rain/oil streaks, dirt in crevices (pointiness), bright worn edges, rougher overall.
    All inputs are rest_pos (metres) or geometry, so it bakes with the Cycles bake below."""
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    L = nt.links.new
    col_in = b.inputs['Base Color']
    src = col_in.links[0].from_socket if col_in.is_linked else None
    base = tuple(col_in.default_value)
    tc = nt.nodes.new('ShaderNodeAttribute'); tc.attribute_name = 'rest_pos'
    geo = nt.nodes.new('ShaderNodeNewGeometry')

    def noise(scale, detail, vec):
        n = nt.nodes.new('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        L(vec, n.inputs['Vector'])
        return n

    def ramp(sock, lo, hi, a=0.0, z=1.0):
        r = nt.nodes.new('ShaderNodeMapRange')
        r.inputs['From Min'].default_value = lo; r.inputs['From Max'].default_value = hi
        r.inputs['To Min'].default_value = a; r.inputs['To Max'].default_value = z
        L(sock, r.inputs['Value'])
        return r.outputs['Result']

    def mix(fac, c1, c2, blend='MIX'):
        x = nt.nodes.new('ShaderNodeMixRGB'); x.blend_type = blend
        if isinstance(fac, float):
            x.inputs['Fac'].default_value = fac
        else:
            L(fac, x.inputs['Fac'])
        for i, c in ((1, c1), (2, c2)):
            if isinstance(c, tuple):
                x.inputs['Color%d' % i].default_value = c
            else:
                L(c, x.inputs['Color%d' % i])
        return x.outputs['Color']

    def mul(a, k):
        x = nt.nodes.new('ShaderNodeMath'); x.operation = 'MULTIPLY'
        L(a, x.inputs[0]); x.inputs[1].default_value = k
        return x.outputs[0]

    # rust patches: blotchy, scale ~1-2 m on a 30 m machine
    rust_m = ramp(noise(0.7, 8.0, tc.outputs['Vector']).outputs['Fac'], 0.58 - 0.12 * amount, 0.70, 0, 1)
    rust_c = mix(noise(6.0, 4.0, tc.outputs['Vector']).outputs['Fac'], (0.16, 0.055, 0.02, 1), (0.30, 0.12, 0.04, 1))
    # vertical streaks: noise squashed in Z
    mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (9.0, 9.0, 0.25)
    L(tc.outputs['Vector'], mp.inputs['Vector'])
    streak = ramp(noise(1.0, 6.0, mp.outputs['Vector']).outputs['Fac'], 0.52, 0.75, 0, 0.7)
    # crevice dirt and worn edges from pointiness
    dirt = ramp(geo.outputs['Pointiness'], 0.47, 0.5, 1, 0)
    edge = ramp(geo.outputs['Pointiness'], 0.53, 0.58, 0, 1)
    edge = mul(edge, 1.0)
    c = src if src is not None else base
    c = mix(mul(streak, amount), c, (0.05, 0.04, 0.03, 1), 'MULTIPLY')
    c = mix(mul(dirt, 0.85 * amount), c, (0.035, 0.03, 0.025, 1))
    c = mix(mul(rust_m, amount), c, rust_c)
    c = mix(mul(edge, 0.6 * amount), c, (0.42, 0.41, 0.39, 1))
    L(c, col_in)
    # roughness: rougher overall, rust and dirt very rough, edges polished
    r_in = b.inputs['Roughness']
    r_src = r_in.links[0].from_socket if r_in.is_linked else None
    add = nt.nodes.new('ShaderNodeMath'); add.operation = 'ADD'; add.use_clamp = True
    if r_src is not None:
        L(r_src, add.inputs[0])
    else:
        add.inputs[0].default_value = r_in.default_value
    add.inputs[1].default_value = 0.18 * amount
    rr = mix(rust_m, add.outputs[0], (0.9, 0.9, 0.9, 1))
    rr = mix(mul(edge, 0.6 * amount), rr, (0.3, 0.3, 0.3, 1))
    bw = nt.nodes.new('ShaderNodeRGBToBW'); L(rr, bw.inputs['Color'])
    L(bw.outputs['Val'], r_in)
    # rust kills metalness
    if b.inputs['Metallic'].default_value > 0.1:
        met = nt.nodes.new('ShaderNodeMath'); met.operation = 'MULTIPLY'
        inv = ramp(rust_m, 0, 1, 1, 0)
        L(inv, met.inputs[0]); met.inputs[1].default_value = b.inputs['Metallic'].default_value
        L(met.outputs[0], b.inputs['Metallic'])


WEAR = 0.0
if '--wear' in ARGS:
    WEAR = float(ARGS[ARGS.index('--wear') + 1])
    for _mn in (CER, CARB, MET, STL, CAB):
        industrial_wear(MATS[_mn], WEAR)
    print('wear', WEAR, flush=True)

# ----------------------------------------------------------------------------------------------
# mesh helpers
# ----------------------------------------------------------------------------------------------


class Pal(dict):
    """material name -> bmesh. A key ending in '!' is not bevelled."""
    def __missing__(self, k):
        self[k] = bmesh.new()
        return self[k]


def face(bm, vs):
    try:
        return bm.faces.new(vs)
    except ValueError:
        return None


def loft(bm, rings, cap0=True, cap1=True, closed=True):
    vr = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(vr[0])
    for a, b in zip(vr, vr[1:]):
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            face(bm, [a[i], a[j], b[j], b[i]])
    if cap0:
        face(bm, list(reversed(vr[0])))
    if cap1:
        face(bm, vr[-1])
    return vr


def tube(bm, pts, r, segs=10, caps=True):
    pts = [Vector(p) for p in pts]
    n = len(pts)
    tans = [(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized() for i in range(n)]
    t0 = tans[0]
    u = t0.cross(ZAX if abs(t0.z) < 0.9 else XAX).normalized()
    rings = []
    for i in range(n):
        t = tans[i]
        u = (u - t * u.dot(t)).normalized()
        w = t.cross(u)
        rr = r[i] if isinstance(r, (list, tuple)) else r
        rings.append([pts[i] + (u * cos(2 * pi * k / segs) + w * sin(2 * pi * k / segs)) * rr for k in range(segs)])
    loft(bm, rings, caps, caps)


def disc(bm, cx, cz, y0, y1, r, segs=28):
    tube(bm, [V(cx, y0, cz), V(cx, y1, cz)], r, segs)


def lathe(bm, prof, M, segs=32):
    """Closed (r, h) profile revolved about M's Z axis."""
    vr = []
    for i in range(segs):
        a = 2 * pi * i / segs
        vr.append([bm.verts.new(M @ V(r * cos(a), r * sin(a), h)) for r, h in prof])
    vr.append(vr[0])
    n = len(prof)
    for a, b in zip(vr, vr[1:]):
        for i in range(n):
            j = (i + 1) % n
            face(bm, [a[i], a[j], b[j], b[i]])


def MY(cx, cz, y=0.0):
    return frame(V(cx, y, cz), x=ZAX, y=XAX, z=YAX)


def ringY(bm, cx, cz, y0, y1, ro, ri, segs=32):
    lathe(bm, [(ri, y0), (ro, y0), (ro, y1), (ri, y1)], MY(cx, cz), segs)


def dome(bm, cx, cz, y0, h, r, segs=32, sz=1.0):
    prof = [(0.002, y0), (r, y0), (r, y0 + 0.3 * h), (0.86 * r, y0 + 0.75 * h), (0.5 * r, y0 + h), (0.002, y0 + h)]
    lathe(bm, prof, MY(cx, cz) @ Matrix.Diagonal((sz, 1, 1, 1)), segs)


def boltsY(bm, cx, cz, R, n, y, h, rb=0.035, a0=0.0):
    for i in range(n):
        a = a0 + 2 * pi * i / n
        p = V(cx + R * cos(a), y, cz + R * sin(a))
        tube(bm, [p, p + V(0, h, 0)], rb, 6)


def hexbolt(bm, pos, nrm, r=0.035, h=0.03):
    tube(bm, [pos, pos + nrm.normalized() * h], r, 6)


def box(bm, lo, hi, M=None):
    c = [V(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    if M is not None:
        c = [M @ p for p in c]
    vs = [bm.verts.new(p) for p in c]
    for idx in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        face(bm, [vs[i] for i in idx])


def tbox(bm, secs, M=None):
    """Loft of rectangles: (centre, half x, half y), each in the local XY plane at the centre."""
    rings = []
    for c, hx, hy in secs:
        r = [c + V(hx, hy, 0), c + V(-hx, hy, 0), c + V(-hx, -hy, 0), c + V(hx, -hy, 0)]
        rings.append([M @ p for p in r] if M is not None else r)
    loft(bm, rings)


def bar(bm, a, b, t, y0, y1):
    """Flat bar in the XZ plane from a to b (x, z), thickness t, spanning y0..y1."""
    ax, az = a
    bx, bz = b
    ln = hypot(bx - ax, bz - az)
    nx, nz = -(bz - az) / ln * t / 2, (bx - ax) / ln * t / 2
    loft(bm, [[V(ax + nx, y, az + nz), V(bx + nx, y, bz + nz), V(bx - nx, y, bz - nz), V(ax - nx, y, az - nz)] for y in (y0, y1)])


def blade(bm, path, hw, th):
    """Flat tapering blade along a path; hw/th are (start, end) half-width and half-thickness."""
    n = len(path)
    rings = []
    for i, p in enumerate(path):
        t = (path[min(i + 1, n - 1)] - path[max(i - 1, 0)]).normalized()
        side = t.cross(ZAX).normalized()
        up = side.cross(t)
        k = i / (n - 1)
        w, h = hw[0] + (hw[1] - hw[0]) * k, th[0] + (th[1] - th[0]) * k
        rings.append([p + side * w, p + up * h, p - side * w, p - up * h])
    loft(bm, rings)


def catenary(p0, p1, sag, n=16):
    return [p0.lerp(p1, i / n) + V(0, 0, -sag * 4 * (i / n) * (1 - i / n)) for i in range(n + 1)]


def smooth_path(pts, n=6):
    Pp = [Vector(p) for p in pts]
    Pp = [Pp[0]] + Pp + [Pp[-1]]
    out = []
    for i in range(1, len(Pp) - 2):
        for k in range(n):
            t = k / n
            out.append(0.5 * ((2 * Pp[i]) + (-Pp[i - 1] + Pp[i + 1]) * t
                              + (2 * Pp[i - 1] - 5 * Pp[i] + 4 * Pp[i + 1] - Pp[i + 2]) * t * t
                              + (-Pp[i - 1] + 3 * Pp[i] - 3 * Pp[i + 1] + Pp[i + 2]) * t ** 3))
    out.append(Pp[-2])
    return out


def thick_grid(bm, O, I):
    n, m = len(O), len(O[0])
    vo = [[bm.verts.new(p) for p in r] for r in O]
    vi = [[bm.verts.new(p) for p in r] for r in I]
    for i in range(n - 1):
        for j in range(m - 1):
            face(bm, [vo[i][j], vo[i][j + 1], vo[i + 1][j + 1], vo[i + 1][j]])
            face(bm, [vi[i][j], vi[i + 1][j], vi[i + 1][j + 1], vi[i][j + 1]])
        face(bm, [vo[i][0], vo[i + 1][0], vi[i + 1][0], vi[i][0]])
        face(bm, [vo[i][m - 1], vi[i][m - 1], vi[i + 1][m - 1], vo[i + 1][m - 1]])
    for j in range(m - 1):
        face(bm, [vo[0][j], vi[0][j], vi[0][j + 1], vo[0][j + 1]])
        face(bm, [vo[n - 1][j], vo[n - 1][j + 1], vi[n - 1][j + 1], vi[n - 1][j]])


def surf_panel(bm, fn, u0, u1, v0, v1, nu, nv, thick, centre, lift=0.0):
    """Thick plate on the parametric surface fn(u, v); the outside is the side away from `centre`."""
    O, I = [], []
    e = 1e-3
    last = ZAX
    for i in range(nv + 1):
        v = v0 + (v1 - v0) * i / nv
        ro, ri = [], []
        for j in range(nu + 1):
            u = u0 + (u1 - u0) * j / nu
            p = fn(u, v)
            n = (fn(u + e, v) - fn(u - e, v)).cross(fn(u, v + e) - fn(u, v - e))
            if n.length < 1e-9:
                n = last
            n = n.normalized()
            if n.dot(p - centre) < 0:
                n = -n
            last = n
            ro.append(p + n * lift)
            ri.append(p + n * (lift - thick))
        O.append(ro)
        I.append(ri)
    thick_grid(bm, O, I)


class Shell:
    """Ceramic channel around a tapering beam: an elliptical arc centred on angle phic (0 = front, pi = back)."""
    def __init__(self, dfun, wfun, off, phic, phiw, Z0, Z1, tip0=0.12, tip1=0.12):
        self.d, self.w, self.off, self.phic, self.phiw, self.Z0, self.Z1, self.tip0, self.tip1 = dfun, wfun, off, phic, phiw, Z0, Z1, tip0, tip1

    def k(self, z):
        g = (z - self.Z0) / (self.Z1 - self.Z0)
        k = 1.0
        if self.tip0 > 0 and g < self.tip0:
            k = 0.14 + 0.86 * sin(0.5 * pi * max(g, 0.0) / self.tip0)
        if self.tip1 > 0 and g > 1 - self.tip1:
            k = min(k, 0.14 + 0.86 * sin(0.5 * pi * max(1 - g, 0.0) / self.tip1))
        return k

    def pt(self, z, t, lift=0.0):
        rx, ry = self.d(z) / 2 + self.off + lift, self.w(z) / 2 + self.off + lift
        ph = self.phic + t * self.phiw * self.k(z)
        return V(rx * cos(ph), ry * sin(ph), z)

    def panel(self, bm, za, zb, t0=-1.0, t1=1.0, thick=0.07, lift=0.0, nz=8, nt=8):
        O = [[self.pt(za + (zb - za) * i / nz, t0 + (t1 - t0) * j / nt, lift) for j in range(nt + 1)] for i in range(nz + 1)]
        I = [[self.pt(za + (zb - za) * i / nz, t0 + (t1 - t0) * j / nt, lift - thick) for j in range(nt + 1)] for i in range(nz + 1)]
        thick_grid(bm, O, I)

    def build(self, P, Pm, panels=3, seam=True, gap=0.05, hatch=True, slots=(), slot_t=0.24):
        """Panels along the length, centre seam, stand-offs to the beam, a hatch, vent slots and corner bolts."""
        L = self.Z1 - self.Z0
        sgn = 1 if L > 0 else -1
        for i in range(panels):
            za, zb = self.Z0 + L * i / panels, self.Z0 + L * (i + 1) / panels
            za += (gap / 2 if i else 0) * sgn
            zb -= (gap / 2 if i < panels - 1 else 0) * sgn
            cuts = sorted({za, zb} | {z for sl in slots for z in sl if min(za, zb) < z < max(za, zb)}, reverse=sgn < 0)
            for ca, cb in zip(cuts, cuts[1:]):
                mid = 0.5 * (ca + cb)
                g = slot_t if any(min(sl) <= mid <= max(sl) for sl in slots) else 0.025
                nzz = max(2, int(round(8 * abs(cb - ca) / abs(zb - za))))
                for t0, t1 in (((-1.0, -g), (g, 1.0)) if (seam or g > 0.03) else ((-1.0, 1.0),)):
                    self.panel(P[CER], ca, cb, t0, t1, nz=nzz, nt=8 if g < 0.03 else 5)
            zm = 0.5 * (za + zb)
            c = V(cos(self.phic), 0, 0)
            for zz in (za + 0.25 * (zb - za), za + 0.75 * (zb - za)):
                for sy in (1, -1):
                    pi_ = self.pt(zz, 0.5 * sy, -0.06)
                    tube(Pm[MET], [V(c.x * (self.d(zz) / 2 - 0.05), sy * self.w(zz) * 0.3, zz), pi_], 0.045, 8)
            for tt in (-0.86, 0.86):
                for zz in (za + 0.12 * (zb - za), zb - 0.12 * (zb - za)):
                    p = self.pt(zz, tt)
                    hexbolt(P[STL + '!'], p, V(p.x, p.y, 0), 0.03, 0.025)
            if hatch and i == panels // 2:
                self.panel(P[CER], zm - 0.45, zm + 0.45, 0.32, 0.74, 0.03, 0.014, 2, 3)
            if i == 0:
                for q in range(4):
                    zq = za + (zb - za) * (0.30 + 0.035 * q)
                    self.panel(P[CARB], zq, zq + 0.045 * sgn, -0.72, -0.40, 0.02, 0.006, 1, 3)


def fork_side(P, Pc, cz, r, w, up, capk=0.66, capz=1.0):
    """Parent side of a pin joint: two cheeks, the motor drum on the outer (+Y) side, bearing cap on the inner."""
    m = P[MET]
    for sg in (1, -1):
        y0, y1 = sorted((sg * 0.47 * w, sg * 0.67 * w))
        disc(m, 0, cz, y0, y1, r, 24)
        box(m, V(-0.55 * r, y0, cz), V(0.55 * r, y1, cz + up))
    yo = 0.67 * w
    ringY(m, 0, cz, yo, yo + 0.10 * w, 1.03 * r, 0.5 * r, 24)
    boltsY(P[STL + '!'], 0, cz, 0.93 * r, 12, yo + 0.10 * w, 0.04, 0.03 + 0.02 * r)
    disc(m, 0, cz, yo + 0.10 * w, yo + 0.30 * w, 0.84 * r, 24)
    for i in range(8):
        a = 2 * pi * (i + 0.5) / 8
        bar(P[CARB], (0.78 * r * cos(a), cz + 0.78 * r * sin(a)), (0.87 * r * cos(a), cz + 0.87 * r * sin(a)), 0.16 * r, yo + 0.13 * w, yo + 0.27 * w)
    disc(m, 0, cz, yo + 0.30 * w, yo + 0.44 * w, 0.70 * r, 24)
    Mc = MY(0, cz) @ Matrix.Diagonal((capz, 1, 1, 1))
    rc_, y0_ = capk * r, yo + 0.44 * w
    dome(Pc[CER], 0, cz, y0_, 0.34 * r, 0.70 * rc_, 24, capz)
    lathe(Pc[CER], [(0.74 * rc_, y0_), (rc_, y0_), (rc_, y0_ + 0.10 * r), (0.93 * rc_, y0_ + 0.19 * r), (0.74 * rc_, y0_ + 0.22 * r)], Mc, 28)
    for i in range(6):
        a = 2 * pi * (i + 0.5) / 6
        hexbolt(Pc[STL + '!'], Mc @ V(0.86 * rc_ * cos(a), 0.86 * rc_ * sin(a), y0_ + 0.20 * r), YAX, 0.03, 0.025)
    disc(P[STL], 0, cz, yo + 0.44 * w, yo + 0.44 * w + 0.27 * r, 0.10 * r, 16)
    yi = -0.67 * w
    disc(m, 0, cz, yi - 0.09 * w, yi, 0.62 * r, 28)
    boltsY(P[STL + '!'], 0, cz, 0.52 * r, 10, yi - 0.09 * w, -0.04, 0.03 + 0.015 * r)
    disc(m, 0, cz, yi - 0.20 * w, yi - 0.09 * w, 0.34 * r, 20)
    disc(P[STL], 0, cz, yi - 0.27 * w, yi - 0.20 * w, 0.15 * r, 16)


def tongue_side(P, r, w, down):
    """Child side of a pin joint: the grooved cable drum between the parent's cheeks, and its neck."""
    m = P[MET]
    disc(m, 0, 0, -0.43 * w, 0.43 * w, 0.80 * r, 24)
    box(m, V(-0.5 * r, -0.40 * w, -down), V(0.5 * r, 0.40 * w, 0))
    for y in (-0.32 * w, 0.32 * w):
        lathe(P[CAB + '!'], [(0.80 * r + 0.01, y - 0.09), (0.80 * r + 0.06, y - 0.045), (0.80 * r + 0.035, y), (0.80 * r + 0.06, y + 0.045), (0.80 * r + 0.01, y + 0.09)], MY(0, 0), 24)
    for y0, y1 in ((0.39 * w, 0.43 * w), (-0.43 * w, -0.39 * w), (-0.02, 0.02)):
        ringY(m, 0, 0, y0, y1, 0.90 * r, 0.5 * r, 24)


def truss(P, za, zb, d0, d1, w, nb):
    """Lattice mast: front and back flanges, diagonal lacing and battens between them."""
    L = za - zb

    def d(z):
        return d0 + (d1 - d0) * (za - z) / L
    c = P[CARB]
    for sg in (1, -1):
        c0, c1 = 0.15 * d0 + 0.03, 0.15 * d1 + 0.03
        tbox(c, [(V(sg * (d0 / 2 - c0 / 2), 0, za), c0 / 2, w / 2), (V(sg * (d1 / 2 - c1 / 2), 0, zb), c1 / 2, w / 2)])
    for i in range(nb):
        z0, z1 = za - L * i / nb, za - L * (i + 1) / nb
        sg = 1 if i % 2 == 0 else -1
        bar(c, (-sg * (d(z0) / 2 - 0.06), z0), (sg * (d(z1) / 2 - 0.06), z1), 0.10, -0.30 * w, 0.30 * w)
        if i < nb - 1:
            bar(c, (-d(z1) / 2 + 0.05, z1), (d(z1) / 2 - 0.05, z1), 0.08, -0.38 * w, 0.38 * w)
            for sy in (1, -1):
                hexbolt(P[STL + '!'], V(sg * (d(z1) / 2 - 0.08), sy * 0.38 * w, z1), V(0, sy, 0), 0.03, 0.03)
    return d


def lug(P, base, tip, yw=0.20, rr=0.19):
    for y0, y1 in ((-yw - 0.07, -yw), (yw, yw + 0.07)):
        bar(P[MET], base, tip, 0.26, y0, y1)
        disc(P[MET], tip[0], tip[1], y0, y1, rr, 16)
    disc(P[STL], tip[0], tip[1], -yw - 0.10, yw + 0.10, 0.08, 12)


def tendons(P, rt, rb, L, wt, wb, dfun, zs, spread=0.45):
    """Two cables down the back, tangent to the cable drum at each end, pushed out by a spreader at zs."""
    rt, rb = 0.80 * rt + 0.06, 0.80 * rb + 0.06
    xs_ = -(dfun(zs) / 2 + spread)
    for sg in (1, -1):
        yt, yb = sg * 0.32 * wt, sg * 0.32 * wb
        ym = 0.5 * (yt + yb)
        pts = [V(-rt * cos(radians(a)), yt, rt * sin(radians(a))) for a in (80, 50, 20)]
        pts += [V(-rt, yt, -0.2), V(xs_, ym, zs + 0.5), V(xs_, ym, zs - 0.5), V(-rb, yb, -L + 0.2)]
        pts += [V(-rb * cos(radians(a)), yb, -L - rb * sin(radians(a))) for a in (20, 50, 80)]
        tube(P[CAB + '!'], smooth_path(pts, 4), 0.04, 8)
        for zz in (zs + 0.5, zs - 0.5):
            box(P[MET], V(xs_ - 0.07, ym - 0.07, zz - 0.09), V(xs_ + 0.07, ym + 0.07, zz + 0.09))
    bar(P[MET], (-dfun(zs) / 2 + 0.05, zs + 0.9), (xs_, zs + 0.5), 0.09, -0.06, 0.06)
    bar(P[MET], (-dfun(zs) / 2 + 0.05, zs - 0.9), (xs_, zs - 0.5), 0.09, -0.06, 0.06)
    for zz in (zs + 0.5, zs - 0.5):
        tube(P[MET], [V(xs_, -0.34 * wt, zz), V(xs_, 0.34 * wt, zz)], 0.045, 10)
    for zz in (-L * 0.14, -L * 0.88):                                   # guide clamps near each end
        xc = -(rt + (rb - rt) * (-zz / L)) + 0.02
        box(P[MET], V(xc - 0.07, -0.36 * wt, zz - 0.06), V(xc + 0.07, 0.36 * wt, zz + 0.06))
        bar(P[MET], (-dfun(max(min(zz, -1.2), -(L - 1.2))) / 2 + 0.05, zz), (xc, zz), 0.08, -0.05, 0.05)


# ----------------------------------------------------------------------------------------------
# object pipeline. Meshes are built in bone-local coordinates; `world=True` parts are built in world space.
# ----------------------------------------------------------------------------------------------
PARTS = {}      # name -> (object, bone)
CANON = []      # objects that own their UV islands (their copies share them)


def finish(ob, name):
    m = ob.modifiers.new('wn', 'WEIGHTED_NORMAL')
    m.keep_sharp = True
    m.weight = 60
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    old = ob.data
    ob.modifiers.clear()
    ob.data = me2
    me2.name = name
    bpy.data.meshes.remove(old)


def piece(name, bm, material, bevel):
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ang = radians(30)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        e.smooth = (len(e.link_faces) == 2 and e.calc_face_angle(0.0) < ang)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    COL.objects.link(ob)
    me.materials.append(MATS[material])
    if bevel > 0:
        m = ob.modifiers.new('bev', 'BEVEL')
        m.width = bevel
        m.segments = 1
        m.limit_method = 'ANGLE'
        m.angle_limit = ang
        m.use_clamp_overlap = True
    finish(ob, name)
    return ob


def part(name, bone, P, world=False):
    fr = REST[bone]
    obs = []
    for key, bm in P.items():
        if not len(bm.faces):
            continue
        mname = key.rstrip('!')
        if world:
            bm.transform(fr.inverted())
        obs.append(piece(name, bm, mname, 0.0 if key.endswith('!') else BEV[mname]))
    ob = obs[0]
    if len(obs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in obs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.join()
    ob.name = name
    ob.data.name = name
    ob.matrix_world = fr
    me = ob.data
    sd = sum((i + 1) * ord(ch) for i, ch in enumerate(name))
    off = Vector(((sd % 97) * 0.173, (sd % 89) * 0.211, (sd % 83) * 0.137))
    at = me.attributes.new('rest_pos', 'FLOAT_VECTOR', 'POINT')
    for i, v in enumerate(me.vertices):
        at.data[i].vector = v.co + off
    PARTS[name] = (ob, bone)
    CANON.append(ob)
    return ob


DUPS = []


def dup(*a):
    DUPS.append(a)


def make_dup(src, name, bone, mirror):
    """Copy of a finished part on another bone (same UV islands); `mirror` flips it in local Y."""
    ob0 = PARTS[src][0]
    me = ob0.data.copy()
    me.name = name
    ob = bpy.data.objects.new(name, me)
    COL.objects.link(ob)
    if mirror:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.scale(bm, vec=(1, -1, 1), verts=bm.verts)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        if 'custom_normal' in me.attributes:
            me.attributes.remove(me.attributes['custom_normal'])
        finish(ob, name)
    ob.matrix_world = REST[bone]
    PARTS[name] = (ob, bone)
    return ob


# ----------------------------------------------------------------------------------------------
# leg (built once for FL in bone-local coordinates, copied to the other three)
# ----------------------------------------------------------------------------------------------
def build_leg():
    k = 'FL'
    ah, ak, aa = ACT['hip'], ACT['knee'], ACT['ankle']
    # hip yoke
    P, Pc = Pal(), Pal()
    tube(P[MET], [V(-0.5, 0, 0), V(0.5, 0, 0)], 0.30, 24)
    tube(P[STL], [V(-0.82, 0, 0), V(0.82, 0, 0)], 0.13, 16)
    box(P[MET], V(-0.52, -0.67 * W_HIP, -0.66), V(0.52, 0.67 * W_HIP, -0.12))
    fork_side(P, Pc, -HIP_DROP, R_HIP, W_HIP, HIP_DROP - 0.6)
    xa, za = ah['tp']
    for y0, y1 in ((-0.27, -0.20), (0.20, 0.27)):
        bar(P[MET], (0.35, -0.40), (xa, za), 0.34, y0, y1)
        bar(P[MET], (0.35, -0.62), (xa - 0.5, za - 0.05), 0.14, y0, y1)
        disc(P[MET], xa, za, y0, y1, 0.22, 16)
    disc(P[STL], xa, za, -0.30, 0.30, 0.09, 12)
    part('Hip_Yoke_' + k, 'hip_' + k, P)
    part('Hip_Cap_' + k, 'hip_' + k, Pc)
    # thigh
    P, Pc, Ps, Pt = Pal(), Pal(), Pal(), Pal()
    tongue_side(P, R_HIP, W_HIP, 1.35)
    d = truss(P, -1.15, -(L1 - 1.0), 1.10, 0.75, 0.62, 9)
    fork_side(P, Pc, -L1, R_KNEE, W_KNEE, 1.15, capk=0.92, capz=1.25)
    lug(P, (d(ah['bp'][1]) / 2 - 0.05, ah['bp'][1] + 0.5), ah['bp'])
    lug(P, (-d(ak['tp'][1]) / 2 + 0.05, ak['tp'][1] + 0.5), ak['tp'])
    Shell(d, lambda z: 0.62 + 0.34 * ss(-4.2, -1.6, z), 0.17, 0.0, radians(112), -1.5, -(L1 - 1.25), tip0=0.07, tip1=0.13).build(
        Ps, P, panels=4, slots=((ah['bp'][1] + 0.95, ah['bp'][1] - 0.45),))
    tendons(Pt, R_HIP, R_KNEE, L1, W_HIP, W_KNEE, d, -L1 * 0.36)
    part('Thigh_Beam_' + k, 'thigh_' + k, P)
    part('Knee_Cap_' + k, 'thigh_' + k, Pc)
    part('Thigh_Shell_' + k, 'thigh_' + k, Ps)
    part('Thigh_Tendons_' + k, 'thigh_' + k, Pt)
    # shin
    P, Pc, Ps, Pt = Pal(), Pal(), Pal(), Pal()
    tongue_side(P, R_KNEE, W_KNEE, 1.1)
    d = truss(P, -0.95, -(L2 - 0.75), 0.75, 0.55, 0.52, 9)
    fork_side(P, Pc, -L2, R_ANK, W_ANK, 0.9, capk=0.74)
    lug(P, (-d(ak['bp'][1]) / 2 + 0.05, ak['bp'][1] - 0.5), ak['bp'])
    lug(P, (d(aa['tp'][1]) / 2 - 0.05, aa['tp'][1] + 0.5), aa['tp'], yw=0.17, rr=0.16)
    Shell(d, lambda z: 0.52, 0.15, 0.0, radians(108), -1.25, -(L2 - 1.25), tip0=0.10, tip1=0.12).build(
        Ps, P, panels=4, slots=((aa['tp'][1] + 0.9, aa['tp'][1] - 0.4),))
    tendons(Pt, R_KNEE, R_ANK, L2, W_KNEE, W_ANK, d, -L2 * 0.5, spread=0.40)
    part('Shin_Beam_' + k, 'shin_' + k, P)
    part('Ankle_Cap_' + k, 'shin_' + k, Pc)
    part('Shin_Shell_' + k, 'shin_' + k, Ps)
    part('Shin_Tendons_' + k, 'shin_' + k, Pt)
    # tarsus
    P, Ps, Pt = Pal(), Pal(), Pal()
    tongue_side(P, R_ANK, W_ANK, 0.8)
    d = truss(P, -0.65, -(L3 - 0.8), 0.55, 0.42, 0.44, 7)
    lug(P, (d(aa['bp'][1]) / 2 - 0.05, aa['bp'][1] - 0.45), aa['bp'], yw=0.17, rr=0.16)
    lathe(P[MET], [(0.20, -(L3 - 0.28)), (0.30, -(L3 - 0.8)), (0.30, -(L3 - 0.95)), (0.002, -(L3 - 0.95)), (0.002, -(L3 - 0.28))], I4, 20)
    bmesh.ops.create_uvsphere(P[STL + '!'], u_segments=24, v_segments=12, radius=0.30, matrix=T(0, 0, -L3))
    Shell(d, lambda z: 0.44, 0.13, 0.0, radians(105), -0.95, -(L3 - 0.85), tip0=0.10, tip1=0.12).build(
        Ps, P, panels=3, slots=((-0.95, aa['bp'][1] - 0.85),))
    rt = 0.80 * R_ANK + 0.06
    for sg in (1, -1):                                    # tendons end in turnbuckles above the foot
        y = sg * 0.32 * W_ANK
        xb = -d(-L3 + 1.2) / 2 - 0.11
        pts = [V(-rt * cos(radians(a)), y, rt * sin(radians(a))) for a in (80, 50, 20)] + [V(-rt, y, -0.2), V(xb, y, -(L3 - 1.6))]
        tube(Pt[CAB + '!'], smooth_path(pts, 4), 0.04, 8)
        tube(Pt[MET], [V(xb, y, -(L3 - 1.65)), V(xb, y, -(L3 - 1.1))], 0.065, 8)
        box(Pt[MET], V(xb - 0.09, y - 0.08, -(L3 - 0.9)), V(xb + 0.16, y + 0.08, -(L3 - 1.12)))
    part('Tarsus_Beam_' + k, 'tarsus_' + k, P)
    part('Tarsus_Shell_' + k, 'tarsus_' + k, Ps)
    part('Tarsus_Tendons_' + k, 'tarsus_' + k, Pt)
    # foot block: socket for the ball, ceramic collar, toe lugs
    P, Pc = Pal(), Pal()
    lathe(P[MET], [(0.33, 0.12), (0.58, 0.06), (0.66, -0.20), (0.52, -0.46), (0.002, -0.46), (0.002, -0.30), (0.30, -0.26)], I4, 28)
    lathe(Pc[CER], [(0.66, 0.10), (0.75, 0.06), (0.78, -0.16), (0.70, -0.26)], I4, 32)
    for i in range(8):
        a = 2 * pi * i / 8
        hexbolt(Pc[STL + '!'], V(0.765 * cos(a), 0.765 * sin(a), -0.05), V(cos(a), sin(a), 0), 0.03, 0.03)
    for tname, ang in TOE_ANG.items():
        M = Rz(radians(ang))
        for y0, y1 in ((-0.24, -0.16), (0.16, 0.24)):
            ym = 0.5 * (y0 + y1)
            loft(P[MET], [[M @ V(x, ym + sy * 0.04, z + sz * hz) for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
                          for x, z, hz in ((0.40, -0.30, 0.13), (TOE_R, TOE_DZ, 0.15), (TOE_R + 0.13, TOE_DZ, 0.11))])
        tube(P[STL], [M @ V(TOE_R, -0.27, TOE_DZ), M @ V(TOE_R, 0.27, TOE_DZ)], 0.055, 12)
    part('Foot_Block_' + k, 'foot_' + k, P)
    part('Foot_Collar_' + k, 'foot_' + k, Pc)
    # toes
    for tname, ang in TOE_ANG.items():
        kk = 0.82 if tname == 'c' else 1.0
        P = Pal()
        disc(P[MET], 0, 0, -0.15, 0.15, 0.13, 16)
        LT = 1.85 * kk
        g = -(FOOT_Z + TOE_DZ)                           # ground in toe coordinates
        rings = []
        for x, zt_, zb_, hw in ((0.0, 0.12, -0.12, 0.14), (0.45 * LT, -0.02, g + 0.05, 0.20), (0.9 * LT, g + 0.17, g + 0.04, 0.12), (LT, g + 0.10, g + 0.04, 0.035)):
            rings.append([V(x, hw, zt_), V(x, -hw, zt_), V(x, -hw, zb_), V(x, hw, zb_)])
        loft(P[CARB], rings)
        rings = []
        for x, zt_, hw in ((0.22 * LT, 0.10, 0.17), (0.5 * LT, 0.03, 0.235), (0.9 * LT, g + 0.215, 0.15), (1.02 * LT, g + 0.14, 0.04)):
            rings.append([V(x, hw, zt_), V(x, -hw, zt_), V(x, -hw, zt_ - 0.07), V(x, hw, zt_ - 0.07)])
        loft(P[CER], rings)
        box(P[RUB], V(0.42 * LT, -0.17, g), V(0.93 * LT, 0.17, g + 0.05))
        for q in range(4):
            box(P[RUB], V((0.46 + 0.11 * q) * LT, -0.19, g), V((0.52 + 0.11 * q) * LT, 0.19, g + 0.03))
        tube(P[STL + '!'], [V(0.12, 0, 0.16), V(0.5 * LT, 0, 0.07)], 0.035, 8)
        part('Toe_Blade_%s_%s' % (tname.upper(), k), 'toe_%s_%s' % (tname, k), P)
    # copies
    names = [n for n in list(PARTS) if n.endswith('_' + k)]
    for k2, lg in LEGS.items():
        if k2 == k:
            continue
        for n in names:
            b = PARTS[n][1]
            dup(n, n[:-2] + k2, b[:-2] + k2, lg['mir'])


def build_acts():
    for n, dd in ACT.items():
        rc = dd['rc']
        Pc, Pr = Pal(), Pal()
        disc(Pc[MET], 0, 0, -0.16, 0.16, rc * 1.3, 16)
        tube(Pc[MET], [V(0, 0, -rc), V(0, 0, -0.36)], rc * 0.8, 12)
        tube(Pc[MET], [V(0, 0, -0.32), V(0, 0, -dd['Lc'])], rc, 20)
        for z in (-0.32, -dd['Lc'] + 0.14):
            tube(Pc[MET], [V(0, 0, z), V(0, 0, z - 0.12)], rc * 1.2, 20)
        tube(Pc[CAB + '!'], smooth_path([V(rc * 0.9, 0, -0.5), V(rc + 0.10, 0, -0.62), V(rc + 0.12, 0, -dd['Lc'] * 0.6), V(rc * 0.9, 0, -dd['Lc'] + 0.25)], 5), 0.03, 6)
        disc(Pr[MET], 0, 0, -0.16, 0.16, rc * 1.15, 16)
        tube(Pr[MET], [V(0, 0, rc * 0.8), V(0, 0, 0.34)], rc * 0.75, 12)
        tube(Pr[STL + '!'], [V(0, 0, 0.3), V(0, 0, dd['Lr'])], rc * 0.55, 16)
        lb0 = dd['limbs'][0]
        nm = n.capitalize()
        part('Act_%s_Cyl_%s' % (nm, lb0), 'act_%s_top_%s' % (n, lb0), Pc)
        part('Act_%s_Rod_%s' % (nm, lb0), 'act_%s_bot_%s' % (n, lb0), Pr)
        for lb in dd['limbs'][1:]:
            mir = (LEGS[lb]['mir'] if lb in LEGS else ARMS[lb]['mir'])
            dup('Act_%s_Cyl_%s' % (nm, lb0), 'Act_%s_Cyl_%s' % (nm, lb), 'act_%s_top_%s' % (n, lb), mir)
            dup('Act_%s_Rod_%s' % (nm, lb0), 'Act_%s_Rod_%s' % (nm, lb), 'act_%s_bot_%s' % (n, lb), mir)


# ----------------------------------------------------------------------------------------------
# arm and hand (built once for L, mirrored to R)
# ----------------------------------------------------------------------------------------------
def loom(P, d, w, z0, z1):
    pts = [V(-d(z0 + (z1 - z0) * i / 10) / 2 - 0.07, 0.05 * sin(3.1 * i) - 0.06, z0 + (z1 - z0) * i / 10) for i in range(11)]
    tube(P[CAB + '!'], pts, 0.045, 8)
    tube(P[CAB + '!'], [p + V(0.0, 0.11, 0) for p in pts], 0.03, 6)
    for i in (1, 4, 7, 9):
        p = pts[i]
        box(P[MET], V(p.x - 0.07, p.y - 0.08, p.z - 0.05), V(p.x + 0.09, p.y + 0.17, p.z + 0.05))


def finger_seg(P, L, i, last):
    th, wy, rb = (0.24, 0.20, 0.17)[i], (0.28, 0.24, 0.20)[i], (0.15, 0.125, 0.105)[i]
    disc(P[MET], 0, 0, -wy / 2, wy / 2, rb, 16)
    for sg in (1, -1):
        y0, y1 = sorted((sg * wy / 2, sg * (wy / 2 + 0.035)))
        disc(P[MET], 0, 0, y0, y1, rb * 0.82, 16)
        boltsY(P[STL + '!'], 0, 0, rb * 0.55, 5, sg * (wy / 2 + 0.035), sg * 0.02, 0.016)
    if not last:
        tbox(P[CARB], [(V(0, 0, -0.06), th / 2, wy * 0.36), (V(0, 0, -(L - 0.10)), th * 0.42, wy * 0.33)])
        rings = []
        for z, hw, x0 in ((-0.20, wy * 0.40, th / 2 + 0.012), (-0.5 * L, wy * 0.48, th / 2 + 0.03), (-(L - 0.16), wy * 0.38, th * 0.42 + 0.012)):
            rings.append([V(x0 + 0.05, hw * 0.8, z), V(x0 + 0.05, -hw * 0.8, z), V(x0 - 0.03, -hw, z), V(x0 - 0.03, hw, z)])
        loft(P[CER], rings)
    else:
        rings = []
        for z, hx, hy in ((-0.06, th / 2, wy * 0.36), (-0.55 * L, th * 0.40, wy * 0.30), (-(L - 0.22), th * 0.30, wy * 0.24)):
            rings.append([V(hx, hy, z), V(-hx, hy, z), V(-hx, -hy, z), V(hx, -hy, z)])
        loft(P[CARB], rings)
        rings = []
        for z, hw, x0 in ((-0.18, wy * 0.40, th / 2 + 0.012), (-0.36 * L, wy * 0.44, th * 0.45 + 0.02), (-0.62 * L, wy * 0.30, th * 0.36 + 0.012)):
            rings.append([V(x0 + 0.045, hw * 0.8, z), V(x0 + 0.045, -hw * 0.8, z), V(x0 - 0.03, -hw, z), V(x0 - 0.03, hw, z)])
        loft(P[CER], rings)
        tube(P[RUB + '!'], [V(0, 0, -(L - 0.30)), V(-0.01, 0, -(L - 0.12)), V(-0.03, 0, -L)], [th * 0.34, th * 0.30, 0.03], 12)
        box(P[RUB], V(-th * 0.42, -wy * 0.22, -(L - 0.10)), V(-th * 0.20, wy * 0.22, -0.45 * L))
    tube(P[CAB + '!'], [V(-th / 2 - 0.02, 0.0, -0.10), V(-th * 0.45 - 0.02, 0.0, -(L - 0.14))], 0.022, 6)


def build_arm():
    sd = 'L'
    ae = ACT['elbow']
    P, Pc = Pal(), Pal()
    tube(P[MET], [V(-0.4, 0, 0), V(0.4, 0, 0)], 0.24, 20)
    tube(P[STL], [V(-0.66, 0, 0), V(0.66, 0, 0)], 0.10, 14)
    box(P[MET], V(-0.40, -0.67 * W_SH, -0.52), V(0.40, 0.67 * W_SH, -0.10))
    fork_side(P, Pc, -SH_DROP, R_SH, W_SH, SH_DROP - 0.48)
    part('Shoulder_Yoke_' + sd, 'shoulder_' + sd, P)
    part('Shoulder_Cap_' + sd, 'shoulder_' + sd, Pc)
    P, Pc, Ps = Pal(), Pal(), Pal()
    tongue_side(P, R_SH, W_SH, 1.0)
    d = truss(P, -0.85, -(LU - 0.72), 0.58, 0.44, 0.42, 9)
    fork_side(P, Pc, -LU, R_ELB, W_ELB, 0.85, capk=0.80)
    lug(P, (d(ae['tp'][1]) / 2 - 0.05, ae['tp'][1] + 0.45), ae['tp'], yw=0.14, rr=0.13)
    loom(P, d, 0.42, -1.0, -(LU - 0.9))
    Shell(d, lambda z: 0.42, 0.12, 0.0, radians(104), -1.15, -(LU - 0.95), tip0=0.10, tip1=0.12).build(
        Ps, P, panels=4, slots=((ae['tp'][1] + 0.85, ae['tp'][1] - 0.4),))
    part('Upperarm_Beam_' + sd, 'upperarm_' + sd, P)
    part('Elbow_Cap_' + sd, 'upperarm_' + sd, Pc)
    part('Upperarm_Shell_' + sd, 'upperarm_' + sd, Ps)
    P, Pc, Ps = Pal(), Pal(), Pal()
    tongue_side(P, R_ELB, W_ELB, 0.8)
    d = truss(P, -0.65, -(LF - 0.55), 0.44, 0.34, 0.36, 9)
    fork_side(P, Pc, -LF, R_WR, W_WR, 0.62, capk=0.76)
    lug(P, (d(ae['bp'][1]) / 2 - 0.05, ae['bp'][1] - 0.4), ae['bp'], yw=0.14, rr=0.13)
    loom(P, d, 0.36, -0.8, -(LF - 0.7))
    Shell(d, lambda z: 0.36, 0.11, 0.0, radians(104), -0.95, -(LF - 0.75), tip0=0.10, tip1=0.12).build(
        Ps, P, panels=4, slots=((-0.95, ae['bp'][1] - 0.8),))
    part('Forearm_Beam_' + sd, 'forearm_' + sd, P)
    part('Wrist_Cap_' + sd, 'forearm_' + sd, Pc)
    part('Forearm_Shell_' + sd, 'forearm_' + sd, Ps)
    # palm
    P, Ps = Pal(), Pal()
    tongue_side(P, R_WR, W_WR, 0.5)
    lathe(P[MET], [(0.20, -0.44), (0.31, -0.44), (0.31, -0.60), (0.20, -0.60)], I4, 24)
    for i in range(10):
        a = 2 * pi * i / 10
        hexbolt(P[STL + '!'], V(0.31 * cos(a), 0.31 * sin(a), -0.52), V(cos(a), sin(a), 0), 0.02, 0.02)
    rings = []
    for z, hx, hy in ((-0.56, 0.17, 0.26), (-0.85, 0.20, 0.50), (-(PALM - 0.14), 0.18, 0.60)):
        rings.append([V(hx, hy, z), V(-hx, hy, z), V(-hx, -hy, z), V(hx, -hy, z)])
    loft(P[CARB], rings)
    for y in (0.40, -0.40):                                # knuckle forks for fingers A and B
        for sg in (1, -1):
            box(P[MET], V(-0.20, y + sg * 0.185 - 0.03, -(PALM + 0.14)), V(0.14, y + sg * 0.185 + 0.03, -(PALM - 0.22)))
    for sg in (1, -1):                                     # thumb bracket
        bar(P[MET], (0.12, -0.62), (0.32, -0.92), 0.22, sg * 0.185 - 0.03, sg * 0.185 + 0.03)
    for sg in (1, -1):
        tube(P[CAB + '!'], smooth_path([V(-0.05, 0.2 * sg, -0.2), V(0.14, 0.2 * sg, -0.5), V(0.26, 0.3 * sg, -0.9), V(0.2, 0.4 * sg, -1.15)], 4), 0.035, 6)
    rings = []
    for z, hy in ((-0.50, 0.20), (-0.75, 0.40), (-(PALM - 0.20), 0.66), (-(PALM + 0.12), 0.50)):
        rings.append([V(-0.21, hy, z), V(-0.21, -hy, z), V(-0.29, -hy * 0.86, z), V(-0.29, hy * 0.86, z)])
    loft(Ps[CER], rings)
    box(Ps[CER], V(-0.305, -0.16, -0.98), V(-0.285, 0.16, -0.72))
    for y in (-0.3, 0.3):
        for z in (-0.72, -1.0):
            hexbolt(Ps[STL + '!'], V(-0.29, y, z), V(-1, 0, 0), 0.025, 0.02)
    part('Hand_Palm_' + sd, 'hand_' + sd, P)
    part('Hand_Shell_' + sd, 'hand_' + sd, Ps)
    for f in 'abc':
        for i in range(3):
            P = Pal()
            finger_seg(P, FING[f][i], i, i == 2)
            part('Finger_Seg_%s%d_%s' % (f.upper(), i + 1, sd), 'finger_%s%d_%s' % (f, i + 1, sd), P)
    names = [n for n in list(PARTS) if n.endswith('_' + sd)]
    for n in names:
        dup(n, n[:-1] + 'R', PARTS[n][1][:-1] + 'R', True)


# ----------------------------------------------------------------------------------------------
# body
# ----------------------------------------------------------------------------------------------
HX = [-4.2, -3.4, -2.2, -1.0, 0.4, 1.8, 3.0, 3.7, 4.0]
HW = [0.10, 0.70, 1.15, 1.30, 1.30, 1.20, 0.95, 0.55, 0.15]
HT = [22.5, 23.2, 23.7, 23.9, 24.0, 24.0, 23.9, 23.6, 23.2]
HB = [22.1, 21.4, 20.9, 20.6, 20.4, 20.5, 20.9, 21.6, 22.3]
HZM = 22.6


def hull_sec(x, n=24, grow=0.0):
    hw, zt, zb = spl(HX, HW, x) + grow, spl(HX, HT, x) + grow, spl(HX, HB, x) - grow
    out = []
    for i in range(n):
        a = 2 * pi * i / n
        c, s_ = cos(a), sin(a)
        if s_ >= 0:
            out.append(V(x, hw * c, HZM + (zt - HZM) * s_))
        else:
            out.append(V(x, hw * c * (1 - 0.45 * s_ * s_), HZM + (HZM - zb) * s_))
    return out


def ztop(x):
    return spl(HX, HT, x) if x >= -4.2 else 22.5 + (x + 4.2) * 1.1


def wing_pt(u, v):
    x = 4.0 - 8.4 * v
    A = V(x, 1.75 + 0.5 * v, 24.2 + 1.0 * sin(pi * min(1.0, max(v, 0.0) / 0.9) ** 0.9))
    Bp = V(x - 0.4 * v, 4.5 + 0.9 * v, 22.1 - 0.5 * v - 2.6 * v ** 3)
    kt = ss(0.55, 1.0, v)
    A2 = A.lerp(Bp, 0.96 * kt)
    C = V(x, Bp.y + 0.45, A2.z + 0.05).lerp((A2 + Bp) / 2, kt)
    p = (1 - u) ** 2 * A2 + 2 * u * (1 - u) * C + u * u * Bp
    return V(4.25, 2.3, 23.5).lerp(p, sin(0.5 * pi * min(1.0, max(v, 0.0) / 0.24)) ** 0.55)


def spine_pt(u, v):
    x = 2.3 - 7.3 * v
    hw = 0.95 * ss(-0.05, 0.12, v) * max(1 - ss(0.55, 1.0, v), 0.0) ** 0.8 + 0.03
    return V(x, u * hw, ztop(x) + 0.26 - 0.32 * u * u * (hw / 0.95))


def chest_pt(u, v):
    hw = (1.4 * max(1 - v, 0.0) ** 0.62 + 0.03) * (0.72 + 0.28 * ss(0.0, 0.12, v))
    y = u * hw
    x = (1 - v) ** 2 * 3.6 + 2 * v * (1 - v) * 5.0 + v * v * 2.9 - 0.5 * (y / 1.4) ** 2
    return V(x, y, 23.6 - 3.5 * v)


def build_body():
    P = Pal()
    loft(P[CARB], [hull_sec(HX[0] + (HX[-1] - HX[0]) * i / 22) for i in range(23)])
    for x in (-2.4, -0.9, 0.6, 2.1):                        # vent grilles on the flanks
        for q in range(5):
            hw = spl(HX, HW, x) * 0.985
            box(P[MET], V(x - 0.35, hw - 0.02, 22.75 + 0.11 * q), V(x + 0.35, hw + 0.05, 22.80 + 0.11 * q))
            box(P[MET], V(x - 0.35, -hw - 0.05, 22.75 + 0.11 * q), V(x + 0.35, -hw + 0.02, 22.80 + 0.11 * q))
    part('Body_Core', 'pelvis', P, world=True)
    # frame: ribs, rails, hip brackets, shoulder outriggers
    P = Pal()
    m = P[MET]
    for x in (-3.0, -2.0, -0.9, 0.3, 1.6, 2.7):
        sec = hull_sec(x, 28, 0.05)
        tube(m, sec + [sec[0]], 0.075, 8, caps=False)
    for sg in (1, -1):
        tube(m, [V(-3.3, sg * 1.2, 22.65), V(-2.0, sg * 1.42, 22.65), V(1.6, sg * 1.42, 22.65), V(3.2, sg * 1.05, 22.65)], 0.14, 12)
        tube(m, [V(-3.0, sg * 0.75, 23.55), V(0.0, sg * 0.9, 23.95), V(2.9, sg * 0.7, 23.8)], 0.10, 10)
    for k, lg in LEGS.items():
        pv = PEL + lg['piv']
        sg = lg['sd']
        for dx in (-0.66, 0.66):
            box(m, V(pv.x + dx - 0.08, sg * 1.25, pv.z - 0.34), V(pv.x + dx + 0.08, pv.y + sg * 0.42, pv.z + 0.34))
            tube(m, [V(pv.x + dx - 0.08, pv.y, pv.z), V(pv.x + dx + 0.08, pv.y, pv.z)], 0.34, 20)
            tube(m, [V(pv.x + dx, pv.y + sg * 0.3, pv.z + 0.3), V(pv.x + dx, sg * 0.8, 23.75)], 0.09, 8)
        tube(m, [V(pv.x - 0.9, sg * 1.42, pv.z), V(pv.x + 0.9, sg * 1.42, pv.z)], 0.17, 12)
    for sd, ar in ARMS.items():
        pv = PEL + ar['piv']
        sg = ar['sd']
        for dx in (-0.54, 0.54):
            box(m, V(pv.x + dx - 0.07, pv.y - sg * 0.3, pv.z - 0.28), V(pv.x + dx + 0.07, pv.y + sg * 0.34, pv.z + 0.28))
            tube(m, [V(pv.x + dx - 0.07, pv.y, pv.z), V(pv.x + dx + 0.07, pv.y, pv.z)], 0.28, 18)
            tube(m, [V(pv.x + dx, pv.y - sg * 0.25, pv.z + 0.1), V(pv.x + dx, sg * 0.85, 23.9)], 0.11, 10)
            tube(m, [V(pv.x + dx, pv.y - sg * 0.25, pv.z - 0.2), V(pv.x + dx, sg * 1.42, 22.65)], 0.10, 10)
        tube(m, [V(pv.x - 0.6, pv.y - sg * 0.3, pv.z), V(pv.x + 0.6, pv.y - sg * 0.3, pv.z)], 0.12, 10)
    nd = (NECK[1] - NECK[0]).normalized()                    # neck turret
    Mt = frame(NECK[0] - nd * 0.32, y=YAX, z=nd)
    lathe(m, [(0.50, 0.0), (0.86, 0.0), (0.86, 0.14), (0.74, 0.14), (0.74, 0.26), (0.50, 0.26)], Mt, 36)
    for i in range(18):
        a = 2 * pi * i / 18
        hexbolt(P[STL + '!'], Mt @ V(0.80 * cos(a), 0.80 * sin(a), 0.14), nd, 0.035, 0.035)
    tube(m, [NECK[0] - nd * 0.3, V(2.3, 0, 23.6)], 0.5, 16)
    part('Body_Frame', 'pelvis', P, world=True)
    # dorsal shell
    P = Pal()
    ctr = V(0, 0, 22.0)
    for v0, v1 in ((0.0, 0.335), (0.345, 0.665), (0.675, 1.0)):
        for u0, u1 in ((-1.0, -0.02), (0.02, 1.0)):
            surf_panel(P[CER], spine_pt, u0, u1, v0, v1, 6, 10, 0.09, ctr)
    for v in (0.05, 0.3, 0.38, 0.63, 0.71):
        for u in (-0.8, 0.8):
            hexbolt(P[STL + '!'], spine_pt(u, v), V(0, u, 1), 0.035, 0.03)
    surf_panel(P[CER], spine_pt, -0.35, 0.35, 0.14, 0.22, 3, 3, 0.03, ctr, lift=0.014)
    part('Body_Spine_Shell', 'pelvis', P, world=True)
    # chest shield
    P = Pal()
    ctr = V(1.5, 0, 22.0)
    for sgn in (1, -1):
        surf_panel(P[CER], chest_pt, sgn * 0.018, sgn * 1.0, 0.0, 1.0, 8, 16, 0.10, ctr)
        for q in range(4):
            surf_panel(P[CARB], chest_pt, sgn * 0.42, sgn * 0.78, 0.08 + 0.035 * q, 0.10 + 0.035 * q, 3, 1, 0.02, ctr, lift=0.008)
        for v, u in ((0.05, 0.88), (0.3, 0.86), (0.55, 0.82), (0.8, 0.7), (0.05, 0.12), (0.45, 0.14), (0.85, 0.2)):
            p = chest_pt(sgn * u, v)
            hexbolt(P[STL + '!'], p, p - ctr, 0.035, 0.03)
        for v, u in ((0.15, 0.5), (0.5, 0.5), (0.8, 0.4)):   # stand-offs back to the hull
            p = chest_pt(sgn * u, v)
            tube(P[MET], [p + V(-0.08, 0, 0), V(p.x - 0.9, p.y * 0.8, p.z)], 0.06, 8)
    part('Chest_Shield', 'pelvis', P, world=True)
    # wings (L built, R mirrored)
    P, Pm = Pal(), Pal()
    ctr = V(0.0, 1.2, 21.5)
    for i, (v0, v1) in enumerate(((0.0, 0.36), (0.335, 0.68), (0.655, 1.0))):
        surf_panel(P[CER], wing_pt, 0.0, 1.0, v0, v1, 10, 12, 0.10, ctr, lift=-0.09 * i)
        for u, v in ((0.08, v0 + 0.04), (0.92, v0 + 0.04), (0.5, v0 + 0.04), (0.08, v1 - 0.06), (0.9, v1 - 0.06)):
            if v < 0.93:
                p = wing_pt(u, v)
                hexbolt(P[STL + '!'], p + (p - ctr).normalized() * (-0.09 * i), p - ctr, 0.04, 0.035)
    surf_panel(P[CER], wing_pt, 0.45, 0.75, 0.16, 0.24, 3, 3, 0.03, ctr, lift=0.014)
    for u, v, base in ((0.30, 0.10, V(3.0, 1.2, 23.6)), (0.45, 0.20, V(2.6, 3.0, 24.5)), (0.35, 0.45, V(0.2, 1.42, 23.0)),
                       (0.55, 0.50, V(0.0, 1.42, 22.65)), (0.40, 0.72, V(-2.0, 1.42, 22.9)), (0.55, 0.80, V(-2.6, 2.6, 22.3))):
        p = wing_pt(u, v)
        n = (p - ctr).normalized()
        tube(Pm[MET], [base, p - n * 0.2], 0.07, 8)
        tube(Pm[MET], [p - n * 0.24, p - n * 0.10], 0.14, 10)
    part('Wing_Shell_L', 'pelvis', P, world=True)
    part('Wing_Struts_L', 'pelvis', Pm, world=True)
    dup('Wing_Shell_L', 'Wing_Shell_R', 'pelvis', True)
    dup('Wing_Struts_L', 'Wing_Struts_R', 'pelvis', True)
    # belly cables
    P = Pal()
    for p0, p1, sag, r in ((V(1.5, 0.75, 20.75), V(-1.7, 0.8, 21.0), 2.6, 0.06), (V(1.5, -0.75, 20.75), V(-1.7, -0.8, 21.0), 3.4, 0.06),
                           (V(1.9, 0.5, 20.9), V(-0.6, -0.6, 20.6), 4.1, 0.05), (V(0.6, 0.7, 20.5), V(-2.3, 0.2, 21.2), 1.7, 0.045),
                           (V(2.4, -0.4, 21.2), V(0.2, 0.5, 20.45), 1.3, 0.045)):
        tube(P[CAB + '!'], catenary(p0, p1, sag, 18), r, 8)
        for p in (p0, p1):
            tube(P[MET], [p + V(0, 0, 0.25), p + V(0, 0, -0.12)], r * 1.7, 10)
    part('Belly_Cables', 'pelvis', P, world=True)


# ----------------------------------------------------------------------------------------------
# neck and head
# ----------------------------------------------------------------------------------------------
NPTS = NECK + [HYAW]
NDEP = [1.25, 1.05, 0.88, 0.74, 0.62, 0.55]
NWID = [0.95, 0.82, 0.70, 0.60, 0.52, 0.48]


def build_neck():
    for i in range(5):
        a, b = NPTS[i], NPTS[i + 1]
        z = (b - a).normalized()
        Ml = frame(a, y=YAX, z=z)
        ln = (b - a).length
        d0, d1, w0, w1 = NDEP[i], NDEP[i + 1], NWID[i], NWID[i + 1]
        P = Pal()
        c = P[CARB]
        for sg in (1, -1):
            y0, y1 = sorted((sg * (w0 / 2 - 0.07), sg * w0 / 2))
            for sx in (1, -1):
                bar(c, (sx * (d0 / 2 - 0.07), 0.0), (sx * (d1 / 2 - 0.07), ln), 0.13, y0, y1)
            bar(c, (-(d0 / 2 - 0.1), 0.05), ((d1 / 2 - 0.1), ln - 0.05), 0.09, y0, y1)
            bar(c, ((d0 / 2 - 0.1), 0.05), (-(d1 / 2 - 0.1), ln - 0.05), 0.09, y0, y1)
            bar(c, (-(d0 / 2), 0.0), (d0 / 2, 0.0), 0.16, y0, y1)
            disc(P[MET], 0, 0, y0, y1, 0.24, 16)
            y2, y3 = sorted((sg * w0 / 2, sg * (w0 / 2 + 0.06)))
            disc(P[MET], 0, 0, y2, y3, 0.17, 14)
        disc(P[STL], 0, 0, -w0 / 2 - 0.09, w0 / 2 + 0.09, 0.09, 12)
        for zz in (0.25 * ln, 0.75 * ln):
            tube(c, [V(-d0 * 0.35, -w0 / 2 + 0.03, zz), V(-d0 * 0.35, w0 / 2 - 0.03, zz)], 0.05, 8)
        sh = Shell(lambda zz: d0 + (d1 - d0) * zz / ln, lambda zz: w0 + (w1 - w0) * zz / ln, 0.10, pi, radians(100), 0.10, ln - 0.03, tip0=0.2, tip1=0.15)
        sh.panel(P[CER], 0.10, ln - 0.03, -1.0, 1.0, 0.06, 0.0, 6, 6)
        tube(P[MET], [V(-d0 / 2 + 0.02, 0, ln / 2), V(-d0 / 2 - 0.06, 0, ln / 2)], 0.05, 8)
        for key in list(P):
            P[key].transform(Ml)
        part('Neck_Link_%d' % (i + 1), 'neck_%02d' % (i + 1), P, world=True)
    # cable bundle along the ventral side, smooth-skinned
    P = Pal()
    for yk, rr, dx in ((-0.30, 0.07, 0.06), (-0.10, 0.085, 0.12), (0.10, 0.085, 0.12), (0.30, 0.07, 0.06), (0.0, 0.05, 0.22)):
        pts = []
        for i in range(6):
            z = (NPTS[min(i + 1, 5)] - NPTS[max(i - 1, 0)]).normalized()
            x = YAX.cross(z)
            pts.append(NPTS[i] + x * (NDEP[i] / 2 + dx) + YAX * (yk * NWID[i]))
        pts = [pts[0] - (pts[1] - pts[0]) * 0.2] + pts
        tube(P[CAB + '!'], smooth_path(pts, 5), rr, 8)
    part('Neck_Bundle', 'neck_01', P, world=True)


HDX = [-1.4, -0.8, 0.0, 1.0, 2.0, 3.0, 3.6]
HDW = [0.25, 0.50, 0.62, 0.56, 0.40, 0.18, 0.03]
HDT = [0.30, 0.48, 0.52, 0.42, 0.28, 0.13, 0.04]
HDB = [0.06, -0.20, -0.30, -0.22, -0.10, -0.02, 0.0]


def head_pt(x, a, grow=0.0):
    hw, zt, zb = spl(HDX, HDW, x) + grow, spl(HDX, HDT, x) + grow, spl(HDX, HDB, x) - grow
    zm = zb + 0.42 * (zt - zb)
    c, s_ = cos(a), sin(a)
    y = hw * (abs(c) ** 0.75) * (1 if c >= 0 else -1)
    return V(x, y, zm + ((zt - zm) if s_ >= 0 else (zm - zb)) * (abs(s_) ** 0.9) * (1 if s_ >= 0 else -1))


def build_head():
    P = Pal()
    m = P[MET]
    lathe(m, [(0.20, -0.10), (0.46, -0.10), (0.46, 0.04), (0.38, 0.04), (0.38, 0.14), (0.20, 0.14)], T(HYAW), 28)
    for i in range(12):
        a = 2 * pi * i / 12
        hexbolt(P[STL + '!'], HYAW + V(0.42 * cos(a), 0.42 * sin(a), 0.04), ZAX, 0.028, 0.03)
    for sg in (1, -1):
        y0, y1 = sorted((sg * 0.30, sg * 0.38))
        box(m, V(HYAW.x - 0.16, y0, HYAW.z + 0.10), V(HYAW.x + 0.16, y1, HPIV.z))
        tube(m, [V(HPIV.x, y0, HPIV.z), V(HPIV.x, y1, HPIV.z)], 0.20, 16)
    tube(P[STL], [HPIV + V(0, -0.44, 0), HPIV + V(0, 0.44, 0)], 0.075, 12)
    part('Head_Yoke', 'head_yaw', P, world=True)
    P = Pal()
    xs_ = [HDX[0] + (HDX[-1] - HDX[0]) * i / 20 for i in range(21)]
    loft(P[CARB], [[HPIV + head_pt(x, 2 * pi * j / 20) for j in range(20)] for x in xs_])
    box(P[MET], HPIV + V(-0.22, -0.28, -0.34), HPIV + V(0.22, 0.28, -0.05))
    for q in range(4):
        box(P[MET], HPIV + V(0.6 + 0.16 * q, -0.22, -0.30), HPIV + V(0.68 + 0.16 * q, 0.22, -0.24))
    part('Head_Core', 'head', P, world=True)
    P = Pal()

    def top(u, v):
        return HPIV + head_pt(-1.25 + 4.8 * v, radians(90 - 80 * u), 0.05)
    ctr = HPIV + V(1.0, 0, -0.2)
    for v0, v1 in ((0.0, 0.445), (0.455, 1.0)):
        for u0, u1 in ((-1.0, -0.03), (0.03, 1.0)):
            surf_panel(P[CER], top, u0, u1, v0, v1, 6, 10, 0.05, ctr)
    for v, u in ((0.06, 0.7), (0.40, 0.8), (0.50, 0.8), (0.85, 0.6)):
        for sg in (1, -1):
            p = top(sg * u, v)
            hexbolt(P[STL + '!'], p, p - ctr, 0.022, 0.02)
    surf_panel(P[CARB], top, -0.3, 0.3, 0.62, 0.66, 3, 1, 0.02, ctr, lift=0.008)
    part('Head_Shell', 'head', P, world=True)
    # antennae: three ceramic blades each side, one thin whisker each
    P = Pal()
    for sg in (1, -1):
        for a, b, bulge in ((V(-0.55, 0.30, 0.44), V(-4.0, 1.25, 1.12), 0.35), (V(-0.75, 0.40, 0.34), V(-3.8, 2.15, 0.78), 0.55),
                            (V(-0.95, 0.42, 0.22), V(-3.2, 2.70, 0.40), 0.65)):
            a, b = V(a.x, a.y * sg, a.z), V(b.x, b.y * sg, b.z)
            mid = a.lerp(b, 0.45) + V(0.3, bulge * sg, 0.10)
            blade(P[CER], [HPIV + p for p in smooth_path([a, mid, b], 8)], (0.11, 0.02), (0.05, 0.015))
            tube(P[MET], [HPIV + a + V(0.25, -0.1 * sg, -0.12), HPIV + a], 0.07, 8)
        w = HPIV + V(-2.6, 1.75 * sg, 0.62)
        tube(P[STL + '!'], [w + V(0, 0, 0.22), w + V(0, 0, -0.95)], 0.018, 6)
    part('Head_Antennae', 'head', P, world=True)
    P = Pal()
    for sg in (1, -1):
        tube(P[CAB + '!'], catenary(HPIV + V(1.9, 0.18 * sg, -0.12), HPIV + V(0.75, 0.24 * sg, -0.24), 0.75 if sg > 0 else 0.5, 14), 0.035, 8)
        tube(P[MET], [HPIV + V(0.75, 0.24 * sg, -0.14), HPIV + V(0.75, 0.24 * sg, -0.32)], 0.06, 8)
    part('Head_Cables', 'head', P, world=True)
    # sensors: bezels on the head, lights on their own bone
    P, Pe = Pal(), Pal()
    for sg in (1, -1):
        for x, da in ((1.55, 14), (1.82, 2), (2.08, 15), (2.34, 3), (2.58, 13), (1.95, 30), (2.22, 31)):
            a = radians(da)
            p = head_pt(x, a)
            e = 1e-3
            n = (head_pt(x + e, a) - head_pt(x - e, a)).cross(head_pt(x, a + 0.02) - head_pt(x, a - 0.02)).normalized()
            if n.y < 0:
                n = -n
            p.y *= sg
            n.y *= sg
            xa = (XAX - n * XAX.dot(n)).normalized()
            Ms = frame(HPIV + p - n * 0.012, x=xa, z=n)
            lathe(P[MET], [(0.075, 0.0), (0.105, 0.0), (0.105, 0.04), (0.075, 0.04)], Ms, 16)
            lathe(Pe[EMIT + '!'], [(0.002, 0.0), (0.075, 0.0), (0.06, 0.035), (0.03, 0.05), (0.002, 0.052)], Ms, 14)
    for x in (2.75, 2.95, 3.12):
        p = HPIV + head_pt(x, radians(90), 0.05)
        lathe(P[MET], [(0.04, 0.0), (0.06, 0.0), (0.06, 0.03), (0.04, 0.03)], T(p), 12)
        lathe(Pe[EMIT + '!'], [(0.002, 0.0), (0.04, 0.0), (0.03, 0.03), (0.002, 0.035)], T(p), 10)
    part('Sensor_Bezels', 'head', P, world=True)
    part('Light_Sensors', 'sensor', Pe, world=True)


build_leg()
build_arm()
build_acts()
build_body()
build_neck()
build_head()
print('BUILT', len(PARTS), 'parts,', len(CANON), 'unique', round(time.time() - T0, 1), 's')

# ----------------------------------------------------------------------------------------------
# UVs. UV_Atlas is channel 0 (what glTF viewers and UE read first): two packed sets, ceramic and frame.
# UV_Tile is channel 1: box projection at 1 UV unit = 1 m for tiling materials.
# Only the unique parts are unwrapped; the copies made afterwards share their islands.
# ----------------------------------------------------------------------------------------------
def box_uv(ob):
    me = ob.data
    uv = me.uv_layers.new(name='UV_Tile')
    for p in me.polygons:
        n = p.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for li in p.loop_indices:
            c = me.vertices[me.loops[li].vertex_index].co
            uv.data[li].uv = (c.y, c.z) if ax == 0 else ((c.x, c.z) if ax == 1 else (c.x, c.y))


for ob in CANON:
    ob.data.uv_layers.new(name='UV_Atlas')
    box_uv(ob)
bpy.ops.object.select_all(action='DESELECT')
for ob in CANON:
    ob.data.uv_layers.active = ob.data.uv_layers['UV_Atlas']
    ob.select_set(True)
bpy.context.view_layer.objects.active = CANON[0]
ATLAS_OK = True
try:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=radians(66), island_margin=0.0, area_weight=0.0, scale_to_bounds=False)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.average_islands_scale()
    for want_cer in (True, False):
        bpy.ops.object.mode_set(mode='OBJECT')
        for ob in CANON:
            me = ob.data
            is_cer = [m.name == CER for m in me.materials]
            me.polygons.foreach_set('hide', [is_cer[p.material_index] != want_cer for p in me.polygons])
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.pack_islands(rotate=True, margin=0.0025)
        bpy.ops.mesh.reveal()
    bpy.ops.object.mode_set(mode='OBJECT')
except Exception as e:
    print('ATLAS FAILED', e)
    ATLAS_OK = False
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
for ob in CANON:
    ob.data.polygons.foreach_set('hide', [False] * len(ob.data.polygons))
    ob.data.uv_layers.active = ob.data.uv_layers['UV_Atlas']
    ob.data.uv_layers['UV_Atlas'].active_render = True
# The bake belongs to one exact UV layout, and packing is not stable against the smallest geometry change. The layout
# that was baked is kept beside the textures and loaded back on --reuse-bake; a topology change fails loudly.
UV_CACHE = os.path.join(HERE, 'textures' if NAME == OUT else 'textures_' + NAME, OUT + '_uv_atlas.npz')
if '--reuse-bake' in ARGS and os.path.exists(UV_CACHE):
    _uvc = np.load(UV_CACHE)
    for ob in CANON:
        uvl = ob.data.uv_layers['UV_Atlas'].data
        assert ob.name in _uvc and len(_uvc[ob.name]) == len(uvl) * 2, 'geometry changed since the bake (%s): run without --reuse-bake' % ob.name
        uvl.foreach_set('uv', _uvc[ob.name])
    UV_REUSED = True
else:
    UV_REUSED = False
for a in DUPS:
    make_dup(*a)
ALL_OBS = [ob for ob, _ in PARTS.values()]
print('UV', ATLAS_OK, len(ALL_OBS), 'objects', round(time.time() - T0, 1), 's')

# ----------------------------------------------------------------------------------------------
# bake: procedural materials -> textures/stilt_<set>_<map>.png on UV_Atlas, then plain image materials
# ----------------------------------------------------------------------------------------------
TEX_DIR = os.path.join(HERE, 'textures' if NAME == OUT else 'textures_' + NAME)
TEX = {}
BAKE_INFO = {}
FRAME = 'M_Stilt_Frame'
_w = bpy.data.worlds.new('World')
SCN.world = _w


def bake_textures():
    os.makedirs(TEX_DIR, exist_ok=True)
    # one joined copy of the unique parts is baked (a single bake per map instead of one per object); the
    # originals are hidden meanwhile so they do not occlude their own copy. The other limbs stay visible for the AO.
    bpy.ops.object.select_all(action='DESELECT')
    cps = []
    for ob in CANON:
        c = ob.copy()
        c.data = ob.data.copy()
        COL.objects.link(c)
        c.select_set(True)
        cps.append(c)
        ob.hide_render = True
    bpy.context.view_layer.objects.active = cps[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    obs = [joined]
    used = sorted({m.name for ob in obs for m in ob.data.materials})
    SCN.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for t in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = t
                prefs.get_devices()
                for dv in prefs.devices:
                    dv.use = True
                SCN.cycles.device = 'GPU'
                break
            except Exception:
                pass
    except Exception as e:
        print('gpu', e)
    SCN.cycles.samples = 24
    SCN.cycles.use_denoising = False
    SCN.world.light_settings.distance = 3.0
    N = TEX_N

    def new_img(nm, srgb, fill):
        im = bpy.data.images.new(nm, N, N, alpha=False)
        im.colorspace_settings.name = 'sRGB' if srgb else 'Non-Color'
        px = np.empty((N * N, 4), dtype=np.float32)
        px[:] = fill
        im.pixels.foreach_set(px.ravel())
        return im

    def get(im):
        px = np.empty(N * N * 4, dtype=np.float32)
        im.pixels.foreach_get(px)
        return px.reshape(-1, 4)

    sets = ('ceramic', 'frame')
    IM = {s: dict(base=new_img('stilt_%s_basecolor' % s, True, (0.05, 0.05, 0.05, 1)), rough=new_img('r_' + s, False, (0.5, 0.5, 0.5, 1)),
                  metal=new_img('m_' + s, False, (0, 0, 0, 1)), ao=new_img('a_' + s, False, (1, 1, 1, 1)),
                  normal=new_img('stilt_%s_normal' % s, False, (0.5, 0.5, 1.0, 1)), orm=new_img('stilt_%s_orm' % s, False, (1, 0.5, 0, 1))) for s in sets}
    tex = {n: MATS[n].node_tree.nodes.new('ShaderNodeTexImage') for n in used}

    def target(kind):
        for n in used:
            tex[n].image = IM['ceramic' if n == CER else 'frame'][kind]
            MATS[n].node_tree.nodes.active = tex[n]
        bpy.ops.object.select_all(action='DESELECT')
        for ob in obs:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = obs[0]

    def emit_bake(kind, socket, is_color):
        saved = []
        for n in used:
            nt = MATS[n].node_tree
            out = nt.nodes['Material Output']
            inp = nt.nodes['Principled BSDF'].inputs[socket]
            old = out.inputs['Surface'].links[0].from_socket
            em = nt.nodes.new('ShaderNodeEmission')
            if inp.is_linked:
                nt.links.new(inp.links[0].from_socket, em.inputs['Color'])
            else:
                v = inp.default_value
                em.inputs['Color'].default_value = tuple(v) if is_color else (v, v, v, 1.0)
            nt.links.new(em.outputs[0], out.inputs['Surface'])
            saved.append((nt, out, old, em))
        target(kind)
        bpy.ops.object.bake(type='EMIT', margin=6, use_clear=False)
        print('bake', kind, round(time.time() - T0, 1), 's', flush=True)
        for nt, out, old, em in saved:
            nt.links.new(old, out.inputs['Surface'])
            nt.nodes.remove(em)

    emit_bake('base', 'Base Color', True)
    emit_bake('rough', 'Roughness', False)
    emit_bake('metal', 'Metallic', False)
    target('normal')
    bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', margin=6, use_clear=False)
    SCN.cycles.samples = 48
    target('ao')
    bpy.ops.object.bake(type='AO', margin=6, use_clear=False)
    for s in sets:
        ao = np.clip(get(IM[s]['ao'])[:, 0], 0, 1)
        r, m = get(IM[s]['rough']), get(IM[s]['metal'])
        base = get(IM[s]['base'])
        k = (0.30 + 0.70 * ao ** 1.2) if s == 'ceramic' else (0.45 + 0.55 * ao)
        base[:, :3] *= k[:, None]
        IM[s]['base'].pixels.foreach_set(base.ravel())
        px = np.ones_like(r)
        px[:, 0], px[:, 1], px[:, 2] = ao, r[:, 0], m[:, 0]
        IM[s]['orm'].pixels.foreach_set(px.ravel())
        for key, kk in (('basecolor', 'base'), ('orm', 'orm'), ('normal', 'normal')):
            im = IM[s][kk]
            im.filepath_raw = os.path.join(TEX_DIR, 'stilt_%s_%s.png' % (s, key))
            im.file_format = 'PNG'
            im.save()
            TEX[(s, key)] = im
        for kk in ('rough', 'metal', 'ao'):
            bpy.data.images.remove(IM[s][kk])
    for n in used:
        MATS[n].node_tree.nodes.remove(tex[n])
    a3 = sum(p.area for ob in obs for p in ob.data.polygons)
    bpy.data.objects.remove(joined)
    for ob in CANON:
        ob.hide_render = False
    BAKE_INFO.update(size=N, unique_area_m2=round(a3, 1), sets=list(sets))


def textured(m, s):
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    uv = nt.nodes.new('ShaderNodeUVMap')
    uv.uv_map = 'UV_Atlas'
    L = nt.links.new
    imgs = {}
    for key in ('basecolor', 'orm', 'normal'):
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = TEX[(s, key)]
        L(uv.outputs['UV'], t.inputs['Vector'])
        imgs[key] = t
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    nm = nt.nodes.new('ShaderNodeNormalMap')
    nm.uv_map = 'UV_Atlas'
    L(imgs['basecolor'].outputs['Color'], b.inputs['Base Color'])
    L(imgs['orm'].outputs['Color'], sep.inputs['Color'])
    L(sep.outputs['Green'], b.inputs['Roughness'])
    L(sep.outputs['Blue'], b.inputs['Metallic'])
    L(imgs['normal'].outputs['Color'], nm.inputs['Color'])
    L(nm.outputs['Normal'], b.inputs['Normal'])
    L(b.outputs[0], out.inputs['Surface'])


BAKED = '--no-bake' not in ARGS and ATLAS_OK
REUSE = UV_REUSED and all(os.path.exists(os.path.join(TEX_DIR, 'stilt_%s_%s.png' % (s_, k_)))
                                       for s_ in ('ceramic', 'frame') for k_ in ('basecolor', 'orm', 'normal'))
if BAKED:
    if REUSE:                       # geometry and UVs unchanged since the last bake: load its textures
        for s_ in ('ceramic', 'frame'):
            for k_ in ('basecolor', 'orm', 'normal'):
                im = bpy.data.images.load(os.path.join(TEX_DIR, 'stilt_%s_%s.png' % (s_, k_)))
                im.colorspace_settings.name = 'sRGB' if k_ == 'basecolor' else 'Non-Color'
                TEX[(s_, k_)] = im
        BAKE_INFO.update(size=TEX[('ceramic', 'basecolor')].size[0], sets=['ceramic', 'frame'], reused=True)
    else:
        bake_textures()
        _d = {}
        for ob in CANON:
            uvl = ob.data.uv_layers['UV_Atlas'].data
            buf = np.empty(len(uvl) * 2, dtype=np.float32)
            uvl.foreach_get('uv', buf)
            _d[ob.name] = buf
        np.savez_compressed(UV_CACHE, **_d)
    textured(MATS[CER], 'ceramic')
    mf = bpy.data.materials.new(FRAME)
    mf.use_nodes = True
    MATS[FRAME] = mf
    textured(mf, 'frame')
    for ob in ALL_OBS:                      # the five frame materials collapse into one textured slot
        me = ob.data
        old = [m.name for m in me.materials]
        new = []
        remap = []
        for n in old:
            n2 = FRAME if n in FRAME_SET else n
            if n2 not in new:
                new.append(n2)
            remap.append(new.index(n2))
        idx = np.empty(len(me.polygons), dtype=np.int32)
        me.polygons.foreach_get('material_index', idx)
        idx = np.array(remap, dtype=np.int32)[idx]
        me.materials.clear()
        for n in new:
            me.materials.append(MATS[n])
        me.polygons.foreach_set('material_index', idx)
        me.update()
    print('BAKED', json.dumps(BAKE_INFO), round(time.time() - T0, 1), 's')

# ----------------------------------------------------------------------------------------------
# armature, skinning
# ----------------------------------------------------------------------------------------------
arm_data = bpy.data.armatures.new('Armature')
ARM = bpy.data.objects.new('Armature', arm_data)      # the name "Armature" makes UE drop the node and use `root`
COL.objects.link(ARM)
bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = ARM
ARM.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
for n in BONE_ORDER:
    eb = arm_data.edit_bones.new(n)
    fr = REST[n]
    o, r = fr.to_translation(), fr.to_3x3()
    eb.head = o
    eb.tail = o + r.col[1] * (0.35 if n.startswith(('finger', 'toe', 'act', 'sensor')) else 0.8)
    eb.align_roll(r.col[2])
    if PARENT[n]:
        eb.parent = arm_data.edit_bones[PARENT[n]]
    eb.use_connect = False
bpy.ops.object.mode_set(mode='OBJECT')
arm_data.display_type = 'STICK'
ARM.show_in_front = True
PB = ARM.pose.bones
for pb in PB:
    pb.rotation_mode = 'QUATERNION'
ARM['sensor_light'] = 1.0
ARM.id_properties_ui('sensor_light').update(min=0.0, max=4.0, description='Sensor light level; 1 = calm')


def neck_weights(co):
    best, bi, bt = 1e9, 0, 0.0
    for i in range(5):
        a, b = NPTS[i], NPTS[i + 1]
        t = max(0.0, min(1.0, (co - a).dot(b - a) / (b - a).length_squared))
        dd = (co - (a + (b - a) * t)).length
        if dd < best:
            best, bi, bt = dd, i, t
    w = {}
    if bt < 0.3 and bi > 0:
        k = 0.5 + 0.5 * bt / 0.3
        w = {bi: k, bi - 1: 1 - k}
    elif bt > 0.7 and bi < 4:
        k = 0.5 + 0.5 * (1 - bt) / 0.3
        w = {bi: k, bi + 1: 1 - k}
    else:
        w = {bi: 1.0}
    return w


tris_total = 0
info_parts = []
for name, (ob, bone) in PARTS.items():
    me = ob.data
    if name == 'Neck_Bundle':
        groups = [ob.vertex_groups.new(name='neck_%02d' % (i + 1)) for i in range(5)]
        mw = ob.matrix_world
        for v in me.vertices:
            for i, wt in neck_weights(mw @ v.co).items():
                groups[i].add([v.index], wt, 'REPLACE')
    else:
        vg = ob.vertex_groups.new(name=bone)
        vg.add(list(range(len(me.vertices))), 1.0, 'REPLACE')
    mw = ob.matrix_world.copy()
    ob.parent = ARM
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.matrix_world = mw
    t = ob.modifiers.new('Triangulate', 'TRIANGULATE')
    t.keep_custom_normals = True
    t.quad_method = 'SHORTEST_DIAGONAL'
    a = ob.modifiers.new('Armature', 'ARMATURE')
    a.object = ARM
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    tris_total += tris
    o = mw.to_translation()
    info_parts.append(dict(name=name, bone=bone if name != 'Neck_Bundle' else 'neck_01..05 (smooth)', tris=tris, unique=ob in CANON,
                           origin=[round(o.x, 3), round(o.y, 3), round(o.z, 3)], materials=[m.name for m in me.materials]))
print('SKINNED tris', tris_total, round(time.time() - T0, 1), 's')

# ----------------------------------------------------------------------------------------------
# actions (keys every second frame, solved above)
# ----------------------------------------------------------------------------------------------
SCN.render.fps = 24
LOCB = {'pelvis'} | {b for b in BONE_ORDER if b.startswith('ik_')}
ACTS = {}
RANGES = {}
for aname, fn, n in ACTIONS:
    act = bpy.data.actions.new(aname)
    act.use_fake_user = True
    if ARM.animation_data is None:
        ARM.animation_data_create()
    ARM.animation_data.action = act
    prev = {}
    frames = list(range(1, n + 1, 2))
    if frames[-1] != n:
        frames.append(n)
    lo = {k: 1e9 for k in ACT}
    hi = {k: 0.0 for k in ACT}
    for f in frames:
        st = fn(f)
        W = add_acts(solve(st))
        for k, dd in ACT.items():
            for lb in dd['limbs']:
                ln = (W['act_%s_top_%s' % (k, lb)].to_translation() - W['act_%s_bot_%s' % (k, lb)].to_translation()).length
                lo[k], hi[k] = min(lo[k], ln), max(hi[k], ln)
        for b in BONE_ORDER:
            if b == 'root':
                continue
            basis = LOCREST[b].inverted() @ (W[PARENT[b]].inverted() @ W[b])
            q = basis.to_quaternion()
            if b in prev and prev[b].dot(q) < 0:
                q.negate()
            prev[b] = q
            pb = PB[b]
            pb.rotation_quaternion = q
            pb.keyframe_insert('rotation_quaternion', frame=f)
            if b in LOCB:
                pb.location = basis.to_translation()
                pb.keyframe_insert('location', frame=f)
        ARM['sensor_light'] = st['light']
        ARM.keyframe_insert('["sensor_light"]', frame=f)
    ACTS[aname] = act
    RANGES[aname] = dict(frames=n, strokes={k: [round(lo[k], 3), round(hi[k], 3)] for k in ACT})
    for k, dd in ACT.items():
        assert lo[k] >= dd['Lc'] + 0.2 and hi[k] <= dd['Lc'] + dd['Lr'] - 0.2 or not dd['ok'], ('actuator stroke', aname, k, lo[k], hi[k])
    print('ACTION', aname, n, 'frames', round(time.time() - T0, 1), 's')
A_WALK = ACTS['A_Stilt_Walk']
ARM.animation_data.action = A_WALK
try:
    ARM.animation_data.action_slot = A_WALK.slots[0]
except Exception as e:
    print('slot', e)
SCN.frame_start, SCN.frame_end = 1, WALK_N
SCN.frame_set(1)


def drive(owner, prop, expr):
    fc = owner.driver_add(prop)
    dr = fc.driver
    dr.type = 'SCRIPTED'
    v = dr.variables.new()
    v.name = 'k'
    v.targets[0].id = ARM
    v.targets[0].data_path = '["sensor_light"]'
    dr.expression = expr


drive(MATS[EMIT].node_tree.nodes['Principled BSDF'].inputs['Emission Strength'], 'default_value', 'k*6.0')

# ----------------------------------------------------------------------------------------------
# studio (not exported): backdrop, lights, camera, a 1.75 m figure for scale
# ----------------------------------------------------------------------------------------------
COL_ST = bpy.data.collections.new('Studio')
SCN.collection.children.link(COL_ST)
_w.use_nodes = True
_nt = _w.node_tree
_bg = _nt.nodes['Background']
_bg2 = _nt.nodes.new('ShaderNodeBackground')
_mix = _nt.nodes.new('ShaderNodeMixShader')
_lp = _nt.nodes.new('ShaderNodeLightPath')
_bg.inputs['Color'].default_value = (0.62, 0.64, 0.68, 1)
_bg.inputs['Strength'].default_value = 0.45
_bg2.inputs['Color'].default_value = (0.215, 0.215, 0.215, 1)
_nt.links.new(_lp.outputs['Is Camera Ray'], _mix.inputs['Fac'])
_nt.links.new(_bg.outputs[0], _mix.inputs[1])
_nt.links.new(_bg2.outputs[0], _mix.inputs[2])
_nt.links.new(_mix.outputs[0], _nt.nodes['World Output'].inputs['Surface'])
_fm = bpy.data.meshes.new('Floor')
_fm.from_pydata([(-600, -600, 0), (600, -600, 0), (600, 600, 0), (-600, 600, 0)], [], [(0, 1, 2, 3)])
_fl = bpy.data.objects.new('Floor', _fm)
COL_ST.objects.link(_fl)
_m = bpy.data.materials.new('M_Studio_Floor')
_m.use_nodes = True
_m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.07, 0.07, 0.07, 1)
_m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.6
_fm.materials.append(_m)
_fl.hide_select = True
_sun = bpy.data.lights.new('Sun', 'SUN')
_sun.energy, _sun.angle = 1.6, radians(14)
_so = bpy.data.objects.new('Sun', _sun)
COL_ST.objects.link(_so)
_so.rotation_euler = (radians(52), 0, radians(-38))
for _n, _loc, _size, _pw in (('Fill', (40, 50, 30), 40, 30000), ('Rim', (-45, -30, 40), 36, 60000)):
    _l = bpy.data.lights.new(_n, 'AREA')
    _l.size, _l.energy = _size, _pw
    _o = bpy.data.objects.new(_n, _l)
    COL_ST.objects.link(_o)
    _o.location = _loc
    _o.rotation_euler = (Vector((0, 0, 16)) - Vector(_loc)).to_track_quat('-Z', 'Y').to_euler()
_bm = bmesh.new()                                          # scale figure, 1.75 m
for _sg in (1, -1):
    tube(_bm, [V(0, 0.10 * _sg, 0.0), V(0, 0.10 * _sg, 0.88)], [0.07, 0.09], 10)
    tube(_bm, [V(0, 0.24 * _sg, 1.42), V(0, 0.27 * _sg, 0.80)], [0.055, 0.045], 8)
tube(_bm, [V(0, 0, 0.86), V(0, 0, 1.18), V(0, 0, 1.46)], [0.15, 0.14, 0.19], 12)
tube(_bm, [V(0, 0, 1.46), V(0, 0, 1.54)], 0.06, 8)
bmesh.ops.create_uvsphere(_bm, u_segments=12, v_segments=8, radius=0.11, matrix=T(0, 0, 1.64))
_hm = bpy.data.meshes.new('Scale_Human')
_bm.to_mesh(_hm)
_bm.free()
_ho = bpy.data.objects.new('Scale_Human', _hm)
COL_ST.objects.link(_ho)
_ho.location = (7.0, 4.4, 0.0)
_m2 = bpy.data.materials.new('M_Studio_Human')
_m2.use_nodes = True
_m2.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.75, 0.25, 0.08, 1)
_hm.materials.append(_m2)
for _p in _hm.polygons:
    _p.use_smooth = True
_cd = bpy.data.cameras.new('Camera')
_cd.lens = 50
_cd.clip_start, _cd.clip_end = 0.5, 2000
_cam = bpy.data.objects.new('Camera', _cd)
SCN.collection.objects.link(_cam)
_cam.location = (62, 46, 12)
_cam.rotation_euler = (Vector((0, 0, 15)) - _cam.location).to_track_quat('-Z', 'Y').to_euler()
SCN.camera = _cam
SCN.render.engine = 'CYCLES'
SCN.cycles.samples = 96
SCN.cycles.use_denoising = True
try:
    SCN.cycles.device = 'GPU'
    SCN.view_settings.look = 'AgX - Medium High Contrast'
except Exception as _e:
    print('studio opts', _e)

# ----------------------------------------------------------------------------------------------
# info + save
# ----------------------------------------------------------------------------------------------
bb_min = Vector((1e9, 1e9, 1e9))
bb_max = -bb_min
for ob in ALL_OBS:
    for vtx in ob.data.vertices:
        w = ob.matrix_world @ vtx.co
        for i in range(3):
            bb_min[i] = min(bb_min[i], w[i])
            bb_max[i] = max(bb_max[i], w[i])
info = dict(
    tris=tris_total, size=[round(bb_max[i] - bb_min[i], 2) for i in range(3)],
    bbox=[[round(v, 2) for v in bb_min], [round(v, 2) for v in bb_max]], atlas=ATLAS_OK, baked=BAKED, bake=BAKE_INFO,
    lengths=dict(thigh=L1, shin=L2, tarsus=L3, upperarm=LU, forearm=LF, palm=PALM, fingers=FING),
    actuators={k: dict(top_bone=v['top'], bot_bone=v['bot'], top_anchor=v['tp'], bot_anchor=v['bp'], sampled_stroke=[round(v['lo'], 3), round(v['hi'], 3)],
                       min_lever=round(v['lever'], 3), cylinder=round(v['Lc'], 3), rod=round(v['Lr'], 3), telescopes=v['ok']) for k, v in ACT.items()},
    actions=RANGES, warnings=sorted(WARN), walk=dict(frames=WALK_N, stride=STRIDE, swing=SWING, lift=LIFT, phase=PHASE),
    bones={n: dict(parent=PARENT[n], head=[round(v, 3) for v in REST[n].to_translation()],
                   x_axis=[round(v, 3) for v in REST[n].to_3x3().col[0]], y_axis=[round(v, 3) for v in REST[n].to_3x3().col[1]],
                   z_axis=[round(v, 3) for v in REST[n].to_3x3().col[2]]) for n in BONE_ORDER},
    parts=info_parts)
with open(os.path.join(HERE, NAME + '_info.json'), 'w') as fh:
    json.dump(info, fh, indent=1)
print('PARTS', len(PARTS), 'BONES', len(BONE_ORDER), 'TRIS', tris_total, 'SIZE', info['size'], 'BBOX', info['bbox'])
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, NAME + '.blend'))
try:
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile()
except Exception as _e:
    print('relpaths', _e)
print('SAVED', round(time.time() - T0, 1), 's')

# ----------------------------------------------------------------------------------------------
# export: GLB (metres) and FBX (centimetres, no armature scale), both with the three actions
# ----------------------------------------------------------------------------------------------
_low = {b.lower() for b in BONE_ORDER}
assert not [n for n in PARTS if n.lower() in _low], 'a mesh shares a bone name (UE would rename the bone)'
if '--no-export' not in ARGS:
    ad = ARM.animation_data
    ad.action = None
    for pb in PB:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
    ARM['sensor_light'] = 1.0
    MATS[EMIT].node_tree.animation_data_clear()        # exporters skip a driven value: write the light level in
    MATS[EMIT].node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 6.0
    for an, act in ACTS.items():                       # one NLA track per action: each exports as its own clip
        tr = ad.nla_tracks.new()
        tr.name = an
        stp = tr.strips.new(an, 1, act)
        try:
            stp.action_slot = act.slots[0]
        except Exception as e:
            print('strip slot', e)
    bpy.context.view_layer.update()

    def select(obs, active):
        bpy.ops.object.select_all(action='DESELECT')
        for o in obs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = active
    select([ARM] + ALL_OBS, ARM)
    bpy.ops.export_scene.gltf(filepath=os.path.join(HERE, NAME + '.glb'), export_format='GLB', use_selection=True,
                              export_apply=False, export_skins=True, export_yup=True, export_animations=True,
                              export_animation_mode='NLA_TRACKS', export_force_sampling=True, export_frame_step=2,
                              export_image_format='JPEG', export_jpeg_quality=88, export_extras=False)
    print('GLB', round(time.time() - T0, 1), 's')
    CM = Matrix.Scale(100.0, 4)
    for ob in ALL_OBS:
        mw = ob.matrix_world.copy()
        ob.parent = None
        ob.matrix_world = mw
    for ob in [ARM] + ALL_OBS:
        ob.matrix_world = CM @ ob.matrix_world
    bpy.context.view_layer.update()
    select([ARM] + ALL_OBS, ARM)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for ob in ALL_OBS:
        mw = ob.matrix_world.copy()
        ob.parent = ARM
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_world = mw
    for act in ACTS.values():                          # bone translations are in armature units: metres -> centimetres
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    for fc in cb.fcurves:
                        if fc.data_path.endswith('location'):
                            for kp in fc.keyframe_points:
                                kp.co.y *= 100.0
                                kp.handle_left.y *= 100.0
                                kp.handle_right.y *= 100.0
                            fc.update()
    SCN.unit_settings.scale_length = 0.01
    bpy.context.view_layer.update()
    select([ARM] + ALL_OBS, ARM)
    bpy.ops.export_scene.fbx(
        filepath=os.path.join(HERE, NAME + '.fbx'), use_selection=True, object_types={'ARMATURE', 'MESH'},
        apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE', global_scale=1.0, axis_forward='-Z', axis_up='Y',
        use_mesh_modifiers=True, mesh_smooth_type='FACE', use_tspace=False, add_leaf_bones=False, primary_bone_axis='Y',
        secondary_bone_axis='X', use_armature_deform_only=False, bake_anim=True, bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=True, bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True, bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0, use_custom_props=False, path_mode='RELATIVE', embed_textures=False)
    print('EXPORTED', round(time.time() - T0, 1), 's')
