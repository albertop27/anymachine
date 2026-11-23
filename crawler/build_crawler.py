"""Rig Crawler: ~40 m six-legged war machine made from a converted offshore oil rig. Hard-surface build, rig, bake, export.
Run:  blender.exe -b --factory-startup -P build_crawler.py -- [--wear 0..1] [--no-bake] [--out DIR] [--name NAME] [--tex 4096]
Everything is rebuilt from scratch on every run. Metres, Z up, front = -Y (same as Stilt/TRELLIS), origin on the ground
under the centre of the machine. The machine's right-hand side is -X (side letter R), left is +X (L).
Built from one painting (runs/2026-11-23_002_crawler-turnaround): see NOTES.md.
"""
import bpy, bmesh, sys, os, json, time, random
import numpy as np
from math import sin, cos, pi, radians, sqrt, atan2, hypot
from mathutils import Vector, Matrix

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def arg(name, default, cast=str):
    return cast(ARGS[ARGS.index(name) + 1]) if name in ARGS else default


RUN_OUT = 'C:/Users/malbe/Art design/runs/2026-11-23_003_crawler-builds/outputs/anymachine'
OUT_DIR = arg('--out', RUN_OUT)
NAME = arg('--name', 'crawler')
WEAR = arg('--wear', 0.5, float)
TEX_N = arg('--tex', 4096, int)
DO_BAKE = '--no-bake' not in ARGS
DO_EXPORT = '--no-export' not in ARGS
os.makedirs(OUT_DIR, exist_ok=True)
TEX_DIR = os.path.join(OUT_DIR, NAME + '_textures')
T0 = time.time()
RNG = random.Random(11)


def V(*a):
    return Vector(a)


bpy.ops.wm.read_factory_settings(use_empty=True)
SCN = bpy.context.scene
SCN.unit_settings.system = 'METRIC'
SCN.unit_settings.scale_length = 1.0
COL = bpy.data.collections.new('Crawler')
SCN.collection.children.link(COL)

# ----------------------------------------------------------------------------------------------
# materials: key -> Blender material. ST/DK/GN/RS/CB/RB are painted-metal families that get baked into one atlas;
# GL/WE/LE/CR (glass, window light, eye lamps, crew) stay as plain materials on their own small objects.
# ----------------------------------------------------------------------------------------------
ST, DK, GN, RS, CB, RB, GL, WE, LE, CR = 'steel', 'deck', 'gun', 'rust', 'cable', 'rubber', 'glass', 'win', 'lamp', 'crew'
BAKED = (ST, DK, GN, RS, CB, RB)
MATS = {}
BEV = {ST: 0.07, DK: 0.05, GN: 0.06, RS: 0.04, CB: 0.0, RB: 0.06, GL: 0.0, WE: 0.0, LE: 0.0, CR: 0.015}


def L_(nt):
    return nt.links.new


def paint(key, base, metallic, rough, tone=0.10, rivets=True, seams=True):
    """Painted-metal base: low-frequency tone variation, roughness noise, and (triplanar on rest_pos) panel seams plus a rivet grid
    as bump, so the normal bake carries 'riveted plates' without geometry."""
    m = bpy.data.materials.new('M_Crawler_' + key)
    m.use_nodes = True
    nt = m.node_tree
    L = nt.links.new
    b = nt.nodes['Principled BSDF']
    b.inputs['Metallic'].default_value = metallic
    tc = nt.nodes.new('ShaderNodeAttribute')
    tc.attribute_name = 'rest_pos'
    N = nt.nodes.new

    def noise(scale, detail):
        n = N('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        L(tc.outputs['Vector'], n.inputs['Vector'])
        return n.outputs['Fac']

    def rng(sock, a, z, lo=0.0, hi=1.0):
        r = N('ShaderNodeMapRange')
        r.inputs['From Min'].default_value = lo
        r.inputs['From Max'].default_value = hi
        r.inputs['To Min'].default_value = a
        r.inputs['To Max'].default_value = z
        L(sock, r.inputs['Value'])
        return r.outputs['Result']

    def math(op, a, c=None):
        x = N('ShaderNodeMath')
        x.operation = op
        for i, v in enumerate((a, c)):
            if v is None:
                continue
            if isinstance(v, float) or isinstance(v, int):
                x.inputs[i].default_value = v
            else:
                L(v, x.inputs[i])
        return x.outputs[0]

    t = rng(noise(0.30, 5.0), 1.0 - tone, 1.0 + tone, 0.3, 0.7)
    mix = N('ShaderNodeMixRGB')
    mix.blend_type = 'MULTIPLY'
    mix.inputs['Fac'].default_value = 1.0
    mix.inputs['Color1'].default_value = (*base, 1)
    L(t, mix.inputs['Color2'])          # grey scalar used as colour
    L(mix.outputs['Color'], b.inputs['Base Color'])
    b.inputs['Base Color'].default_value = (*base, 1)
    L(rng(noise(2.0, 6.0), rough - 0.10, rough + 0.10, 0.3, 0.7), b.inputs['Roughness'])
    # bump
    fine = noise(40.0, 3.0)
    h = math('MULTIPLY', fine, 0.15)
    if seams or rivets:
        geo = N('ShaderNodeNewGeometry')
        sn = N('ShaderNodeSeparateXYZ')
        L(geo.outputs['Normal'], sn.inputs[0])
        wts = []
        for i in range(3):
            a = math('ABSOLUTE', sn.outputs[i])
            wts.append(math('POWER', a, 4.0))
        tot = math('ADD', math('ADD', wts[0], wts[1]), wts[2])
        sp = N('ShaderNodeSeparateXYZ')
        L(tc.outputs['Vector'], sp.inputs[0])
        acc = None
        for i, (ia, ib) in enumerate(((1, 2), (0, 2), (0, 1))):     # plane perpendicular to axis i
            cv = N('ShaderNodeCombineXYZ')
            L(sp.outputs[ia], cv.inputs['X'])
            L(sp.outputs[ib], cv.inputs['Y'])
            hh = None
            if seams:
                br = N('ShaderNodeTexBrick')
                br.inputs['Scale'].default_value = 1.0
                br.inputs['Mortar Size'].default_value = 0.012
                br.inputs['Mortar Smooth'].default_value = 0.01
                br.inputs['Brick Width'].default_value = 2.4
                br.inputs['Row Height'].default_value = 1.2
                br.inputs['Bias'].default_value = 0.0
                L(cv.outputs['Vector'], br.inputs['Vector'])
                hh = math('MULTIPLY', br.outputs['Fac'], 0.5)
            if rivets:
                vm = N('ShaderNodeVectorMath')
                vm.operation = 'SCALE'
                vm.inputs[3].default_value = 1.7
                L(cv.outputs['Vector'], vm.inputs[0])
                fr = N('ShaderNodeVectorMath')
                fr.operation = 'FRACTION'
                L(vm.outputs[0], fr.inputs[0])
                sb = N('ShaderNodeVectorMath')
                sb.operation = 'SUBTRACT'
                sb.inputs[1].default_value = (0.5, 0.5, 0.5)
                L(fr.outputs[0], sb.inputs[0])
                mm = N('ShaderNodeVectorMath')
                mm.operation = 'MULTIPLY'
                mm.inputs[1].default_value = (1.0, 1.0, 0.0)
                L(sb.outputs[0], mm.inputs[0])
                ln = N('ShaderNodeVectorMath')
                ln.operation = 'LENGTH'
                L(mm.outputs[0], ln.inputs[0])
                dot = rng(ln.outputs['Value'], 0.5, 0.0, 0.07, 0.13)
                hh = dot if hh is None else math('ADD', hh, dot)
            term = math('MULTIPLY', hh, math('DIVIDE', wts[i], math('MAXIMUM', tot, 1e-4)))
            acc = term if acc is None else math('ADD', acc, term)
        h = math('ADD', h, acc)
    bp = N('ShaderNodeBump')
    bp.inputs['Strength'].default_value = 0.6
    bp.inputs['Distance'].default_value = 0.02
    L(h, bp.inputs['Height'])
    L(bp.outputs['Normal'], b.inputs['Normal'])
    m.diffuse_color = (*base, 1)
    MATS[key] = m
    return m


def plain(key, base, rough=0.5, metallic=0.0, emit=None, strength=0.0, alpha=1.0):
    m = bpy.data.materials.new('M_Crawler_' + key)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*base, 1)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metallic
    if emit:
        b.inputs['Emission Color'].default_value = (*emit, 1)
        b.inputs['Emission Strength'].default_value = strength
    if alpha < 1.0:
        b.inputs['Alpha'].default_value = alpha
        try:
            m.surface_render_method = 'BLENDED'
        except Exception:
            pass
    m.diffuse_color = (*base, 1)
    MATS[key] = m
    return m


paint(ST, (0.150, 0.162, 0.178), 0.0, 0.55, tone=0.12)                 # mid-grey painted steel (blue-ish grey)
paint(DK, (0.055, 0.058, 0.064), 0.0, 0.65, tone=0.15)                 # dark deck / underside
paint(GN, (0.040, 0.044, 0.052), 0.55, 0.42, tone=0.10)                # gunmetal (rifle)
paint(RS, (0.260, 0.100, 0.040), 0.0, 0.62, tone=0.18, rivets=False)   # orange-brown painted lattice / cranes
paint(CB, (0.012, 0.012, 0.013), 0.0, 0.62, tone=0.05, rivets=False, seams=False)
paint(RB, (0.022, 0.021, 0.020), 0.0, 0.88, tone=0.08, rivets=False, seams=False)
plain(GL, (0.55, 0.75, 0.85), 0.05, 0.0, alpha=0.22)
plain(WE, (0.9, 0.55, 0.20), 0.4, 0.0, emit=(1.0, 0.62, 0.25), strength=7.0)
plain(LE, (0.9, 0.8, 0.5), 0.3, 0.0, emit=(1.0, 0.85, 0.5), strength=25.0)
plain(CR, (0.10, 0.13, 0.07), 0.8)


def industrial_wear(m, amount, edge_col):
    """Adapted from stilt/build_stilt.py (albertop27 fork): rust patches, vertical rain/oil streaks, crevice dirt (pointiness),
    worn edges (paint chipped to rust or bare metal), rougher overall. All inputs are rest_pos or geometry, so it bakes."""
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    L = nt.links.new
    col_in = b.inputs['Base Color']
    src = col_in.links[0].from_socket if col_in.is_linked else None
    base = tuple(col_in.default_value)
    tc = nt.nodes.new('ShaderNodeAttribute')
    tc.attribute_name = 'rest_pos'
    geo = nt.nodes.new('ShaderNodeNewGeometry')

    def noise(scale, detail, vec):
        n = nt.nodes.new('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        L(vec, n.inputs['Vector'])
        return n

    def ramp(sock, lo, hi, a=0.0, z=1.0):
        r = nt.nodes.new('ShaderNodeMapRange')
        r.inputs['From Min'].default_value = lo
        r.inputs['From Max'].default_value = hi
        r.inputs['To Min'].default_value = a
        r.inputs['To Max'].default_value = z
        L(sock, r.inputs['Value'])
        return r.outputs['Result']

    def mix(fac, c1, c2, blend='MIX'):
        x = nt.nodes.new('ShaderNodeMixRGB')
        x.blend_type = blend
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
        x = nt.nodes.new('ShaderNodeMath')
        x.operation = 'MULTIPLY'
        L(a, x.inputs[0])
        x.inputs[1].default_value = k
        return x.outputs[0]

    # rust patches: blotchy, 1-3 m on a 40 m machine
    rust_m = ramp(noise(0.55, 8.0, tc.outputs['Vector']).outputs['Fac'], 0.60 - 0.13 * amount, 0.71, 0, 1)
    rust_c = mix(noise(5.0, 4.0, tc.outputs['Vector']).outputs['Fac'], (0.17, 0.060, 0.022, 1), (0.32, 0.13, 0.045, 1))
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (8.0, 8.0, 0.22)
    L(tc.outputs['Vector'], mp.inputs['Vector'])
    streak = ramp(noise(1.0, 6.0, mp.outputs['Vector']).outputs['Fac'], 0.52, 0.75, 0, 0.7)
    dirt = ramp(geo.outputs['Pointiness'], 0.47, 0.5, 1, 0)
    edge = ramp(geo.outputs['Pointiness'], 0.53, 0.58, 0, 1)
    c = src if src is not None else base
    c = mix(mul(streak, amount), c, (0.05, 0.04, 0.03, 1), 'MULTIPLY')
    c = mix(mul(dirt, 0.85 * amount), c, (0.035, 0.03, 0.025, 1))
    c = mix(mul(rust_m, amount), c, rust_c)
    c = mix(mul(edge, 0.7 * amount), c, edge_col)
    L(c, col_in)
    r_in = b.inputs['Roughness']
    r_src = r_in.links[0].from_socket if r_in.is_linked else None
    add = nt.nodes.new('ShaderNodeMath')
    add.operation = 'ADD'
    add.use_clamp = True
    if r_src is not None:
        L(r_src, add.inputs[0])
    else:
        add.inputs[0].default_value = r_in.default_value
    add.inputs[1].default_value = 0.16 * amount
    rr = mix(rust_m, add.outputs[0], (0.9, 0.9, 0.9, 1))
    bw = nt.nodes.new('ShaderNodeRGBToBW')
    L(rr, bw.inputs['Color'])
    L(bw.outputs['Val'], r_in)
    if b.inputs['Metallic'].default_value > 0.1:
        met = nt.nodes.new('ShaderNodeMath')
        met.operation = 'MULTIPLY'
        inv = ramp(rust_m, 0, 1, 1, 0)
        L(inv, met.inputs[0])
        met.inputs[1].default_value = b.inputs['Metallic'].default_value
        L(met.outputs[0], b.inputs['Metallic'])


if WEAR > 0:
    for k_, ec in ((ST, (0.30, 0.125, 0.05, 1)), (DK, (0.22, 0.10, 0.04, 1)), (GN, (0.40, 0.39, 0.37, 1)),
                   (RS, (0.34, 0.14, 0.05, 1)), (CB, (0.10, 0.10, 0.10, 1)), (RB, (0.10, 0.09, 0.08, 1))):
        industrial_wear(MATS[k_], WEAR, ec)
print('wear', WEAR, flush=True)

# ----------------------------------------------------------------------------------------------
# geometry helpers (all in world metres, rest pose)
# ----------------------------------------------------------------------------------------------


class Pal(dict):
    """material key -> bmesh. A key ending in '!' is not bevelled."""
    def __missing__(self, k):
        self[k] = bmesh.new()
        return self[k]


def face(bm, vs):
    try:
        return bm.faces.new(vs)
    except ValueError:
        return None


def loft(bm, rings, cap0=True, cap1=True):
    vr = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(vr[0])
    for a, b in zip(vr, vr[1:]):
        for i in range(n):
            j = (i + 1) % n
            face(bm, [a[i], a[j], b[j], b[i]])
    if cap0:
        face(bm, list(reversed(vr[0])))
    if cap1:
        face(bm, vr[-1])


def frame(d, up=None):
    """(side, up, dir) orthonormal frame for direction d."""
    d = Vector(d).normalized()
    up = Vector(up) if up is not None else V(0, 0, 1)
    if abs(d.dot(up.normalized())) > 0.95:
        up = V(1, 0, 0) if abs(d.x) < 0.9 else V(0, 1, 0)
    s = up.cross(d).normalized()
    u = d.cross(s).normalized()
    return s, u, d


def Rm(x, y, z):
    return Matrix((x, y, z)).transposed()


def box(bm, c, sz, R=None):
    R = R if R is not None else Matrix.Identity(3)
    c = Vector(c)
    h = [x / 2 for x in sz]
    vs = [bm.verts.new(c + R @ V(sx * h[0], sy * h[1], sz_ * h[2])) for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]
    for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        face(bm, [vs[i] for i in q])


def bar(bm, a, b, w, h, up=None):
    """box from a to b, cross-section w (along side) x h (along up-ish)."""
    a, b = Vector(a), Vector(b)
    s, u, d = frame(b - a, up)
    box(bm, (a + b) / 2, (w, h, (b - a).length), Rm(s, u, d))


def cyl(bm, a, b, r, n=16, r2=None, caps=True, up=None, a0=0.0):
    a, b = Vector(a), Vector(b)
    s, u, d = frame(b - a, up)
    r2 = r if r2 is None else r2
    ang = [a0 + 2 * pi * i / n for i in range(n)]
    loft(bm, [[p + (s * cos(t) + u * sin(t)) * rr for t in ang] for p, rr in ((a, r), (b, r2))], caps, caps)


def strut(bm, a, b, t):
    cyl(bm, a, b, t * 0.7071, 4, caps=False, a0=pi / 4)


def seg(bm, a, b, w0, h0, w1, h1, up, ch=0.25, mid=None):
    """armoured limb segment: chamfered-rectangle cross-section, tapered. w along side, h along up'."""
    a, b = Vector(a), Vector(b)
    s, u, d = frame(b - a, up)

    def ring(p, w, h):
        c = min(ch, w * 0.45, h * 0.45)
        pts = [(-w + c, -h), (w - c, -h), (w, -h + c), (w, h - c), (w - c, h), (-w + c, h), (-w, h - c), (-w, -h + c)]
        return [p + s * x + u * y for x, y in pts]
    rings = [ring(a, w0, h0)]
    if mid:
        rings.append(ring((a + b) / 2, *mid))
    rings.append(ring(b, w1, h1))
    loft(bm, rings)


def jag_plate(bm, c, uvec, vvec, w, h, th, jag=0.22, chop=True):
    """thin armour plate with broken/jagged outline in the plane (uvec, vvec) centred at c."""
    uvec, vvec = Vector(uvec).normalized(), Vector(vvec).normalized()
    n = uvec.cross(vvec).normalized()
    base = [(-w, -h), (0, -h), (w, -h), (w, 0), (w, h), (0, h), (-w, h), (-w, 0)]
    cut = RNG.randrange(0, 8, 2) if chop else -1
    pts = []
    for i, (x, y) in enumerate(base):
        k = RNG.uniform(0.0, jag) if i % 2 else RNG.uniform(0.0, jag * 0.5)
        if i == cut:
            k = RNG.uniform(0.35, 0.55)
        pts.append((x * (1 - k), y * (1 - k)))
    c = Vector(c)
    loft(bm, [[c + uvec * x + vvec * y + n * z for x, y in pts] for z in (-th / 2, th / 2)])


def smooth_path(pts, sub=6):
    P = [Vector(p) for p in pts]
    ext = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(sub):
            t = k / sub
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1])
    return out


def tube(bm, pts, r, segs=8, sub=6, caps=True):
    path = smooth_path(pts, sub) if sub > 1 else [Vector(p) for p in pts]
    rings = []
    nrm = None
    for i, p in enumerate(path):
        t = (path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]).normalized()
        if nrm is None:
            nrm = frame(t)[0]
        nrm = (nrm - t * nrm.dot(t)).normalized()
        bn = t.cross(nrm)
        rr = r if not isinstance(r, (tuple, list)) else r[0] + (r[1] - r[0]) * i / (len(path) - 1)
        rings.append([p + (nrm * cos(2 * pi * k / segs) + bn * sin(2 * pi * k / segs)) * rr for k in range(segs)])
    loft(bm, rings, caps, caps)


def railing(bm, pts, h=1.1, sp=1.6, t=0.06):
    for p, q in zip(pts, pts[1:]):
        p, q = Vector(p), Vector(q)
        ln = (q - p).length
        n = max(1, int(round(ln / sp)))
        for i in range(n + 1):
            x = p + (q - p) * (i / n)
            strut(bm, x, x + V(0, 0, h), t)
        for z in (h, h * 0.5):
            strut(bm, p + V(0, 0, z), q + V(0, 0, z), t * 0.8)


def lattice(bm, a, b, w0, w1, nlev, t, up=None, tr=1.0):
    """open steel lattice mast: 4 posts, rings and X-bracing, from a to b, half-width w0 -> w1 (real thin struts)."""
    a, b = Vector(a), Vector(b)
    s, u, d = frame(b - a, up)
    ln = (b - a).length
    cs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

    def cn(k, i):
        w = w0 + (w1 - w0) * k / nlev
        return a + d * (ln * k / nlev) + s * w * cs[i][0] + u * w * cs[i][1]
    for k in range(nlev):
        for i in range(4):
            j = (i + 1) % 4
            strut(bm, cn(k, i), cn(k + 1, i), t * 1.6 * tr)
            strut(bm, cn(k, i), cn(k, j), t * tr)
            strut(bm, cn(k, i), cn(k + 1, j), t * 0.8 * tr)
            strut(bm, cn(k, j), cn(k + 1, i), t * 0.8 * tr)
    for i in range(4):
        strut(bm, cn(nlev, i), cn(nlev, (i + 1) % 4), t * tr)


def mannequin(bm, x, y, z0):
    """stylised 1.8 m figure standing on z0, facing -Y."""
    for sg in (-1, 1):
        box(bm, (x + sg * 0.11, y, z0 + 0.44), (0.17, 0.24, 0.88))
        box(bm, (x + sg * 0.27, y, z0 + 1.18), (0.10, 0.12, 0.60))
    box(bm, (x, y, z0 + 1.19), (0.40, 0.24, 0.62))
    box(bm, (x, y, z0 + 1.55), (0.12, 0.12, 0.10))
    bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=7, radius=0.115, matrix=Matrix.Translation((x, y, z0 + 1.68)))


# ----------------------------------------------------------------------------------------------
# part registry
# ----------------------------------------------------------------------------------------------
OBJS = []          # (object, bone)
PARTS = {}         # logical part name -> bone
CANON = []         # objects whose materials are baked
BONES = {}         # name -> (head, tail, parent)


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


def piece(name, bm, key, bevel):
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
    me.materials.append(MATS[key])
    if bevel > 0:
        md = ob.modifiers.new('bev', 'BEVEL')
        md.width = bevel
        md.segments = 1
        md.limit_method = 'ANGLE'
        md.angle_limit = ang
        md.use_clamp_overlap = True
    finish(ob, name)
    return ob


def add_rest(ob):
    me = ob.data
    at = me.attributes.new('rest_pos', 'FLOAT_VECTOR', 'POINT')
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get('co', co)
    at.data.foreach_set('vector', co)


def part(name, bone, P):
    """P: Pal. Baked materials are joined into object `name`; glass/emissive/crew go on `name_<key>` objects."""
    assert bone in BONES, bone
    baked, extra = [], []
    for key, bm in P.items():
        if not len(bm.faces):
            continue
        mk = key.rstrip('!')
        ob = piece(name if mk in BAKED else name + '_' + mk, bm, mk, 0.0 if key.endswith('!') else BEV[mk])
        (baked if mk in BAKED else extra).append((ob, mk))
    if baked:
        ob = baked[0][0]
        if len(baked) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for o, _ in baked:
                o.select_set(True)
            bpy.context.view_layer.objects.active = ob
            bpy.ops.object.join()
        ob.name = name
        ob.data.name = name
        add_rest(ob)
        OBJS.append((ob, bone))
        CANON.append(ob)
    for ob, mk in extra:
        OBJS.append((ob, bone))
    PARTS[name] = bone


def bone(name, head, tail, parent):
    BONES[name] = (Vector(head), Vector(tail), parent)


# ----------------------------------------------------------------------------------------------
# bones (rest pose = the standing stance; feet at z = 0)
# ----------------------------------------------------------------------------------------------
bone('root', (0, 0, 0), (0, 0, 2), None)
bone('body', (0, 2, 10), (0, 2, 13), 'root')
RIFLE_PIV = V(0, -8.6, 12.2)
bone('rifle_yaw', RIFLE_PIV, RIFLE_PIV + V(0, -2, 0), 'body')
bone('rifle_pitch', RIFLE_PIV, RIFLE_PIV + V(0, -6, 0), 'rifle_yaw')

# leg table: index, hip y, knee y, foot y, foot x (absolute)
LEGS = [(1, -5.0, -6.6, -9.4, 14.0), (2, 3.0, 3.2, 3.2, 15.6), (3, 10.0, 11.4, 13.2, 14.0)]
HIP_X, KNEE_X, HIP_Z, KNEE_Z, ANK_Z = 10.7, 13.2, 12.4, 17.2, 3.7
SIDES = (('L', 1), ('R', -1))
LEGGEO = {}
for sn, sg in SIDES:
    for i, hy, ky, fy, fx in LEGS:
        hip = V(sg * HIP_X, hy, HIP_Z)
        knee = V(sg * KNEE_X, ky, KNEE_Z)
        ank = V(sg * fx, fy, ANK_Z)
        pad_top = V(sg * fx, fy, 0.95)
        k = '%s%d' % (sn, i)
        LEGGEO[k] = (sg, hip, knee, ank, pad_top)
        bone(k + '_upper', hip, knee, 'body')
        bone(k + '_lower', knee, ank, k + '_upper')
        bone(k + '_foot', ank, pad_top, k + '_lower')
        bone('IK_' + k, ank, pad_top, 'root')       # same orientation as the foot bone
        ax = (ank - hip).normalized()                       # pole sits in the hip-knee-ankle plane, pushed out from the axis
        perp = ((knee - hip) - ax * (knee - hip).dot(ax)).normalized()
        bone('POLE_' + k, knee + perp * 6, knee + perp * 7, 'root')


# ----------------------------------------------------------------------------------------------
# hull, decks, bridge, derrick, cranes
# ----------------------------------------------------------------------------------------------
def build_hull():
    P = Pal()
    # lower hull block, z 9..14, y -8..14; keel frames below
    box(P[ST], (0, 3, 11.5), (19, 22, 5))
    box(P[DK], (0, 3, 8.8), (15, 19, 0.6))
    for x in (-6, -2, 2, 6):
        box(P[DK], (x, 3, 8.55), (0.9, 20, 0.5))
    for y in (-6, -1, 4, 9, 13):
        box(P[DK], (0, y, 8.6), (17, 0.8, 0.45))
    part('hull_lower', 'body', P)

    P = Pal()   # mid block with sloped front glacis, z 14..18
    box(P[ST], (0, 5.5, 16.0), (15, 15, 4))
    for sg in (-1, 1):
        # glacis wedge on the front of the mid block
        loft(P[ST], [[V(sg * 7.5, -2.0, 14.0), V(sg * 0.0, -2.0, 14.0), V(0.0, -2.0, 18.0), V(sg * 7.5, -2.0, 18.0)],
                     [V(sg * 7.5, -4.2, 14.0), V(sg * 0.0, -4.2, 14.0), V(0.0, -2.6, 18.0), V(sg * 7.5, -2.6, 18.0)]])
    part('hull_mid', 'body', P)

    P = Pal()   # front armour plates + side armour
    for x in (-8.6, -6.4, -4.4, 4.4, 6.4, 8.6):
        for z, hh in ((10.4, 1.2), (12.8, 1.1)):
            jag_plate(P[ST], (x, -8.2, z), (1, 0, 0), (0, 0, 1), 1.0, hh, 0.35)
    for x in (-6.5, -3.0, 3.0, 6.5):
        jag_plate(P[ST], (x, -3.0, 16.0), (1, 0, 0), (0, -0.2, 1), 1.5, 1.7, 0.3)
    for sg in (-1, 1):
        for y in (-5.5, 0.5, 6.5, 12.0):
            jag_plate(P[ST], (sg * 9.75, y, 10.6), (0, 1, 0), (0, 0, 1), 2.4, 1.3, 0.3)
        for y in (-1.0, 5.5, 11.5):
            jag_plate(P[ST], (sg * 9.75, y, 13.0), (0, 1, 0), (0, 0, 1), 2.4, 0.8, 0.3)
    part('hull_armour', 'body', P)

    P = Pal()   # side catwalks on the ledge (z 14) with rails and stairs
    for sg in (-1, 1):
        box(P[DK], (sg * 8.5, 5.5, 13.95), (2.0, 15.5, 0.12))
        railing(P[RS + '!'], [(sg * 9.45, -2.0, 14.0), (sg * 9.45, 13.0, 14.0)], 1.1, 1.8)
        railing(P[RS + '!'], [(sg * 7.55, -2.0, 14.0), (sg * 7.55, 13.0, 14.0)], 1.1, 1.8)
    box(P[DK], (0, -5.0, 13.95), (19, 6.0, 0.12))                      # front shelf
    railing(P[RS + '!'], [(-9.45, -8.0, 14.0), (9.45, -8.0, 14.0)], 1.1, 1.8)
    railing(P[RS + '!'], [(-9.45, -8.0, 14.0), (-9.45, -2.0, 14.0)], 1.1, 1.8)
    railing(P[RS + '!'], [(9.45, -8.0, 14.0), (9.45, -2.0, 14.0)], 1.1, 1.8)
    part('catwalks_lower', 'body', P)

    P = Pal()   # stairs: ledge z14 -> roof z18 along each side
    for sg in (-1, 1):
        y0, y1 = 9.0, 3.0
        n = 12
        for k in range(n):
            y = y0 + (y1 - y0) * k / (n - 1)
            z = 14.0 + 4.0 * k / (n - 1)
            box(P[DK], (sg * 7.0, y, z + 0.05), (1.0, 0.55, 0.06))
        for off in (-0.55, 0.55):
            strut(P[RS + '!'], (sg * (7.0 + off), y0, 14.0), (sg * (7.0 + off), y1, 18.0), 0.1)
            strut(P[RS + '!'], (sg * (7.0 + off), y0, 15.0), (sg * (7.0 + off), y1, 19.0), 0.05)
    part('stairs', 'body', P)

    P = Pal()   # roof deck level 18: derrick substructure, rear accommodation blocks
    box(P[DK], (0, 5.5, 18.05), (15, 15, 0.12))
    box(P[ST], (0, 6.0, 19.0), (7.0, 6.5, 2.0))           # derrick substructure
    box(P[DK], (0, 11.5, 19.5), (17, 4.0, 3.0))           # block A  y 9.5..13.5 z 18..21
    box(P[ST], (5.5, 11.8, 22.5), (6.5, 3.4, 3.0))        # block B  z 21..24
    box(P[ST], (-3.5, 11.0, 21.9), (2.4, 2.4, 1.8))       # vent house
    for x in (-7, -5.2, -3.4, -1.6, 0.2, 1.8):            # windows band on block A (back face)
        box(P[DK], (x, 13.55, 19.7), (1.0, 0.1, 1.0))
    for sg in (-1, 1):
        railing(P[RS + '!'], [(sg * 7.4, -1.5, 18.0), (sg * 7.4, 9.5, 18.0)], 1.1, 1.8)
    railing(P[RS + '!'], [(-8.5, 13.5, 21.0), (8.5, 13.5, 21.0), (8.5, 9.5, 21.0)], 1.1, 1.8)
    part('roof_deck', 'body', P)

    # greebles: machinery blocks, vents, tanks on roof and hull sides
    P = Pal()
    for sg in (-1, 1):
        for y in (2.5, 8.0):
            box(P[ST], (sg * 5.8, y, 19.4), (2.2, 3.0, 2.8))
        cyl(P[ST], (sg * 5.0, 6.8, 18.0), (sg * 5.0, 6.8, 20.8), 1.0, 20)
        cyl(P[DK], (sg * 5.0, 6.8, 20.8), (sg * 5.0, 6.8, 21.1), 1.05, 20)
    for sg in (-1, 1):
        for _ in range(26):
            y = RNG.uniform(-6.5, 12.5)
            z = RNG.uniform(9.6, 13.3)
            sy, sz = RNG.uniform(0.8, 2.2), RNG.uniform(0.5, 1.6)
            box(P[RNG.choice((ST, DK, ST))], (sg * 9.7, y, z), (0.35, sy, sz))
        for k in range(8):                                  # louvre grille
            box(P[DK], (sg * 9.75, 8.2 + k * 0.18, 12.2), (0.2, 0.08, 1.2))
    for _ in range(18):                                     # mid block side/rear fittings
        x = RNG.choice((-1, 1)) * 7.65
        y = RNG.uniform(-1.5, 12.5)
        z = RNG.uniform(14.6, 17.5)
        box(P[RNG.choice((ST, DK))], (x, y, z), (0.35, RNG.uniform(0.6, 1.8), RNG.uniform(0.5, 1.2)))
    for x in (-6, -3, 0, 3, 6):                             # rear face
        box(P[DK], (x, 14.15, 11.8), (1.6, 0.3, 2.6))
    part('machinery_blocks', 'body', P)

    P = Pal()   # pipes on the roof
    for sg in (-1, 1):
        tube(P[ST], [(sg * 3.4, 4.5, 20.0), (sg * 4.2, 3.5, 20.6), (sg * 5.2, 2.5, 20.6), (sg * 5.6, 1.5, 19.4), (sg * 5.6, 1.0, 18.1)], 0.32, 12)
        tube(P[DK], [(sg * 3.4, 8.0, 19.8), (sg * 5.0, 8.4, 20.0), (sg * 6.2, 9.6, 20.0), (sg * 6.2, 9.8, 21.2)], 0.28, 12)
    tube(P[ST], [(-6.5, 3.0, 18.4), (-6.5, 6.0, 18.4), (-4.0, 6.5, 18.4)], 0.4, 12)
    part('roof_pipes', 'body', P)

    # exhaust stacks / funnels on block A roof
    P = Pal()
    for x in (-6.5, -7.8, 2.0):
        cyl(P[DK], (x, 11.5, 21.0), (x, 11.5, 24.6), 0.55, 16)
        cyl(P[GN], (x, 11.5, 24.6), (x, 11.5, 24.9), 0.65, 16)
    part('stacks', 'body', P)
    return


def build_bridge():
    P = Pal()
    bx, y0, y1, z0, z1 = 5.5, -5.0, 2.0, 18.0, 21.8
    zs, zh = 19.0, 21.0               # window sill / header
    box(P[ST], (0, (y0 + y1) / 2, 18.2), (2 * bx + 0.6, y1 - y0 + 0.6, 0.45))   # floor slab
    box(P[ST], (0, (y0 + y1) / 2, z1 - 0.2), (2 * bx + 0.8, y1 - y0 + 0.8, 0.4))   # roof
    box(P[ST], (0, y0, (z0 + zs) / 2 + 0.1), (2 * bx, 0.3, zs - z0 - 0.2))      # sill wall
    box(P[ST], (0, y0, (zh + z1) / 2 - 0.1), (2 * bx, 0.3, z1 - zh - 0.3))      # header
    npil = 8
    for k in range(npil + 1):
        x = -bx + 0.15 + (2 * bx - 0.3) * k / npil
        box(P[ST], (x, y0, (zs + zh) / 2), (0.3, 0.32, zh - zs))
    for sg in (-1, 1):                                          # side walls with small ports
        box(P[ST], (sg * bx, (y0 + y1) / 2, (z0 + z1) / 2), (0.3, y1 - y0, z1 - z0))
        for y in (-3.5, -1.5, 0.5):
            box(P[DK], (sg * (bx + 0.16), y, 20.0), (0.08, 1.0, 0.9))
    box(P[ST], (0, y1, (z0 + z1) / 2), (2 * bx, 0.3, z1 - z0))   # back wall
    for sg in (-1, 1):                                          # underside braces (the bridge overhangs the front shelf)
        for x in (sg * 1.5, sg * 4.5):
            bar(P[DK], (x, y0 + 0.4, 18.0), (x, -2.0, 14.5), 0.5, 0.4, (1, 0, 0))
    part('bridge', 'body', P)

    P = Pal()
    box(P[GL], (0, y0 - 0.02, (zs + zh) / 2), (2 * bx - 0.3, 0.05, zh - zs))
    part('bridge_glass', 'body', P)

    P = Pal()   # warm interior: back wall panel, ceiling panels, consoles (lit)
    box(P[WE], (0, y1 - 0.2, 19.9), (2 * bx - 0.8, 0.06, 1.9))
    for x in (-3.0, 0.0, 3.0):
        box(P[WE], (x, -1.5, z1 - 0.45), (1.8, 5.0, 0.06))
    for x in (-3.5, 0.0, 3.5):
        box(P[WE], (x, -4.2, 19.2), (2.2, 0.5, 0.35))     # console strips under the glass
    part('bridge_interior_light', 'body', P)

    P = Pal()   # bridge roof deck gear
    railing(P[RS + '!'], [(-bx - 0.3, y0 - 0.3, 22.0), (bx + 0.3, y0 - 0.3, 22.0), (bx + 0.3, y1 + 0.3, 22.0), (-bx - 0.3, y1 + 0.3, 22.0),
                          (-bx - 0.3, y0 - 0.3, 22.0)], 1.1, 1.8)
    cyl(P[GN], (-3.5, -0.5, 22.0), (-3.5, -0.5, 22.9), 0.6, 16)
    box(P[GN], (-3.5, -0.5, 23.2), (2.4, 0.3, 0.7))          # radar bar
    cyl(P[DK], (3.5, 0.5, 22.0), (3.5, 0.5, 26.0), 0.08, 8)
    cyl(P[DK], (4.5, 1.0, 22.0), (4.5, 1.0, 24.5), 0.06, 8)
    part('bridge_roof_gear', 'body', P)

    names = ['crew_1', 'crew_2', 'crew_3', 'crew_4']
    for nm, x in zip(names, (-3.4, -1.2, 1.4, 3.6)):
        P = Pal()
        mannequin(P[CR], x, -3.9, 18.43)
        part(nm, 'body', P)


def build_derrick():
    P = Pal()
    cx, cy, z0 = 0.0, 6.0, 20.0
    lattice(P[RS + '!'], (cx, cy, z0), (cx, cy, 27.0), 2.7, 0.9, 12, 0.09)
    # monkey-board platform with rails and crown
    box(P[ST], (cx, cy, 23.4), (6.0, 6.0, 1.0))
    box(P[DK], (cx, cy, 24.0), (6.2, 6.2, 0.15))
    railing(P[RS + '!'], [(cx - 3.1, cy - 3.1, 24.1), (cx + 3.1, cy - 3.1, 24.1), (cx + 3.1, cy + 3.1, 24.1), (cx - 3.1, cy + 3.1, 24.1),
                          (cx - 3.1, cy - 3.1, 24.1)], 1.1, 1.8)
    box(P[ST], (cx, cy, 27.4), (2.2, 2.2, 0.8))                       # crown block
    strut(P[RS + '!'], (cx, cy, 27.8), (cx, cy, 28.0), 0.25)
    cyl(P[DK], (cx, cy, 27.8), (cx, cy, 28.0), 0.09, 8)
    # inner rig floor + drill string
    cyl(P[GN], (cx, cy, 20.0), (cx, cy, 26.5), 0.22, 12)
    part('derrick', 'body', P)

    P = Pal()
    cx, cy, z0 = -4.5, 11.5, 22.8
    lattice(P[RS + '!'], (cx, cy, z0), (cx, cy, 26.6), 1.3, 0.45, 8, 0.07)
    box(P[ST], (cx, cy, 26.8), (1.5, 1.5, 0.4))
    part('derrick_small', 'body', P)


def build_cranes():
    for sg, nm in ((-1, 'crane_R'), (1, 'crane_L')):
        P = Pal()
        base = V(sg * 6.4, 6.0, 18.0)
        cyl(P[ST], base, base + V(0, 0, 1.6), 1.5, 24)
        box(P[ST], base + V(0, 0.9, 2.4), (2.6, 3.4, 1.8))          # cab / machinery house
        box(P[DK], base + V(sg * 0.0, -0.5, 3.4), (1.2, 1.0, 0.25))
        piv = base + V(0, 0, 2.0)
        h = V(sg * 1.0, -0.35, 0).normalized()
        el = radians(36)
        dvec = V(h.x * cos(el), h.y * cos(el), sin(el))
        tip = piv + dvec * 10.8
        lattice(P[RS + '!'], piv, tip, 0.65, 0.22, 9, 0.065, up=(0, 0, 1))
        # A-frame and pendant lines
        top = piv - h * 1.2 + V(0, 0, 4.6)
        for off in (-0.6, 0.6):
            strut(P[RS + '!'], piv - h * 1.4 + V(off * 0.9, 0, 0), top, 0.1)
        tube(P[CB], [top, (top + tip) / 2 + V(0, 0, 0.45), tip], 0.05, 5, 3)
        tube(P[CB], [piv + dvec * 5.0 + V(0, 0, 0.3), (piv + dvec * 5.0 + top) / 2 + V(0, 0, 0.5), top], 0.04, 5, 3)
        # hoist cable and hook block
        hook = tip + V(0, 0, -7.4)
        tube(P[CB], [tip, tip + V(0, 0, -3.7), hook + V(0, 0, 0.5)], 0.045, 5, 2)
        box(P[GN], hook + V(0, 0, 0.1), (0.55, 0.35, 0.8))
        cyl(P[GN], hook + V(0, 0, -0.3), hook + V(0, 0, -0.9), 0.1, 8)
        part(nm, 'body', P)


def build_lamps():
    P = Pal()
    P2 = Pal()
    for x in (-2.6, 2.6):
        cyl(P[LE], (x, -8.36, 15.6), (x, -8.46, 15.6), 0.34, 16)
        cyl(P2[GN], (x, -8.0, 15.6), (x, -8.36, 15.6), 0.5, 16)
        cyl(P2[GN], (x, -8.36, 15.6), (x, -8.4, 15.6), 0.52, 16)
    part('eye_lamps', 'body', P)
    part('eye_lamp_housings', 'body', P2)


# ----------------------------------------------------------------------------------------------
# rifle and pods
# ----------------------------------------------------------------------------------------------
def build_rifle():
    zc = RIFLE_PIV.z
    # mount (fixed to the body): mantlet ring on the hull front
    P = Pal()
    cyl(P[GN], (0, -7.6, zc), (0, -9.0, zc), 3.5, 28)
    cyl(P[ST], (0, -9.0, zc), (0, -9.3, zc), 3.0, 28)
    for k in range(12):
        a = 2 * pi * k / 12
        cyl(P[DK], (3.25 * cos(a), -9.0, zc + 3.25 * sin(a)), (3.25 * cos(a), -9.35, zc + 3.25 * sin(a)), 0.16, 6)
    box(P[GN], (0, -8.4, zc - 4.1), (7.5, 1.6, 1.4))            # yaw ring housing
    part('rifle_mount', 'body', P)

    P = Pal()  # receiver
    box(P[GN], (0, -10.2, zc), (5.8, 6.4, 5.0))                # y -13.4..-7.0
    box(P[GN], (0, -10.0, zc + 3.0), (3.6, 4.6, 1.4))           # capacitor housing on top
    box(P[DK], (0, -10.0, zc - 3.0), (3.0, 5.0, 1.0))           # underside rail
    for sg in (-1, 1):
        for k in range(13):                                     # radiator fins
            box(P[GN], (sg * 3.7, -7.8 - k * 0.45, zc), (1.7, 0.12, 4.2))
        box(P[ST], (sg * 3.0, -10.4, zc), (0.5, 5.0, 4.4))
        jag_plate(P[ST], (sg * 3.05, -10.2, zc + 0.3), (0, 1, 0), (0, 0, 1), 2.4, 1.9, 0.3)
    jag_plate(P[ST], (0, -9.8, zc + 3.75), (1, 0, 0), (0, 1, 0), 1.4, 2.0, 0.25)
    part('rifle_receiver', 'rifle_pitch', P)

    P = Pal()  # barrel sections joined by ring collars
    cyl(P[GN], (0, -13.2, zc), (0, -17.1, zc), 1.85, 28, 1.75)
    cyl(P[GN], (0, -18.0, zc), (0, -20.1, zc), 1.6, 28, 1.5)
    cyl(P[GN], (0, -21.0, zc), (0, -21.5, zc), 1.4, 28)
    for yc, rr in ((-17.55, 2.25), (-20.55, 2.05), (-13.4, 2.4)):
        cyl(P[GN], (0, yc - 0.5, zc), (0, yc + 0.5, zc), rr, 28)
        for k in range(10):                                     # collar bolts
            a = 2 * pi * k / 10
            cyl(P[DK], (rr * cos(a), yc - 0.35, zc + rr * sin(a)), (rr * cos(a), yc + 0.35, zc + rr * sin(a)), 0.13, 6)
    part('rifle_barrel', 'rifle_pitch', P)

    P = Pal()  # cooling rods around the third section, muzzle block
    for k in range(14):
        a = 2 * pi * k / 14
        cyl(P[DK], (1.85 * cos(a), -18.2, zc + 1.85 * sin(a)), (1.85 * cos(a), -21.4, zc + 1.85 * sin(a)), 0.14, 6)
    cyl(P[DK], (0, -21.1, zc), (0, -21.6, zc), 2.0, 20)
    box(P[GN], (0, -22.65, zc), (3.5, 2.7, 3.5))                # y -21.3..-24.0
    for sg in (-1, 1):
        jag_plate(P[ST], (sg * 1.78, -22.7, zc), (0, 1, 0), (0, 0, 1), 1.2, 1.5, 0.22)
    jag_plate(P[ST], (0, -22.6, zc + 1.8), (1, 0, 0), (0, 1, 0), 1.4, 1.1, 0.22)
    cyl(P[CB], (0, -23.97, zc), (0, -24.0, zc), 1.0, 20)        # bore
    for k in range(6):
        a = 2 * pi * k / 6
        cyl(P[DK], (1.2 * cos(a), -23.95, zc + 1.2 * sin(a)), (1.2 * cos(a), -24.05, zc + 1.2 * sin(a)), 0.1, 6)
    part('rifle_muzzle', 'rifle_pitch', P)

    # thick black power cables from the receiver into the hull
    P = Pal()
    for sg in (-1, 1):
        for k in range(4):
            z = zc - 1.3 + k * 0.85
            tube(P[CB], [(sg * 3.5, -12.6, z), (sg * 5.2, -12.2, z - 2.2 - 0.3 * k), (sg * 6.4, -9.8, z - 2.6 - 0.2 * k),
                         (sg * 6.8, -8.2, z - 1.2)], 0.28, 10)
        tube(P[CB], [(sg * 2.0, -12.4, zc + 2.6), (sg * 3.6, -12.4, zc + 4.2), (sg * 5.0, -9.0, zc + 3.8), (sg * 5.8, -7.8, zc + 2.4)], 0.32, 10)
    # big cable loops on the hull side
    for sg in (-1, 1):
        for k in range(7):
            y = 3.6 + k * 0.55
            tube(P[CB], [(sg * 9.65, y, 13.6), (sg * 10.35, y, 12.0), (sg * 10.4, y, 10.2), (sg * 9.65, y + 0.4, 9.6)], 0.26, 10)
    part('power_cables', 'body', P)


def build_pods():
    for sg, nm in ((-1, 'pod_R'), (1, 'pod_L')):
        P = Pal()
        piv = V(sg * 7.9, 1.2, 18.4)
        # cradle pylon on the hull top edge
        box(P[ST], (sg * 7.9, 1.0, 16.2), (3.4, 4.4, 4.4))
        jag_plate(P[ST], (sg * 9.65, 1.0, 16.6), (0, 1, 0), (0, 0, 1), 1.8, 1.8, 0.3)
        el, yaw = radians(45), radians(12) * sg
        f = Matrix.Rotation(yaw, 3, 'Z') @ V(0, -cos(el), sin(el))
        s_, u_, d_ = frame(f, (0, 0, 1))
        R = Rm(s_, u_, d_)
        box(P[ST], piv + d_ * 1.8, (3.7, 3.5, 5.2), R)
        box(P[GN], piv + d_ * 1.8 + u_ * 1.7, (3.3, 0.3, 4.6), R)               # top vent block
        for k in range(8):
            box(P[DK], piv + d_ * (0.2 + k * 0.5) + u_ * 1.9, (3.0, 0.1, 0.18), R)
        for sgn in (-1, 1):
            jag_plate(P[ST], piv + d_ * 1.6 + s_ * sgn * 1.9, d_, u_, 2.4, 1.6, 0.3)
        jag_plate(P[ST], piv + d_ * 1.2 - u_ * 1.78, d_, s_, 2.3, 1.7, 0.3)
        # launch tubes 3 x 4 poking out the front
        for a in range(4):
            for b_ in range(3):
                off = s_ * (-1.2 + 0.8 * a) + u_ * (-0.8 + 0.8 * b_)
                p0 = piv + d_ * 0.0 + off
                cyl(P[GN], p0, p0 + d_ * 6.4, 0.38, 14, caps=False)
                cyl(P[CB], p0 + d_ * 5.5, p0 + d_ * 5.55, 0.33, 14)             # dark inside
                cyl(P[DK], p0 + d_ * 4.4, p0 + d_ * 4.7, 0.46, 14)              # reinforcing ring
        box(P[ST], piv + d_ * 4.4, (3.8, 3.6, 0.35), R)                          # front frame
        part(nm, 'body', P)


# ----------------------------------------------------------------------------------------------
# legs
# ----------------------------------------------------------------------------------------------
def lerp(a, b, t):
    return a + (b - a) * t


def build_leg(sn, i):
    k = '%s%d' % (sn, i)
    sg, hip, knee, ank, pad_top = LEGGEO[k]
    yv = V(0, 1, 0)

    # coxa: housing on the hull side + hip hub  (fixed to the body)
    P = Pal()
    box(P[ST], (sg * 10.0, hip.y, 12.0), (1.8, 4.8, 4.6))
    cyl(P[GN], hip + V(0, -2.5, 0), hip + V(0, 2.5, 0), 1.75, 24, up=(0, 0, 1))
    cyl(P[DK], hip + V(0, -2.7, 0), hip + V(0, -2.5, 0), 1.2, 24, up=(0, 0, 1))
    cyl(P[DK], hip + V(0, 2.5, 0), hip + V(0, 2.7, 0), 1.2, 24, up=(0, 0, 1))
    for q in range(8):
        a = 2 * pi * q / 8
        cyl(P[DK], hip + V(1.4 * cos(a), -2.5, 1.4 * sin(a)), hip + V(1.4 * cos(a), -2.75, 1.4 * sin(a)), 0.12, 6)
    jag_plate(P[ST], (sg * 11.0, hip.y, 14.3), (0, 1, 0), (1, 0, 0), 2.2, 1.0, 0.3)
    part(k + '_coxa', 'body', P)

    # upper segment (rises from hip to the high knee)
    dU = (knee - hip).normalized()
    sU, uU, _ = frame(dU, yv)
    pu = sU if sU.dot(V(0, 0, 1)) > 0 else -sU          # in-plane perpendicular facing up
    po = sU if sU.dot(V(sg, 0, 0)) > 0 else -sU         # in-plane perpendicular facing outward
    P = Pal()
    seg(P[ST], hip, knee, 1.3, 1.65, 1.2, 1.55, yv, 0.35)
    cyl(P[GN], knee + V(0, -2.2, 0), knee + V(0, 2.2, 0), 1.75, 24, up=(0, 0, 1))
    cyl(P[DK], knee + V(0, -2.4, 0), knee + V(0, -2.2, 0), 1.2, 24, up=(0, 0, 1))
    cyl(P[DK], knee + V(0, 2.2, 0), knee + V(0, 2.4, 0), 1.2, 24, up=(0, 0, 1))
    mid = (hip + knee) / 2
    for sgn in (-1, 1):
        jag_plate(P[ST], mid + yv * sgn * 1.75, dU, pu, 2.3, 1.1, 0.3)
    jag_plate(P[RS], mid + pu * 1.45, dU, yv, 2.0, 1.5, 0.3)
    # hose runs from hull to knee
    for sgn in (-1, 1):
        tube(P[CB], [hip + V(-sg * 1.2, sgn * 1.0, 1.6), mid + V(-sg * 0.9, sgn * 2.0, 2.4), knee + V(-sg * 0.5, sgn * 1.2, 0.2)], 0.17, 8)
    part(k + '_upper', k + '_upper', P)

    # lower segment (angles down and out to the ankle)
    dL = (ank - knee).normalized()
    sL, uL, _ = frame(dL, yv)
    poL = sL if sL.dot(V(sg, 0, 0)) > 0 else -sL
    P = Pal()
    seg(P[ST], knee, ank, 1.2, 1.7, 0.85, 1.25, yv, 0.3, mid=(1.1, 1.6))
    cyl(P[GN], ank + V(0, -1.8, 0), ank + V(0, 1.8, 0), 1.2, 22, up=(0, 0, 1))
    cyl(P[DK], ank + V(0, -2.0, 0), ank + V(0, -1.8, 0), 0.8, 22, up=(0, 0, 1))
    cyl(P[DK], ank + V(0, 1.8, 0), ank + V(0, 2.0, 0), 0.8, 22, up=(0, 0, 1))
    for t in (0.3, 0.68):
        c = knee.lerp(ank, t)
        wy = lerp(1.7, 1.25, t) + 0.12
        for sgn in (-1, 1):
            jag_plate(P[ST if t < 0.5 else RS], c + yv * sgn * wy, dL, poL, 2.1, 1.0, 0.3)
    c = knee.lerp(ank, 0.45)
    jag_plate(P[ST], c + poL * 1.25, dL, yv, 3.0, 1.55, 0.3)
    jag_plate(P[RS], knee + poL * 1.9 + V(0, 0, 0.4), dL, yv, 1.2, 1.7, 0.3)      # knee guard
    part(k + '_lower', k + '_lower', P)

    # pistons: A hull->upper (cylinder on body, rod on upper); B upper->lower; ankle piston lower->foot
    def piston(Pc, Pr, a, b, rc=0.42, rr=0.24, fr=0.55):
        m = a.lerp(b, fr)
        cyl(Pc[GN], a, m, rc, 14)
        cyl(Pr[ST], m, b, rr, 12)
        cyl(Pc[DK], a.lerp(m, 0.9), a.lerp(m, 1.0), rc * 1.25, 14)
        cyl(Pr[DK], b - (b - m).normalized() * 0.3, b, rr * 1.7, 10)
    PA_c, PA_r = Pal(), Pal()
    PB_c, PB_r = Pal(), Pal()
    PC_c, PC_r = Pal(), Pal()
    for sgn in (-1, 1):
        o = yv * sgn * 1.0
        a = V(sg * 8.0, hip.y, 14.5) + o
        b = hip.lerp(knee, 0.88) - pu * 1.9 + o * 1.1
        piston(PA_c, PA_r, a, b)
        a = hip.lerp(knee, 0.22) + po * 1.7 + o * 1.4
        b = knee.lerp(ank, 0.36) + poL * 1.5 + o * 1.4
        piston(PB_c, PB_r, a, b, 0.48, 0.28)
        a = knee.lerp(ank, 0.72) + poL * 1.1 + o * 1.6
        b = pad_top + V(sg * 1.3, 0, 0.35) + o * 1.4
        piston(PC_c, PC_r, a, b, 0.36, 0.2)
    part(k + '_piston_hull', 'body', PA_c)
    part(k + '_piston_hull_rod', k + '_upper', PA_r)
    part(k + '_piston_knee', k + '_upper', PB_c)
    part(k + '_piston_knee_rod', k + '_lower', PB_r)
    part(k + '_piston_ankle', k + '_lower', PC_c)
    part(k + '_piston_ankle_rod', k + '_foot', PC_r)

    # foot: padded round pad with toe plates, ankle block
    P = Pal()
    fx, fy = pad_top.x, pad_top.y
    cyl(P[RB], (fx, fy, 0.0), (fx, fy, 0.5), 2.45, 20, 2.3)
    cyl(P[ST], (fx, fy, 0.5), (fx, fy, 0.95), 2.15, 20, 1.9)
    for q in range(5):
        a = pi / 2 + 2 * pi * q / 5 + (0.15 if i == 2 else 0.0)
        cr = V(cos(a), sin(a), 0)
        Rz = Rm(cr, V(-sin(a), cos(a), 0), V(0, 0, 1))
        c = V(fx, fy, 0.0) + cr * 2.65
        box(P[ST], c + V(0, 0, 0.45), (1.5, 1.3, 0.7), Rz)
        box(P[RB], c + V(0, 0, 0.15), (1.9, 1.5, 0.3), Rz)
    box(P[ST], (fx, fy, 2.0), (2.2, 2.6, 2.1))
    for sgn in (-1, 1):
        jag_plate(P[ST], (fx, fy + sgn * 1.4, 2.0), (1, 0, 0), (0, 0, 1), 1.0, 1.0, 0.25)
    for q in range(6):
        a = 2 * pi * q / 6
        cyl(P[DK], (fx + 1.9 * cos(a), fy + 1.9 * sin(a), 0.95), (fx + 1.9 * cos(a), fy + 1.9 * sin(a), 1.12), 0.14, 6)
    part(k + '_foot', k + '_foot', P)


# ----------------------------------------------------------------------------------------------
# build all
# ----------------------------------------------------------------------------------------------
T_B = time.time()
build_hull()
build_bridge()
build_derrick()
build_cranes()
build_lamps()
build_rifle()
build_pods()
for sn, _ in SIDES:
    for i, *_r in LEGS:
        build_leg(sn, i)
BUILD_T = time.time() - T_B
print('built', len(PARTS), 'parts', len(OBJS), 'objects', round(BUILD_T, 1), 's', flush=True)

# ----------------------------------------------------------------------------------------------
# UV atlas + CPU bake
# ----------------------------------------------------------------------------------------------
PAINT = 'M_Crawler_Paint'
TEX = {}
BAKE_INFO = {}
ATLAS_OK = False
BAKED_OK = False
_w = bpy.data.worlds.new('World')
SCN.world = _w
BAKE_T = 0.0


def unwrap():
    global ATLAS_OK
    for ob in CANON:
        ob.data.uv_layers.new(name='UV_Atlas')
    bpy.ops.object.select_all(action='DESELECT')
    for ob in CANON:
        ob.data.uv_layers.active = ob.data.uv_layers['UV_Atlas']
        ob.select_set(True)
    bpy.context.view_layer.objects.active = CANON[0]
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=radians(66), island_margin=0.0, area_weight=0.0, scale_to_bounds=False)
        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.uv.average_islands_scale()
        bpy.ops.uv.pack_islands(rotate=True, margin=0.0015)
        bpy.ops.object.mode_set(mode='OBJECT')
        ATLAS_OK = True
    except Exception as e:
        print('ATLAS FAILED', e)
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    for ob in CANON:
        ob.data.uv_layers.active = ob.data.uv_layers['UV_Atlas']
        ob.data.uv_layers['UV_Atlas'].active_render = True


def bake_textures():
    os.makedirs(TEX_DIR, exist_ok=True)
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
    used = sorted({m.name for m in joined.data.materials})
    SCN.render.engine = 'CYCLES'
    SCN.cycles.device = 'CPU'                     # the GPU is reserved for another job
    SCN.cycles.samples = 16
    SCN.cycles.use_denoising = False
    SCN.world.light_settings.distance = 4.0
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

    IM = dict(base=new_img(NAME + '_paint_basecolor', True, (0.05, 0.05, 0.05, 1)), rough=new_img('r_p', False, (0.5, 0.5, 0.5, 1)),
              metal=new_img('m_p', False, (0, 0, 0, 1)), ao=new_img('a_p', False, (1, 1, 1, 1)),
              normal=new_img(NAME + '_paint_normal', False, (0.5, 0.5, 1.0, 1)), orm=new_img(NAME + '_paint_orm', False, (1, 0.5, 0, 1)))
    tex = {n: bpy.data.materials[n].node_tree.nodes.new('ShaderNodeTexImage') for n in used}

    def target(kind):
        for n in used:
            tex[n].image = IM[kind]
            bpy.data.materials[n].node_tree.nodes.active = tex[n]
        bpy.ops.object.select_all(action='DESELECT')
        joined.select_set(True)
        bpy.context.view_layer.objects.active = joined

    def emit_bake(kind, socket, is_color):
        saved = []
        for n in used:
            nt = bpy.data.materials[n].node_tree
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
    print('bake normal', round(time.time() - T0, 1), 's', flush=True)
    SCN.cycles.samples = 24
    target('ao')
    bpy.ops.object.bake(type='AO', margin=6, use_clear=False)
    print('bake ao', round(time.time() - T0, 1), 's', flush=True)
    ao = np.clip(get(IM['ao'])[:, 0], 0, 1)
    r, m = get(IM['rough']), get(IM['metal'])
    base = get(IM['base'])
    base[:, :3] *= (0.45 + 0.55 * ao)[:, None]
    IM['base'].pixels.foreach_set(base.ravel())
    px = np.ones_like(r)
    px[:, 0], px[:, 1], px[:, 2] = ao, r[:, 0], m[:, 0]
    IM['orm'].pixels.foreach_set(px.ravel())
    for key, kk in (('basecolor', 'base'), ('orm', 'orm'), ('normal', 'normal')):
        im = IM[kk]
        im.filepath_raw = os.path.join(TEX_DIR, '%s_paint_%s.png' % (NAME, key))
        im.file_format = 'PNG'
        im.save()
        TEX[key] = im
    for kk in ('rough', 'metal', 'ao'):
        bpy.data.images.remove(IM[kk])
    for n in used:
        bpy.data.materials[n].node_tree.nodes.remove(tex[n])
    a3 = sum(p.area for p in joined.data.polygons)
    bpy.data.objects.remove(joined)
    for ob in CANON:
        ob.hide_render = False
    BAKE_INFO.update(size=N, surface_area_m2=round(a3, 1), maps=['basecolor', 'orm', 'normal'], device='CPU')


def textured():
    m = bpy.data.materials.new(PAINT)
    m.use_nodes = True
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
        t.image = TEX[key]
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
    MATS[PAINT] = m
    return m


if DO_BAKE:
    unwrap()
    print('UV', ATLAS_OK, round(time.time() - T0, 1), 's', flush=True)
    if ATLAS_OK:
        tb = time.time()
        bake_textures()
        BAKE_T = time.time() - tb
        mp = textured()
        for ob in CANON:
            me = ob.data
            me.materials.clear()
            me.materials.append(mp)
            me.polygons.foreach_set('material_index', [0] * len(me.polygons))
            me.update()
        BAKED_OK = True
        print('BAKED', json.dumps(BAKE_INFO), round(BAKE_T, 1), 's', flush=True)

# ----------------------------------------------------------------------------------------------
# armature, rigid skinning, IK
# ----------------------------------------------------------------------------------------------
arm_data = bpy.data.armatures.new('Armature')
ARM = bpy.data.objects.new('Armature', arm_data)
COL.objects.link(ARM)
bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = ARM
ARM.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
for n, (h, t, p) in BONES.items():
    eb = arm_data.edit_bones.new(n)
    eb.head = h
    eb.tail = t
    if (t - h).length < 1e-3:
        eb.tail = h + V(0, 0, 1)
    if p:
        eb.parent = arm_data.edit_bones[p]
    eb.use_connect = False
bpy.ops.object.mode_set(mode='OBJECT')
arm_data.display_type = 'STICK'
ARM.show_in_front = True
for ob, bn in OBJS:
    vg = ob.vertex_groups.new(name=bn)
    vg.add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
    md = ob.modifiers.new('Armature', 'ARMATURE')
    md.object = ARM
    ob.parent = ARM

bpy.ops.object.mode_set(mode='POSE')
PB = ARM.pose.bones
IK_DEV = {}
for sn, sg in SIDES:
    for i, *_r in LEGS:
        k = '%s%d' % (sn, i)
        c = PB[k + '_lower'].constraints.new('IK')         # 2-bone IK (hip->knee->ankle); the foot bone copies the IK target's rotation
        c.target = ARM
        c.subtarget = 'IK_' + k
        c.pole_target = ARM
        c.pole_subtarget = 'POLE_' + k
        c.chain_count = 2
        cr = PB[k + '_foot'].constraints.new('COPY_ROTATION')
        cr.target = ARM
        cr.subtarget = 'IK_' + k
        cr.target_space = 'WORLD'
        cr.owner_space = 'WORLD'
        def dev_at(deg):
            c.pole_angle = radians(deg)
            bpy.context.view_layer.update()
            EB = ARM.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones
            return max((EB[k + '_lower'].head - PB[k + '_lower'].bone.head_local).length,
                       (EB[k + '_foot'].head - PB[k + '_foot'].bone.head_local).length,
                       (EB[k + '_foot'].tail - PB[k + '_foot'].bone.tail_local).length)
        d0 = min(range(-180, 180, 10), key=dev_at)
        d1 = min([d0 + x * 0.5 for x in range(-20, 21)], key=dev_at)
        best = (dev_at(d1), radians(d1))
        c.pole_angle = best[1]
        bpy.context.view_layer.update()
        IK_DEV[k] = round(best[0], 4)
bpy.ops.object.mode_set(mode='OBJECT')
print('IK rest deviation per leg (m):', IK_DEV, flush=True)

# ----------------------------------------------------------------------------------------------
# checks + info
# ----------------------------------------------------------------------------------------------
bpy.context.view_layer.update()
INFO_PARTS = {}
alln = np.empty((0, 3))
bbs = {}
for ob, bn in OBJS:
    co = np.empty(len(ob.data.vertices) * 3, dtype=np.float32)
    ob.data.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    bbs[ob.name] = (co.min(0), co.max(0), bn)
    INFO_PARTS[ob.name] = dict(bone=bn, verts=len(co), faces=len(ob.data.polygons),
                               bbox=[[round(float(v), 2) for v in co.min(0)], [round(float(v), 2) for v in co.max(0)]])
mn = np.min([b[0] for b in bbs.values()], axis=0)
mx = np.max([b[1] for b in bbs.values()], axis=0)
feet = {}
for ob, bn in OBJS:
    if bn.endswith('_foot') and ob.name.endswith('_foot'):
        feet[bn[:-5]] = round(float(bbs[ob.name][0][2]), 3)
# connectivity of part bounding boxes (floating-part check)
names = list(bbs)
parent = {n: n for n in names}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for ia in range(len(names)):
    for ib in range(ia + 1, len(names)):
        a, b = bbs[names[ia]], bbs[names[ib]]
        if np.all(a[0] <= b[1] + 0.1) and np.all(b[0] <= a[1] + 0.1):
            parent[find(names[ia])] = find(names[ib])
comps = {}
for n in names:
    comps.setdefault(find(n), []).append(n)
comp_list = sorted(comps.values(), key=len, reverse=True)
floating = [c for c in comp_list[1:]]
faces = sum(len(ob.data.polygons) for ob, _ in OBJS)
tris = sum(len(p.vertices) - 2 for ob, _ in OBJS for p in ob.data.polygons)
mats = sorted({m.name for ob, _ in OBJS for m in ob.data.materials})
dims = [round(float(mx[i] - mn[i]), 2) for i in range(3)]
info = dict(name=NAME, wear=WEAR, size_m=dict(x_width=dims[0], y_length=dims[1], z_height=dims[2]),
            bbox=[[round(float(v), 2) for v in mn], [round(float(v), 2) for v in mx]],
            parts=len(PARTS), objects=len(OBJS), bones=len(BONES), faces=faces, tris=tris, materials=mats,
            baked=BAKED_OK, bake=BAKE_INFO, build_seconds=round(BUILD_T, 1), bake_seconds=round(BAKE_T, 1),
            foot_min_z=feet, ik_rest_deviation=IK_DEV, floating_groups=floating, part_bone={p: b for p, b in PARTS.items()},
            part_bboxes=INFO_PARTS, bone_list=list(BONES))
print('PARTS', len(PARTS), 'OBJECTS', len(OBJS), 'BONES', len(BONES), 'FACES', faces, 'SIZE', dims, 'FEET_MIN_Z', feet)
print('floating groups:', floating)
if len(comp_list) > 1:
    print('main group size', len(comp_list[0]), 'of', len(names))

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, NAME + '.blend'))
if BAKED_OK:
    try:
        bpy.ops.file.make_paths_relative()
        bpy.ops.wm.save_mainfile()
    except Exception as e:
        print('relpaths', e)

if DO_EXPORT:
    bpy.ops.object.select_all(action='DESELECT')
    for ob, _ in OBJS:
        ob.select_set(True)
    ARM.select_set(True)
    bpy.context.view_layer.objects.active = ARM
    try:
        bpy.ops.export_scene.gltf(filepath=os.path.join(OUT_DIR, NAME + '.glb'), export_format='GLB', use_selection=True,
                                  export_apply=False, export_skins=True, export_yup=True, export_animations=False,
                                  export_image_format='JPEG', export_jpeg_quality=88, export_extras=False)
        info['glb_bytes'] = os.path.getsize(os.path.join(OUT_DIR, NAME + '.glb'))
    except Exception as e:
        print('GLB FAILED', e)
        info['glb_error'] = str(e)
info['total_seconds'] = round(time.time() - T0, 1)
with open(os.path.join(OUT_DIR, NAME + '_info.json'), 'w') as fh:
    json.dump(info, fh, indent=1)
print('DONE', info['total_seconds'], 's')
