# Rig Crawler (AnyMachine build)

`build_crawler.py` builds the whole machine from code: ~40 m six-legged crawler made from a converted oil rig.
Design of record: `runs/2026-11-23_002_crawler-turnaround/outputs/gun_rifle_vls_s8403_00001_.png` plus the s8100 turnaround
and s8200 close-ups. Metres, Z up, front = -Y, origin on the ground under the machine centre. Right-hand side = -X (letter R), left = +X (L).

## Run
```
blender.exe -b --factory-startup -P crawler/build_crawler.py -- [--wear 0..1] [--no-bake] [--no-export] [--out DIR] [--name NAME] [--tex 4096]
```
| Option | Effect |
|---|---|
| `--wear` (default 0.5) | industrial wear layered over the materials (adapted from Stilt): rust patches, rain/oil streaks, crevice dirt, chipped edges, rougher. Baked into the textures. 0 = clean paint |
| `--no-bake` | skip UV atlas + bake (procedural materials, no UVs); build takes ~3 s |
| `--no-export` | skip the GLB |
| `--out` | output dir (default `runs/2026-11-23_003_crawler-builds/outputs/anymachine/`) |
| `--name` | base name for `.blend`, `.glb`, `_info.json`, `<name>_textures/` |
| `--tex` | atlas size, default 4096 (~16,100 m2 of surface, about 3 cm per texel) |

Outputs: `<name>.blend`, `<name>.glb` (skinned, no animation), `<name>_info.json` (size, parts, bones, faces, materials, times,
foot min z, per-part bounding boxes, floating-group check), `<name>_textures/<name>_paint_{basecolor,orm,normal}.png`.
Bake is Cycles on CPU (the GPU stays free): about 2 minutes total at 4096, wear 0.6.

## Result (wear 0.6)
Size 38.4 m wide (leg to leg, incl. pads) x 40.85 m long (rifle muzzle to rear foot) x 28.0 m high (derrick tip). 90 parts, 34 bones, 46,753 faces (95,686 tris).
Hull underside z = 8.3 m. All six pads have min z = 0.0. Part bounding boxes form one connected group (no floating parts).

## Parts (90)
- Hull: `hull_lower`, `hull_mid` (with sloped front glacis), `hull_armour` (jagged plates), `catwalks_lower`, `stairs`, `roof_deck`, `machinery_blocks`, `roof_pipes`, `stacks`
- Bridge: `bridge` (walls, window pillars), `bridge_glass_glass`, `bridge_interior_light_win` (emissive), `bridge_roof_gear`, `crew_1..4_crew` (1.8 m mannequins behind the glass), `eye_lamps_lamp`, `eye_lamp_housings`
- Derrick (lattice from real thin struts): `derrick` (28.0 m top, monkey board, crown), `derrick_small`; cranes `crane_L`, `crane_R` (lattice boom, A-frame, cable, hook block)
- Rifle: `rifle_mount`, `rifle_receiver` (radiator fins), `rifle_barrel` (3 sections + ring collars), `rifle_muzzle` (cooling rods, boxy muzzle), `power_cables`
- Pods: `pod_L`, `pod_R` (3 x 4 launch tubes, 45 deg up, 12 deg splayed outward)
- Per leg (L1..L3, R1..R3, 1 = front): `_coxa`, `_upper`, `_lower`, `_foot` (padded round pad + 5 toe plates), 6 piston parts

Materials: painted-metal families steel / deck / gunmetal / rust-orange / cable / rubber (baked into one atlas `M_Crawler_Paint`:
tone variation, roughness noise, triplanar panel seams + rivet grid as bump) and plain glass, window light, eye lamps, crew.

## Rig
Bones: `root`, `body`, `rifle_yaw`, `rifle_pitch`, per leg `Xn_upper/_lower/_foot`, `IK_Xn`, `POLE_Xn`. Meshes are rigidly skinned
(one vertex group, weight 1). IK: 2-bone IK on `_lower` (hip -> knee -> ankle) to `IK_Xn`; the foot bone copies the IK target rotation.
Rest pose is the planted standing stance; IK rest deviation is about 3 cm. IK constraints are not exported to GLB. No animation.

## Known limits
Not checked visually (no renders were allowed when it was written). Rivets and seams live in the bump/normal map only; legs are not
instanced, so each has its own atlas area.
