import re, sys
# stage bodies = spans between consecutive consumer "empty" arrives (SYNCS.ARRIVE.TRANS64.A1T0); the first span starts
# at the first full-barrier try-wait. Prints length, R2UR, HGMMA, LDL/STL per span.
for f in sys.argv[1:]:
    ins = [re.sub(r'^\s*/\*[0-9a-f]+\*/\s*', '', l).split(';')[0].strip()
           for l in open(f) if re.match(r'^\s+/\*[0-9a-f]{4,}\*/', l)]
    start = next(i for i, t in enumerate(ins) if 'TRYWAIT' in t)
    ends = [i for i, t in enumerate(ins) if 'SYNCS.ARRIVE.TRANS64.A1T0' in t]
    print(f, len(ins), 'instructions')
    a = start
    for e in ends:
        seg = ins[a:e + 1]
        cnt = lambda k: sum(k in t for t in seg)
        print(f'  [{a:5d},{e:5d}] len {len(seg):4d}  HGMMA {cnt("HGMMA"):3d}  R2UR {cnt("R2UR"):3d}  IMAD.X {cnt("IMAD.X"):3d}  LDL {cnt("LDL"):2d} STL {cnt("STL"):2d}')
        a = e + 1
