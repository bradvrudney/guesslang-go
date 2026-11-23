"""Turn model/weights.json (from extract.py) into model/model.bin, which the
Go package embeds: little-endian float32 tensors, each stored as
  uint16 name length, name (utf-8), uint8 number of dimensions,
  uint32 each dimension, then the values row by row.
The file starts with the 4 bytes "GLG1" and a uint16 tensor count.
Standard library only."""
import json, struct, sys

root = sys.argv[1]
w = json.load(open(root + '/model/weights.json'))
out = bytearray(b'GLG1' + struct.pack('<H', len(w)))
for name in sorted(w):
    t = w[name]
    dims = []
    x = t
    while isinstance(x, list):
        dims.append(len(x))
        x = x[0] if x else None
    flat = []
    def walk(a):
        for e in a:
            if isinstance(e, list):
                walk(e)
            else:
                flat.append(e)
    walk(t)
    nb = name.encode()
    out += struct.pack('<H', len(nb)) + nb + struct.pack('<B', len(dims))
    for d in dims:
        out += struct.pack('<I', d)
    out += struct.pack('<%df' % len(flat), *flat)
    print(name, dims)
open(root + '/model/model.bin', 'wb').write(out)
print(len(out), 'bytes')
