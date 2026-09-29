#!/usr/bin/env python3
"""Print the HFQ header/index (no tensor data read). usage: hfq_index.py FILE [filter]"""
import struct, sys, json

def read_index(path):
    with open(path, 'rb') as f:
        hdr = f.read(32)
        magic, ver, arch, ntens, meta_off, data_off = struct.unpack('<4sIIIQQ', hdr)
        f.seek(meta_off)
        # index starts right after metadata; metadata size unknown -> read until index parse works
        blob = f.read(data_off - meta_off)
    # metadata JSON: find end by json raw decode
    dec = json.JSONDecoder()
    meta, end = dec.raw_decode(blob.decode('latin-1'))
    p = end
    (cnt,) = struct.unpack_from('<I', blob, p); p += 4
    tensors = []
    off = data_off
    for _ in range(cnt):
        (nl,) = struct.unpack_from('<H', blob, p); p += 2
        name = blob[p:p+nl].decode(); p += nl
        qt, nd = struct.unpack_from('<BB', blob, p); p += 2
        shape = list(struct.unpack_from('<%dI' % nd, blob, p)); p += 4*nd
        gs, dl = struct.unpack_from('<IQ', blob, p); p += 12
        tensors.append(dict(name=name, qt=qt, shape=shape, gs=gs, nbytes=dl, offset=off))
        off += dl
    return dict(magic=magic, ver=ver, arch=arch, ntens=ntens, meta_off=meta_off, data_off=data_off, meta=meta, tensors=tensors)

if __name__ == '__main__':
    r = read_index(sys.argv[1])
    flt = sys.argv[2] if len(sys.argv) > 2 else None
    print('magic', r['magic'], 'ver', r['ver'], 'arch', r['arch'], 'ntens', r['ntens'], 'data_off', r['data_off'])
    m = dict(r['meta'])
    for k in list(m.keys()):
        s = json.dumps(m[k])
        if len(s) > 300: m[k] = s[:300] + '...'
    print(json.dumps(m, indent=1)[:6000])
    for t in r['tensors']:
        if flt and flt not in t['name']: continue
        print(t['name'], 'qt', t['qt'], 'shape', t['shape'], 'gs', t['gs'], 'nbytes', t['nbytes'], 'off', t['offset'])
