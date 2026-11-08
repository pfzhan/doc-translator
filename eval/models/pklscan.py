"""List the global names referenced by the pickles in a .pt (zip) without executing them. Usage: pklscan.py <pt>"""

import collections
import pickletools
import sys
import zipfile

z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if n.endswith(".pkl")]
print("pkl files:", names, "entries:", len(z.namelist()))
refs: collections.Counter[str] = collections.Counter()
for n in names:
    data = z.read(n)
    strings: list[str] = []
    for op, arg, _pos in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            assert isinstance(arg, str)
            strings.append(arg)
        elif op.name == "STACK_GLOBAL":
            refs[f"{strings[-2]}.{strings[-1]}"] += 1
        elif op.name in ("GLOBAL", "INST"):
            # The argument of GLOBAL and INST is "module name"
            assert isinstance(arg, str)
            refs[arg.replace(" ", ".")] += 1
        elif op.name in ("REDUCE", "BUILD", "NEWOBJ", "OBJ"):
            pass
for k, v in sorted(refs.items()):
    print(v, k)
