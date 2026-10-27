"""只读扫描 .pt（zip）里 pickle 引用的全局名，不执行。用法: pklscan.py <pt>"""
import sys, zipfile, pickletools, collections
z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if n.endswith(".pkl")]
print("pkl files:", names, "entries:", len(z.namelist()))
refs = collections.Counter()
for n in names:
    data = z.read(n)
    strings = []
    for op, arg, pos in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strings.append(arg)
        elif op.name == "STACK_GLOBAL":
            refs[f"{strings[-2]}.{strings[-1]}"] += 1
        elif op.name in ("GLOBAL", "INST"):
            refs[arg.replace(" ", ".")] += 1
        elif op.name in ("REDUCE", "BUILD", "NEWOBJ", "OBJ"):
            pass
for k, v in sorted(refs.items()):
    print(v, k)
