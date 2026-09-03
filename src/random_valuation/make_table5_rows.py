import csv
from pathlib import Path
R=Path(__file__).resolve().parents[2]/"artifacts"/"random_valuation"
rows=list(csv.DictReader(open(R/"results/experiment_zfix.csv")))
def mean(x): return sum(x)/len(x)
def agg(method,scale,cost,field):
    v=[float(r[field]) for r in rows if r["method"]==method and r["variant"]=="fixed" and r["scale"]==scale and r["cost"]==cost and r[field] not in("","None")]
    return mean(v)
# Published Table 5 (current paper): InS, OOS, Time. CPBSD-A reused wholesale;
# FCP/BSP keep published Time (Z-fix does not affect runtime), update InS/OOS.
PUB={("N10_K50","zero"):{"FCP":(51.911,50.440,0.084),"BSP":(51.996,50.081,0.541),"CPBSD-A":(52.009,50.009,1.828)},
     ("N10_K50","random_ind"):{"FCP":(19.970,18.867,1.054),"BSP":(18.112,16.950,1.528),"CPBSD-A":(15.895,15.287,5.742)},
     ("N10_K50","random_corr"):{"FCP":(27.917,25.960,1.191),"BSP":(27.390,25.528,1.365),"CPBSD-A":(22.020,21.367,12.899)},
     ("N30_K50","zero"):{"FCP":(158.769,156.569,0.114),"BSP":(159.008,156.282,1.367),"CPBSD-A":(159.051,154.968,300.147)},
     ("N30_K50","random_ind"):{"FCP":(56.410,54.287,2.860),"BSP":(46.391,44.441,35.186),"CPBSD-A":(24.154,22.829,300.141)},
     ("N30_K50","random_corr"):{"FCP":(83.593,79.377,4.686),"BSP":(81.181,77.362,5.149),"CPBSD-A":(60.684,58.735,300.162)}}
SC=[("N10_K50","N=10,K=50"),("N30_K50","N=30,K=50")]
COST=[("zero","zero"),("random_ind","random\\_ind"),("random_corr","random\\_corr")]
def b(val,best): 
    s="%.3f"%val
    return "\\textbf{%s}"%s if abs(val-best)<5e-4 else s
out=[]
for si,(sc,sclab) in enumerate(SC):
    for ci,(co,colab) in enumerate(COST):
        p=PUB[(sc,co)]
        fcp=(agg("FCP",sc,co,"ins"),agg("FCP",sc,co,"oos"),p["FCP"][2])   # InS/OOS corrected, Time published
        bsp=(agg("BSP",sc,co,"ins"),agg("BSP",sc,co,"oos"),p["BSP"][2])
        ca=p["CPBSD-A"]
        bins=max(fcp[0],bsp[0],ca[0]); boos=max(fcp[1],bsp[1],ca[1]); btim=min(fcp[2],bsp[2],ca[2])
        def cell(t): return f"{b(t[0],bins)} & {b(t[1],boos)} & {b(t[2],btim)}"
        lead=f"\\({sclab}\\) & \\texttt{{{colab}}}" if ci==0 else f" & \\texttt{{{colab}}}"
        out.append(f"{lead} & {cell(fcp)} & {cell(bsp)} & {cell(ca)} \\\\")
    if si==0: out.append("\\midrule")
print("\n".join(out))
