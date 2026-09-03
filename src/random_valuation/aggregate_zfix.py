"""Aggregate the z-fix sweep into before/after tables + paired significance.

Inputs : results/experiment_zfix.csv (FCP & BSP, variant in {buggy,fixed}, 5 seeds)
Outputs: results/zfix_report.md  (printed too)

Table A: Z-fix effect (buggy vs fixed solver, SAME retrained model & instances)
         per scale x cost: mean InS/OOS over seeds, delta, paired t (df=4).
Table B: corrected (fixed) FCP/BSP vs the published Table 5, CPBSD-A reused.
"""
import csv, math
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "artifacts" / "random_valuation"
CSV = R / "results" / "experiment_zfix.csv"
OUT = R / "results" / "zfix_report.md"
COSTS = ["zero", "random_ind", "random_corr"]
SCALES = ["N10_K50", "N30_K50"]

# Published Table 5 (current paper): {scale: {cost: {method: (InS, OOS, Time)}}}
PUB = {
 "N10_K50": {"zero":(("FCP",51.911,50.440,0.084),("BSP",51.996,50.081,0.541),("CPBSD-A",52.009,50.009,1.828)),
             "random_ind":(("FCP",19.970,18.867,1.054),("BSP",18.112,16.950,1.528),("CPBSD-A",15.895,15.287,5.742)),
             "random_corr":(("FCP",27.917,25.960,1.191),("BSP",27.390,25.528,1.365),("CPBSD-A",22.020,21.367,12.899))},
 "N30_K50": {"zero":(("FCP",158.769,156.569,0.114),("BSP",159.008,156.282,1.367),("CPBSD-A",159.051,154.968,300.147)),
             "random_ind":(("FCP",56.410,54.287,2.860),("BSP",46.391,44.441,35.186),("CPBSD-A",24.154,22.829,300.141)),
             "random_corr":(("FCP",83.593,79.377,4.686),("BSP",81.181,77.362,5.149),("CPBSD-A",60.684,58.735,300.162))},
}


def t_sf_two_sided(t, df=4, steps=200000, hi=80.0):
    t = abs(t)
    if math.isnan(t): return float("nan")
    if t == 0: return 1.0
    c = 3.0/8.0  # t-pdf df=4 normalizing constant
    h = (hi - t)/steps; s = 0.0
    for i in range(steps+1):
        x = t + i*h; w = 1 if i in (0,steps) else (4 if i % 2 else 2)
        s += w * c*(1+x*x/4)**-2.5
    return 2*(s*h/3)


def mean(x): return sum(x)/len(x)
def sd(x):
    m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/(len(x)-1)) if len(x)>1 else 0.0
def paired(deltas):
    m=mean(deltas); s=sd(deltas); n=len(deltas)
    t = m/(s/math.sqrt(n)) if s>0 else float("nan")
    return m, t, (t_sf_two_sided(t) if not math.isnan(t) else (1.0 if abs(m)<1e-9 else 0.0))


rows = list(csv.DictReader(open(CSV)))
def cell(method, variant, scale, cost, field):
    vals=[(r["seed"], float(r[field])) for r in rows
          if r["method"]==method and r["variant"]==variant and r["scale"]==scale and r["cost"]==cost and r[field] not in ("","None")]
    return dict(vals)

L=["# Z-fix re-run — random-valuation experiments","",
   "Retrained Hanson GCN on z-fixed MB labels (eval F1 0.944 / test F1 0.946 vs old 0.940/0.943).",
   "FCP & BSP solved under the original (buggy, Z lb=0) and fixed (Z lb=-inf) solver on the",
   "SAME model and SAME Table-5 instances (seeds 20260413-20260417). CPBSD-A is unaffected and",
   "reused from the published Table 5. Paired t-test df=4, two-sided; 5% critical |t|=2.776.",""]

L+=["## Table A — Z-fix effect on the solver (buggy -> fixed, identical model & instances)",""]
for metric,field in [("In-sample","ins"),("Out-of-sample","oos")]:
    L+=[f"### {metric}",
        "| Scale | Cost | FCP buggy->fixed (ΔdΔ%) | t | p | BSP buggy->fixed (ΔdΔ%) | t | p |",
        "|---|---|---|---|---|---|---|---|"]
    for sc in SCALES:
        for co in COSTS:
            row=f"| {sc} | {co} |"
            for meth in ("FCP","BSP"):
                b=cell(meth,"buggy",sc,co,field); f=cell(meth,"fixed",sc,co,field)
                seeds=sorted(set(b)&set(f))
                bv=[b[s] for s in seeds]; fv=[f[s] for s in seeds]
                d=[x-y for x,y in zip(fv,bv)]
                md,t,p=paired(d)
                mb_,mf=mean(bv),mean(fv)
                rel = 100*md/mb_ if abs(mb_)>1e-9 else 0.0
                sig=" *" if (not math.isnan(t) and abs(t)>2.776) else ""
                tt = f"{t:+.2f}{sig}" if not math.isnan(t) else "0var"
                row+=f" {mb_:.3f}->{mf:.3f} ({md:+.3f}, {rel:+.2f}%) | {tt} | {p:.3f} |"
            L.append(row)
    L.append("")

L+=["## Table B — corrected (fixed) numbers vs published Table 5","",
    "| Scale | Cost | Method | InS pub->fixed | OOS pub->fixed | (CPBSD-A reused) |",
    "|---|---|---|---|---|---|"]
for sc in SCALES:
    for co in COSTS:
        pub={m:(ins,oos,t) for (m,ins,oos,t) in PUB[sc][co]}
        for meth in ("FCP","BSP"):
            fi=cell(meth,"fixed",sc,co,"ins"); fo=cell(meth,"fixed",sc,co,"oos")
            seeds=sorted(set(fi)&set(fo))
            mfi=mean([fi[s] for s in seeds]); mfo=mean([fo[s] for s in seeds])
            pi,po,_=pub[meth]
            ca=pub["CPBSD-A"]
            L.append(f"| {sc} | {co} | {meth} | {pi:.3f}->{mfi:.3f} | {po:.3f}->{mfo:.3f} | {ca[0]:.3f}/{ca[1]:.3f} |")
L.append("")

text="\n".join(L)
OUT.write_text(text, encoding="utf-8")
print(text)
