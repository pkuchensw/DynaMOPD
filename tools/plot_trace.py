"""Plot measured per-group execution intervals for one round of a completed run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = dict(decode="#276fbf",score="#198754",train="#a449aa",commit="#513f85",
              unload="#a7adb5",load="#e99b34",move="#e35442",
              tp_expand="#ad761f",tp_shrink="#886334",wide_decode="#064588")


def plot(folder, version, dest):
    rows = [json.loads(s) for s in (folder/"events.jsonl").read_text().splitlines()]
    cfg = next(x for x in rows if x["event"]=="config")
    timed = [x for x in rows if x.get("version")==version]
    zero=next(x["t"] for x in timed if x["event"]=="round_start")
    end=next(x["t"] for x in timed if x["event"]=="round_end")
    spans = [x for x in timed if x["event"]=="complete"]
    width=cfg["config"].get("group_size",1)
    count=cfg["config"]["groups"]
    visible=(cfg.get("visible_devices") or ",".join(map(str,range(count*width)))).split(",")
    phases=[("Decode, scoring and backward",zero,next(x["start"] for x in spans if x["op"]=="commit")),
            ("Full synchronous round",zero,end)]
    fig, axes=plt.subplots(2,1,figsize=(12,5.8),layout="constrained")
    for ax,(name,lo,hi) in zip(axes,phases,strict=True):
        def span(g, start, stop, kind):
            if stop>lo and start<hi and stop>start:
                ax.broken_barh([(start-zero,stop-start)],(g-.32,.64),facecolors=COLORS[kind],linewidth=0)
        for x in spans:
            g, op = x["gpu"],x["op"]
            if op=="switch":
                cut=min(x["end"],x["start"]+x["unload_s"])
                span(g,x["start"],cut,"unload")
                span(g,cut,x["end"],"load")
            else:
                kind = "wide_decode" if op=="decode" and x.get("peer") is not None else op
                span(g,x["start"],x["end"],kind)
                if x.get("peer") is not None:
                    span(x["peer"],x["start"],x["end"],kind)
        ax.set(xlim=(lo-zero,hi-zero),yticks=list(range(count)),
               yticklabels=[",".join(visible[i*width:(i+1)*width]) for i in range(count)],
               ylabel="Physical GPUs",xlabel="Seconds since round start",title=name)
        ax.invert_yaxis()
        ax.grid(axis="x",alpha=.18)
    has_tp = any(x["op"]=="tp_expand" for x in spans)
    legend = [Patch(color=v,label=k) for k,v in COLORS.items()
              if has_tp or k not in ("tp_expand", "tp_shrink", "wide_decode")]
    axes[0].legend(handles=legend, loc="upper center",bbox_to_anchor=(.5,1.38),
                   ncol=min(8,len(legend)),frameon=False)
    fig.savefig(dest,dpi=150)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    p.add_argument("--version",type=int,default=0)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    plot(args.folder,args.version,args.output)
