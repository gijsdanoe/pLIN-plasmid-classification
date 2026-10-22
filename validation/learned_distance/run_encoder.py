#!/usr/bin/env python3
"""5-fold CV: pLIN kNN on raw tetra space vs kNN on a SupCon-learned metric.
Only the space changes; k=5 cosine distance-weighted in both, as in pLIN.
Reports per-class F1 and out-of-sample neighbour purity for both spaces."""
import sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

torch.set_num_threads(2)
SEED=0; K=5
CFGS={
 "default": dict(temperature=0.07, emb_dim=128, epochs=200, per_class=4,
                 hidden=(512,256), dropout=0.2, lr=1e-3),
 "swept":   dict(temperature=0.2,  emb_dim=256, epochs=600, per_class=4,
                 hidden=(512,256), dropout=0.2, lr=3e-4),
}
class Enc(nn.Module):
    def __init__(s,hidden,emb,do,ind=256):
        super().__init__(); L=[];p=ind
        for h in hidden:
            L+=[nn.Linear(p,h),nn.BatchNorm1d(h),nn.ReLU()]
            if do>0: L.append(nn.Dropout(do))
            p=h
        L.append(nn.Linear(p,emb)); s.net=nn.Sequential(*L)
    def forward(s,x): return F.normalize(s.net(x),dim=1)

def supcon(e,l,t):
    sim=e@e.T/t; n=e.size(0)
    eye=torch.eye(n,dtype=torch.bool); sim=sim.masked_fill(eye,-1e9)
    pos=(l[:,None]==l[None,:])&~eye
    lp=sim-torch.logsumexp(sim,1,keepdim=True)
    npos=pos.sum(1); v=npos>0
    if v.sum()==0: return torch.tensor(0.0,requires_grad=True)
    return -((lp*pos).sum(1)[v]/npos[v]).mean()

def train(Xtr,ytr_idx,n_classes,cfg,seed=SEED):
    torch.manual_seed(seed)
    enc=Enc(cfg["hidden"],cfg["emb_dim"],cfg["dropout"])
    opt=torch.optim.AdamW(enc.parameters(),lr=cfg["lr"],weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["epochs"])
    Xt=torch.tensor(Xtr); yt=torch.tensor(ytr_idx)
    rng=np.random.default_rng(seed)
    pools=[p for p in (np.where(ytr_idx==c)[0] for c in range(n_classes)) if len(p)>0]
    pc=cfg["per_class"]; nb=max(1,len(Xtr)//(len(pools)*pc))
    enc.train()
    for _ in range(cfg["epochs"]):
        for _ in range(nb):
            b=torch.tensor(np.concatenate([rng.choice(p,pc,replace=len(p)<pc) for p in pools]))
            loss=supcon(enc(Xt[b]),yt[b],cfg["temperature"])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    enc.eval(); return enc

def embed(enc,Xa):
    with torch.no_grad(): return enc(torch.tensor(Xa)).numpy()

def oos_purity_nbrs(Ztr,ytr,Zte,k=K):
    _,idx=NearestNeighbors(n_neighbors=k,metric="cosine").fit(Ztr).kneighbors(Zte)
    return ytr[idx]

def main(which):
    cfg=CFGS[which]
    d=np.load('tetra_matrix.npz',allow_pickle=True)
    X=d['X'].astype(np.float32); y=d['y'].astype(str)
    classes=sorted(set(y)); c2i={c:i for i,c in enumerate(classes)}
    yi=np.array([c2i[c] for c in y])
    print(f"config={which}  {cfg}",flush=True)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    pr=np.empty(len(y),dtype=object); pe=np.empty(len(y),dtype=object)
    nr=np.empty((len(y),K),dtype=object); ne=np.empty((len(y),K),dtype=object)
    kw=dict(n_neighbors=K,metric="cosine",weights="distance")
    for f,(tr,te) in enumerate(cv.split(X,y),1):
        t0=time.time()
        pr[te]=KNeighborsClassifier(**kw).fit(X[tr],y[tr]).predict(X[te])
        nr[te]=oos_purity_nbrs(X[tr],y[tr],X[te])
        enc=train(X[tr],yi[tr],len(classes),cfg)
        Ztr,Zte=embed(enc,X[tr]),embed(enc,X[te])
        pe[te]=KNeighborsClassifier(**kw).fit(Ztr,y[tr]).predict(Zte)
        ne[te]=oos_purity_nbrs(Ztr,y[tr],Zte)
        print(f"  fold {f}/5 done in {time.time()-t0:.0f}s",flush=True)
    pr=pr.astype(str); pe=pe.astype(str); nr=nr.astype(str); ne=ne.astype(str)
    out={"config":which,"cfg":{k:list(v) if isinstance(v,tuple) else v for k,v in cfg.items()}}
    for nm,p in [("raw",pr),("enc",pe)]:
        out[nm]={"acc":accuracy_score(y,p),"macroF1":f1_score(y,p,average="macro"),
                 "weightedF1":f1_score(y,p,average="weighted"),
                 "balacc":balanced_accuracy_score(y,p)}
        print(f"\n=== {nm} ===  acc {out[nm]['acc']:.3f}  macroF1 {out[nm]['macroF1']:.3f}  "
              f"wF1 {out[nm]['weightedF1']:.3f}  balacc {out[nm]['balacc']:.3f}",flush=True)
    rows=[]
    for c in classes:
        m=y==c
        rows.append(dict(cls=c,n=int(m.sum()),
            f1_raw=f1_score(m,pr==c),f1_enc=f1_score(m,pe==c),
            pur_raw=float((nr[m]==c).mean()),pur_enc=float((ne[m]==c).mean()),
            nbr_enc=dict(Counter(ne[m].ravel()).most_common(3))))
    out["per_class"]=rows
    print(f"\n{'class':>14s} {'n':>6s} {'F1raw':>7s} {'F1enc':>7s} {'dF1':>7s} {'purR':>6s} {'purE':>6s}")
    for r in sorted(rows,key=lambda r:r["pur_raw"]):
        print(f"{r['cls']:>14s} {r['n']:6d} {r['f1_raw']:7.3f} {r['f1_enc']:7.3f} "
              f"{r['f1_enc']-r['f1_raw']:+7.3f} {r['pur_raw']:6.2f} {r['pur_enc']:6.2f}")
    json.dump(out,open(f"results_{which}.json","w"),indent=1)
    print(f"\nwrote results_{which}.json")

if __name__=="__main__": main(sys.argv[1])
