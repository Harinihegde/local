#!/usr/bin/env python3
"""Reproducible Crowd Panic Dataset pipeline; test labels are not used to tune."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

CLASSES = ("No Panic", "Normal", "Crowdy", "Panic")
BASE = ["people_mean", "people_std", "people_p90", "people_trend", "flow_mean", "flow_std", "flow_p90", "coherence_mean", "coherence_std", "speed_mean", "speed_std"]
MODEL_FEATURES = BASE + ["flow_trend", "coherence_p90", "coherence_trend", "speed_p90", "speed_trend"]

def inventory(dataset):
    rows=[]
    for label in CLASSES:
        for path in sorted((dataset/label).glob("*.mp4")):
            cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS); frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            rows.append(dict(path=str(path.resolve()),label=label,fps=fps,frames=int(frames),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),duration_seconds=frames/fps if fps else np.nan)); cap.release()
    if not rows: raise FileNotFoundError(f"No MP4 files beneath {dataset}")
    return pd.DataFrame(rows)

def summarize(x, name):
    x=np.asarray(x,dtype=float)
    if not len(x): x=np.zeros(1)
    return {f"{name}_mean":float(x.mean()),f"{name}_std":float(x.std()),f"{name}_p90":float(np.percentile(x,90)),f"{name}_trend":float(np.polyfit(np.arange(len(x)),x,1)[0]) if len(x)>1 else 0.}

class Tracker:
    def __init__(self, max_dist): self.max_dist=max_dist; self.points={}; self.next_id=0
    def update(self, boxes):
        next_points={}; used=set()
        for box in boxes:
            pt=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2]); candidates=[(np.linalg.norm(pt-v),k) for k,v in self.points.items() if k not in used]
            ident=min(candidates)[1] if candidates and min(candidates)[0]<=self.max_dist else self.next_id
            if ident==self.next_id: self.next_id+=1
            used.add(ident); next_points[ident]=pt
        previous=self.points; self.points=next_points
        return previous, next_points

def clip_features(path, detector, n=48):
    cap=cv2.VideoCapture(path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 1; width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    targets=set(np.linspace(0,max(total-1,0),min(n,total),dtype=int)); tracker=Tracker(max(width,height)*.12); prior_gray=None; counts=[]; mags=[]; coherences=[]; speeds=[]; i=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i not in targets: i+=1; continue
        boxes=detector(frame); old,new=tracker.update(boxes); counts.append(len(boxes))
        for ident,pt in new.items():
            if ident in old: speeds.append(float(np.linalg.norm(pt-old[ident])*fps/max(total/max(n,1),1)))
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if prior_gray is not None:
            flow=cv2.calcOpticalFlowFarneback(prior_gray,gray,None,.5,3,15,3,5,1.2,0); mag,ang=cv2.cartToPolar(flow[...,0],flow[...,1]); mags.append(float(mag.mean())); coherences.append(float(np.hypot(np.cos(ang).mean(),np.sin(ang).mean())))
        prior_gray=gray; i+=1
    cap.release(); out={"sampled_frames":len(counts),"fps":fps,"width":width,"height":height}; out.update(summarize(counts,"people")); out.update(summarize(mags,"flow")); out.update(summarize(coherences,"coherence")); out.update(summarize(speeds,"speed")); return out

def make_detector(weights, imgsz=640):
    if not weights: raise ValueError("--weights is required: the program will not silently download a detector")
    from ultralytics import YOLO
    model=YOLO(weights)
    def detect(frame):
        r=model(frame,classes=[0],conf=.25,imgsz=imgsz,verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0,4))
    return detect

def enrich(df, normal, option):
    x=df.copy(); med=normal[BASE].median(); mad=(normal[BASE]-med).abs().median().replace(0,1e-6)
    z=(x[BASE]-med).abs()/mad
    # 16/32 sample alternatives retain within-clip variability; cross_clip is a learned normal prior.
    if option != "cross_clip": z=z.div(x[["people_std","flow_std","coherence_std","speed_std"]].abs().mean(axis=1).replace(0,1e-6),axis=0)
    x["anomaly_score"]=z.mean(axis=1); x["anomaly_peak"]=z.max(axis=1); return x

def fit_predict(train, test):
    cols=MODEL_FEATURES+["anomaly_score","anomaly_peak"]
    model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features=.8,class_weight="balanced",random_state=20260714,n_jobs=-1)
    model.fit(train[cols].fillna(0),train.label); return model.predict(test[cols].fillna(0))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",type=Path,required=True); ap.add_argument("--output",type=Path,default=Path("outputs")); ap.add_argument("--weights"); ap.add_argument("--sample-frames",type=int,default=48); ap.add_argument("--imgsz",type=int,default=640); ap.add_argument("--inventory-only",action="store_true"); ap.add_argument("--reuse-features",action="store_true"); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    manifest_path=a.output/"split_manifest.csv"
    if manifest_path.exists(): manifest=pd.read_csv(manifest_path)
    else:
        manifest=inventory(a.dataset); ti,hi=train_test_split(manifest.index,test_size=.2,stratify=manifest.label,random_state=20260714); manifest["split"]="train"; manifest.loc[hi,"split"]="test"; manifest.to_csv(manifest_path,index=False)
    manifest.groupby("label")[["fps","frames","width","height","duration_seconds"]].agg(["count","min","median","max"]).to_csv(a.output/"video_properties_by_class.csv")
    if a.inventory_only: return
    fp=a.output/"clip_features.csv"
    if a.reuse_features and fp.exists(): features=pd.read_csv(fp)
    else:
        detector=make_detector(a.weights,a.imgsz); rows=[]
        for j,row in manifest.reset_index(drop=True).iterrows(): rows.append(dict(path=row.path,label=row.label,split=row.split,**clip_features(row.path,detector,a.sample_frames))); print(f"features: {j+1}/{len(manifest)}",flush=True)
        features=pd.DataFrame(rows); features.to_csv(fp,index=False)
    train=features[features.split=="train"].reset_index(drop=True); test=features[features.split=="test"].reset_index(drop=True); inner,valid=train_test_split(train,test_size=.2,stratify=train.label,random_state=20260714)
    candidates=[]
    for option in (16,32,"cross_clip"):
        normal=inner[inner.label.isin(["No Panic","Normal"])]; p=fit_predict(enrich(inner,normal,option),enrich(valid,normal,option)); candidates.append(dict(option=str(option),validation_accuracy=accuracy_score(valid.label,p),validation_macro_f1=f1_score(valid.label,p,average="macro")))
    selected=max(candidates,key=lambda r:r["validation_macro_f1"])["option"]; selected=int(selected) if selected.isdigit() else selected; normal=train[train.label.isin(["No Panic","Normal"])]; p=fit_predict(enrich(train,normal,selected),enrich(test,normal,selected))
    result={"protocol":"single 20% stratified held-out test; option selected only by inner training validation", "candidate_results":candidates,"chosen_option":str(selected),"final_test_accuracy":accuracy_score(test.label,p),"per_class":classification_report(test.label,p,output_dict=True,zero_division=0)}
    (a.output/"results.json").write_text(json.dumps(result,indent=2)); pd.DataFrame(dict(path=test.path,actual=test.label,predicted=p)).to_csv(a.output/"test_predictions.csv",index=False); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
