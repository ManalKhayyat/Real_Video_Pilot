"""Replay counting and feature-cue policies from real-video landmark streams."""
from collections import deque
import csv
import json
from itertools import product
from pathlib import Path
import warnings

import numpy as np

ROOT = Path(__file__).parent
RATE = 10
QUALITY = 0.55
MODELS = ('five_vote_proxy', 'score_only', 'strict_phase', 'gap_phase')
CONDITIONS = ('clean', 'cue_landmark_dropout', 'phase_pose_dropout')
POINTS = {
    'squats': ((23,25,27),(24,26,28)),
    'pushups': ((11,13,15),(12,14,16)),
}

def angle(a,b,c):
    u=a-b
    v=c-b
    norm=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
    dot=(u*v).sum(axis=-1)
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.degrees(np.arccos(np.clip(dot/norm,-1,1)))

def process(data, exercise):
    # Pick one side from the first second and keep it fixed for the clip.
    choices=POINTS[exercise]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        strengths=[np.nanmean(np.min(data[:10,list(ids),2],axis=1)) for ids in choices]
    side=int(np.argmax(np.nan_to_num(strengths,nan=-1)))
    ids=list(choices[side])
    a,b,c=(data[:,i,:2] for i in ids)
    phase_angle=angle(a,b,c)
    phase_q=np.min(data[:,ids,2],axis=1)
    shoulder,hip,ankle=(11+side,23+side,27+side)
    if exercise=='squats':
        vector=data[:,shoulder,:2]-data[:,hip,:2]
        vertical=np.tile([0,-1],(len(data),1))
        zero=np.zeros_like(vertical)
        cue_angle=angle(vector,zero,vertical)
        cue_geometric=cue_angle>=35
        cue_ids=[shoulder,hip]
    else:
        cue_angle=180-angle(data[:,shoulder,:2],data[:,hip,:2],data[:,ankle,:2])
        cue_geometric=cue_angle>=22
        cue_ids=[shoulder,hip,ankle]
    cue_q=np.min(data[:,cue_ids,2],axis=1)
    return phase_angle,phase_q,cue_geometric,cue_q

def predict(features,bottom,top,model,condition):
    phase_angle,phase_q,cue_geometry,cue_q=features
    gated=model!='five_vote_proxy'
    ordered=model in ('strict_phase','gap_phase')
    threshold_mid=(bottom+top)/2
    max_gap=4 if model=='gap_phase' else 0
    history=deque(maxlen=5)
    phase=None
    run_top=run_bottom=run_cue=0
    gap=0
    count=accepted=cues=cue_in_mask=0
    emitted=False
    for i,(theta,q,cue,cq) in enumerate(zip(phase_angle,phase_q,cue_geometry,cue_q)):
        dropout=i>=10 and (i-10)%20<4
        valid=np.isfinite(theta) and not (dropout and condition=='phase_pose_dropout') and (not gated or q>=QUALITY)
        if not valid:
            gap+=1
            run_top=run_bottom=run_cue=0
            if ordered and gap>max_gap:
                phase=None
                emitted=False
            continue
        accepted+=1
        gap=0
        if ordered:
            run_top=run_top+1 if theta>=top else 0
            run_bottom=run_bottom+1 if theta<=bottom else 0
            if run_top==2:
                if phase=='bottom':
                    count+=1
                phase='top'
                emitted=False
            if run_bottom==2 and phase=='top':
                phase='bottom'
            is_bottom=phase=='bottom' and theta<=bottom
        else:
            history.append(theta<=threshold_mid)
            if len(history)<5:
                continue
            is_bottom=sum(history)>=3
            if not is_bottom and phase=='bottom':
                count+=1
            phase='bottom' if is_bottom else 'top'
            if not is_bottom:
                emitted=False
        unavailable=dropout and condition=='cue_landmark_dropout'
        if ordered:
            eligible=is_bottom and cue and cq>=QUALITY and not unavailable
            run_cue=run_cue+1 if eligible else 0
            fire=run_cue>=2 and not emitted
        else:
            fire=is_bottom and cue and not emitted
        if fire:
            cues+=1
            cue_in_mask+=unavailable
            emitted=True
    return count,accepted,cues,cue_in_mask

def load(split):
    with (ROOT/'manifest.csv').open() as f:
        rows=[row for row in csv.DictReader(f) if row['split']==split]
    for row in rows:
        stem = Path(row['clip']).stem
        feature_file = ROOT/'features'/f'{stem}.npz'
        if feature_file.exists():
            with np.load(feature_file) as archive:
                row['features'] = tuple(archive[f'arr_{i}'] for i in range(4))
        else:
            path = ROOT/'poses'/f'{stem}.npz'
            with np.load(path) as archive:
                row['features'] = process(archive['xyq'],row['exercise'])
    return rows

def calibrate(rows):
    configurations={}
    for exercise in ('squats','pushups'):
        subset=[r for r in rows if r['exercise']==exercise]
        if not subset:
            raise ValueError(f'No calibration clips for {exercise}')
        prior=(115,150) if exercise=='squats' else (105,150)
        grid=product(range(75,141,5),range(115,171,5))
        scores=[]
        for bottom,top in grid:
            if top-bottom<20:
                continue
            abs_err=sum(abs(predict(r['features'],bottom,top,m,'clean')[0]-int(r['reference']))
                        for r in subset for m in MODELS[:3])
            distance=abs(bottom-prior[0])+abs(top-prior[1])
            scores.append((abs_err,distance,bottom,top))
        _,_,bottom,top=min(scores)
        configurations[exercise]={'bottom':bottom,'top':top}
    return configurations

def evaluate(rows,config):
    for row in rows:
        bottom,top=config[row['exercise']].values()
        for condition,model in product(CONDITIONS,MODELS):
            count,accepted,cues,withheld=predict(row['features'],bottom,top,model,condition)
            yield {k:v for k,v in row.items() if k!='features'} | dict(
                policy=model, condition=condition, count=count, count_error=count-int(row['reference']),
                phase_frames=len(row['features'][0]), phase_accepted=accepted, cue_events=cues,
                cues_in_known_dropout=withheld)

def summarize(rows):
    for condition,exercise,policy in product(CONDITIONS,('all','squats','pushups'),MODELS):
        sub=[r for r in rows if r['condition']==condition and r['policy']==policy and (exercise=='all' or r['exercise']==exercise)]
        actual=sum(int(r['reference']) for r in sub)
        errors=[r['count_error'] for r in sub]
        yield dict(condition=condition,exercise=exercise,policy=policy,n=len(sub),
                   reference_total=actual,predicted_total=sum(r['count'] for r in sub),
                   mae=round(sum(map(abs,errors))/len(sub),4),bias=round(sum(errors)/len(sub),4),
                   exact=sum(e==0 for e in errors),under=sum(-e for e in errors if e<0),
                   over=sum(e for e in errors if e>0), phase_coverage=round(
                       sum(r['phase_accepted'] for r in sub)/sum(r['phase_frames'] for r in sub),5),
                   cues=sum(r['cue_events'] for r in sub),cues_inside_dropout=sum(r['cues_in_known_dropout'] for r in sub))

def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

if __name__=='__main__':
    config=calibrate(load('development'))
    (ROOT/'frozen_thresholds.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=list(evaluate(load('evaluation'),config))
    write_csv(ROOT/'per_clip.csv',rows)
    write_csv(ROOT/'summary.csv',list(summarize(rows)))
