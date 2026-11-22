from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault('GLOG_minloglevel','2')
import cv2
import mediapipe as mp
import numpy as np

ROOT=Path(__file__).parent

def extract(row):
    path=ROOT/'clips'/row['clip']
    destination=ROOT/'poses'/f'{path.stem}.npz'
    if destination.exists():
        return
    if not path.exists():
        return
    options=mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'data'/'pose_landmarker_lite.task')),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1)
    cap=cv2.VideoCapture(str(path))
    points=[]
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as estimator:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=estimator.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),100*len(points))
            if result.pose_landmarks:
                points.append([[p.x*rgb.shape[1],p.y*rgb.shape[0],min(p.visibility,p.presence)] for p in result.pose_landmarks[0]])
            else:
                points.append(np.full((33,3),np.nan))
    cap.release()
    n_expect=(int(row['end_frame'])-int(row['start_frame'])+1)/3
    if not points or abs(len(points)-n_expect)>2:
        raise RuntimeError((path.name,len(points),n_expect))
    destination.parent.mkdir(exist_ok=True)
    tmp=destination.with_suffix('.part.npz')
    np.savez_compressed(tmp,xyq=np.asarray(points,dtype=np.float32))
    tmp.replace(destination)
    return path.name,len(points)

if __name__=='__main__':
    with (ROOT/'manifest.csv').open() as fp:
        rows=list(csv.DictReader(fp))
    if len(sys.argv)>1:
        rows=[r for r in rows if r['split']==sys.argv[1]]
    with ProcessPoolExecutor(max_workers=4) as p:
        for result in p.map(extract,rows):
            if result:
                print(result,flush=True)
