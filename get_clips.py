import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import json
import time

root=Path(__file__).parent
sessions=('w00','w01','w02','w03','w04','w05')
for _ in range(30):
    if all((root/'data'/f'{s}_labels.csv').exists() for s in sessions):
        break
    time.sleep(1)
jobs=[]
for session in sessions:
    with (root/'data'/f'{session}_labels.csv').open() as f:
        for start,end,count,exercise in csv.reader(f):
            if exercise in ('squats','pushups'):
                jobs.append((session,exercise,int(start),int(end),int(count)))
with (root/'manifest.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(('session','exercise','start_frame','end_frame','reference','split','clip'))
    for s,e,a,b,y in jobs:
        w.writerow((s,e,a,b,y,'development' if s in sessions[:2] else 'evaluation',f'{s}_{e}_{a}.mp4'))

def fetch(job):
    s,e,a,b,_=job
    url=f'https://zenodo.org/records/7672767/files/{s}_rgb.mp4?download=1'
    out=root/'clips'/f'{s}_{e}_{a}.mp4'
    root.joinpath('clips').mkdir(exist_ok=True)
    def usable(path):
        if not path.exists() or path.stat().st_size<4000:
            return False
        check=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
                              '-show_entries','stream=nb_frames','-of','json',str(path)],capture_output=True,text=True)
        try:
            n=int(json.loads(check.stdout)['streams'][0]['nb_frames'])
            return abs(n-(b-a+1)/3)<2
        except (ValueError,KeyError,IndexError):
            return False
    if usable(out):
        return f'existing {out.name}' 
    tmp=out.with_suffix('.part.mp4')
    for attempt,read_seconds in enumerate((20,25,35)):
        cmd=['ffmpeg','-nostdin','-hide_banner','-v','error','-rw_timeout',str(read_seconds*1000000),
             '-reconnect','1','-reconnect_on_network_error','1','-reconnect_delay_max','2',
             '-ss',f'{a/30:.6f}','-i',f'{url}&clip={a}&attempt={attempt}','-t',f'{(b-a+1)/30:.6f}','-an',
             '-vf','fps=10,scale=640:-2','-c:v','libx264','-preset','ultrafast','-crf','23','-threads','1',
             '-y',str(tmp)]
        try:
            call=subprocess.run(cmd,capture_output=True,text=True,timeout=65)
            if call.returncode==0 and usable(tmp):
                tmp.replace(out)
                return f'ok {out.name} {out.stat().st_size}'
            failure=call.stderr.strip()[-350:]
        except subprocess.TimeoutExpired:
            failure='process timeout'
    raise RuntimeError(f'{out.name}: {failure}')
with ThreadPoolExecutor(max_workers=7) as pool:
    for future in pool.map(fetch,jobs):
        print(future,flush=True)
