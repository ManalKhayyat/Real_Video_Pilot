import io
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

URL = 'https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip'
class RemoteArchive(io.RawIOBase):
    def __init__(self,url):
        self.url=url
        self.pos=0
        with urlopen(Request(url,headers={'Range':'bytes=0-0'}),timeout=15) as r:
            self.length=int(r.headers['Content-Range'].split('/')[-1])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=(0,self.pos,self.length)[whence]+offset
        return self.pos
    def read(self,size=-1):
        size=min(self.length-self.pos, size if size >=0 else self.length)
        if size<=0: return b''
        with urlopen(Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+size-1}'}),timeout=22) as r:
            expected=f'bytes {self.pos}-{self.pos+size-1}/{self.length}'
            if r.headers.get('Content-Range') != expected:
                raise IOError(f'Unexpected byte range for {self.pos} {size}')
            data=r.read()
        self.pos+=len(data)
        return data

with ZipFile(RemoteArchive(URL)) as z:
    names = [n for n in z.namelist() if n.endswith('.csv')]
    data = Path(__file__).parent/'data'
    data.mkdir(exist_ok=True)
    for n in names:
        if 'label' in n and Path(n).parent.name in {'w00','w01','w02','w03','w04','w05'}:
            (data/Path(n).name).write_bytes(z.read(n))
