import h5py
import numpy as np
from typing import Union
from rich import print
from rich.console import Console
import datetime
import matplotlib

from PIL import Image

class Datafile:
    
    def __init__(self, fname: str, operator: str ='unknown', new_groupname=''):
        self.fname=fname
        self.new_groupname=new_groupname
        self.console=Console()
        with h5py.File(self.fname, "a") as f:
            try:
                keys=f.keys()
                if '/pyqlab_log' in keys:
                    grp = f['/pyqlab_log']
                    if not isinstance(grp, h5py.Group):
                        raise Exception("the name pyqlab_log is already occupied and cannot be opened as group for logging.")
                else:
                    self.console.print('pyqlab_log not exist, will be created now')
                    grp = f.create_group('/pyqlab_log')
                ct = datetime.datetime.now()
                dataset_name = f"[{ct}] Log INIT"
                dt = h5py.string_dtype(encoding='utf-8')
                data = np.array(f"Initiated by operator: [{operator}]").astype(dt)
                
                logmsg = grp.create_dataset(dataset_name, data=data)#, dtype=dt)
            except KeyError:
                self.console.print_exception(max_frames=20)
            except Exception:
                self.console.print_exception(max_frames=20)
            
    def savefig_as_array(self, fig: matplotlib.figure.Figure, dpi: int=300) -> np.array:
        fig.tight_layout(pad=0)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1]+(3,))        
        return data
    
    def save_image(self, key: str, data: np.array):
        try:
            with h5py.File(self.fname, "a") as f:
                f[key] = data
                f[key].attrs = "image"
    
    def save_data(self, data: dict, note: str):
        None