import h5py
import numpy as np
from typing import Union
from rich.console import Console
import datetime
from matplotlib.figure import Figure
from PIL import Image


class Datafile:
    def __init__(self, fname: str, operator: str = "unknown", new_groupname=""):
        self.fname = fname
        self.new_groupname = new_groupname
        self.console = Console()
        with h5py.File(self.fname, "a") as f:
            try:
                keys = f.keys()
                if "/pyqlab_log" in keys:
                    grp = f["/pyqlab_log"]
                    if not isinstance(grp, h5py.Group):
                        raise Exception(
                            "the name pyqlab_log is already occupied and cannot be opened as group for logging."
                        )
                else:
                    self.console.print("pyqlab_log not exist, will be created now")
                    grp = f.create_group("/pyqlab_log")
                ct = datetime.datetime.now()
                dataset_name = f"[{ct}] Log INIT"
                dt = h5py.string_dtype(encoding="utf-8")
                data = np.array(f"Initiated by operator: [{operator}]").astype(dt)

                logmsg = grp.create_dataset(dataset_name, data=data)  # , dtype=dt)
            except KeyError:
                self.console.print_exception(max_frames=20)
            except Exception:
                self.console.print_exception(max_frames=20)

    def savefig_as_array(self, fig: Figure, dpi: int = 300) -> np.array:
        fig.tight_layout(pad=0)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def generate_attr_str(self, attrs: dict, sep: str = "\n", eq: str = ":") -> str:
        str_list = []
        for k, v in attrs.items():
            str_list.append(f"{k}{eq}{v}")
        return sep.join(str_list)

    def prep_dataset_key(self, name: str, groupkey: str) -> h5py.Dataset:
        None

    def save_image(
        self,
        name: str,
        groupkey: str,
        data: np.array,
        dimension: tuple,
        color_channels: str,
        note: str,
    ):
        try:
            if data.shape != dimension:
                print(
                    "Warning: dimension provided {str(dimension)} is not consistent with image data shape {str(data.shape)}"
                )
            with h5py.File(self.fname, "a") as f:
                if not (groupkey in f.keys()):
                    f.create_group(groupkey)
                grp = f[groupkey]
                if not isinstance(grp, h5py.Group):
                    raise Exception(
                        "Name of group {groupkey} already occupied. Failed to create the group with that name"
                    )
                if (name in grp.keys()) and isinstance(grp[name], h5py.Dataset):
                    del grp[name]
                if name in grp.keys():
                    raise Exception(
                        "Cannot overwrite key {name} in group {groupkey} as dataset"
                    )
                dset = grp.create_dataset(name, data=data)
                dt = h5py.string_dtype(encoding="utf-8")
                attr_str = np.array(
                    self.generate_attr_str(
                        {
                            "type": "image",
                            "dimensions": str(dimension),
                            "color_channels": color_channels,
                            "note": note,
                        }
                    )
                ).astype(dt)
                dset.attrs.create("created_by_pyqlab", attr_str)
        except Exception:
            self.console.print_exception(max_frames=20)

    def save_string(self, name: str, groupkey: str, data: str, note: str):
        dt = h5py.string_dtype(encode="byte")

    def save_variable(self, name: str, groupkey: str, data: np.double, note: str):
        None

    def save_nparray(self, name: str, groupkey: str, data: np.array, note: str):
        None

    def save_data(self, data: dict, note: str):
        None
