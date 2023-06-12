import h5py
import numpy as np
from typing import Union
from rich.console import Console
import datetime
from matplotlib.figure import Figure
from PIL import Image
from pathlib import Path
from pathlib import PurePath

class Datafile:
    def __init__(self, fname: str, operator: str = 'unknown', root_group='/', note: str = ''):
        self.fname = fname
        self.root_group = root_group
        self.console = Console()

        ct = Datafile.get_timestamp_now()
        dataset_name = f"LogINIT[{ct}]"
        dt = h5py.string_dtype(encoding="utf-8")
        data = np.array(f"Initiated by operator: [{operator}]. Note: [{note}]").astype(dt)
        self.save_as_dataset(data, dataset_name, '/pyqlab_log', '', overwrite=True)

    @staticmethod
    def fig2nparray(fig: Figure, dpi: int = 300) -> np.array:
        """
        convert matplotlib figure into an image object saved as numpy array
        fig: the handle of the figure
        dpi: the dpi of the image, default is 300 DPI
        return: numpy array of dtype uint8, containing all the data of the image
        """
        fig.tight_layout(pad=0)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    @staticmethod
    def generate_attr_str(attrs: dict, sep: str = "\n", eq: str = ":") -> str:
        """
        generate a string that contains all attributes with the format key : value \\n
        where the ':' and the \\n are the equal sign and the separator, respectively, 
        which can be changed by the parameters

        attrs: dictionary containing pairs of name and property as attributes for saving
        sep: string as separator between each pair of attribute
        eq: string as equal sign between the key name and the property
        return: string containing all attributes as formatted
        """
        str_list = []
        for k, v in attrs.items():
            str_list.append(f"{k}{eq}{v}")
        return sep.join(str_list)

    @staticmethod
    def get_timestamp_now() -> str:
        ct = datetime.datetime.now()
        ct = ct.replace(":", "_").replace("-", "_").replace(" ", "-")
        return ct

    def create_group(self, groupname: str, parentkey: str, note:str) -> bool:
        success = False
        p = PurePath(self.root_group, parentkey, groupname)
        parent_key = p.parent
        grp_key = p.name

        with h5py.File(self.fname, 'a') as f:
            try:
                try:
                    parent = f[parent_key]
                except KeyError:
                    parent = f.create_group(parent_key)
            
                if isinstance(parent, h5py.Datagroup):
                    grp = parent.create_group(grp_key)
                    ct = Datafile.get_timestamp_now()
                    attr_str = Datafile.generate_attr_str(
                        {
                            "timestamp": ct,
                            "note": note,
                        }
                    )
                    success = True
            except ValueError:
                self.console.print(f"group {p} cannot be created...")
                self.console.print_exception(max_frames=20)
            except Exception:
                    self.console.print_exception(max_frames=20)

        return success

    def save_as_dataset(
        self, data: np.array, name: str, groupkey: str, attr_str: str, overwrite=True
    ) -> bool:
        success = False
        p = PurePath(groupkey, name)
        grp_key = p.parent
        n_key = p.name
        
        if self.create_group(grp_key, '/', ''):
            with h5py.File(self.fname, "a") as f:
                grp_key = PurePath(self.root_group, grp_key)
                grp = f[grp_key]
                if not isinstance(grp, h5py.Group):
                    raise Exception(
                        "Name {grp_key} already occupied. Failed to create a data group with that name"
                    )
                if (n_key in grp.keys()) and isinstance(grp[n_key], h5py.Dataset):
                    if overwrite:
                        del grp[n_key]
                if n_key in grp.keys():
                    raise Exception(
                        "Cannot overwrite key {n_key} in group {grp_key} as dataset"
                    )
                dset = grp.create_dataset(n_key, data=data)
                dt = h5py.string_dtype(encoding="utf-8")
                dset.attrs.create("created_by_pyqlab", np.array(attr_str).astype(dt))
                success = True
        
        return success

    def save_image(
        self,
        name: str,
        groupkey: str,
        data: np.array,
        dimension: tuple,
        color_channels: str,
        note: str,
        overwrite=True,
    ) -> bool:
        try:
            if data.shape != dimension:
                print(
                    "Warning: dimension provided {str(dimension)} is not consistent with image data shape {str(data.shape)}"
                )
            ct = Datafile.get_timestamp_now()
            attr_str = Datafile.generate_attr_str(
                {
                    "type": "image",
                    "dimensions": str(dimension),
                    "color_channels": color_channels,
                    "timestamp": ct,
                    "note": note,
                }
            )
            return self.save_as_dataset(
                data, name, groupkey, attr_str, overwrite=overwrite
            )
        except Exception:
            self.console.print_exception(max_frames=20)

    def save_string(
        self,
        name: str,
        groupkey: str,
        data: str,
        note: str,
        encode="utf-8",
        overwrite=True,
    ) -> bool:
        dt = h5py.string_dtype(encode=encode)
        ct = self.get_timestamp_now()
        d = np.fromstring(data).astype(dt)
        attr_str = Datafile.generate_attr_str(
            {
                "type": "str",
                "timestamp": ct,
                "note": note,
            }
        )
        self.save_as_dataset(d, name, groupkey, attr_str, overwrite=overwrite)

    def save_variable(self, name: str, groupkey: str, data: np.double, note: str, typestr="double"):
        ct = Datafile.get_timestamp_now()
        attr_str = Datafile.generate_attr_str(
            {
                "typestr": typestr,
                "timestamp": ct,
                "note": note,
            }
        )
        self.save_as_dataset(data, name, groupkey, attr_str)

    def save_nparray(self, name: str, groupkey: str, data: np.array, note: str):
        ct = Datafile.get_timestamp_now()
        attr_str = Datafile.generate_attr_str(
            {
                "timestamp": ct,
                "note": note,
            }
        )
        self.save_as_dataset(data, name, groupkey, attr_str)

    # def save_hierarchy_data(self, data: dict, groupkey: str):
    #     for k, v in enumerate(data):
    #         if isinstance(v, dict):
    #             self.save_hierarchy_data(v, grp_key)
    #         else:
    #             match type(v):
    #                 case int:
    #                     None
    #                 case double:
    #                     None
    #                 case str:
    #                     None
    #                 case np.array:
    #                     None
