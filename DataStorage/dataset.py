from collections import UserDict, namedtuple
from typing import Union

import h5py
import numpy as np
from rich import print
from rich.console import Console
from rich.markup import escape
from rich.tree import Tree

Datanode = namedtuple(
    "Datanode",
    "fname path id type data attrs subgroup_list dataset_list unknown_data_list",
)


class HDFDataset(UserDict):
    """
    - Read from a hdf5 data set, and keep information of the structure
    - Provide a consistent way to read from HDF5, and only read data that is requested by a correct key

    The attributes for data is saved in the first initial read of the file so for attributes for all nodes
    """

    def __init__(self, fname, mode="read_only"):
        """
        will read from hdf5 and update a tree structure to represent how data are stored
        """
        self.console = Console()
        self.fname = fname
        self.dsCount = 0
        self.update_dstree()

    def __walk_ds(self, ds: Union[h5py.File, h5py.Dataset], tree: Tree) -> None:
        paths = sorted(ds.keys())

        for path in paths:
            sub_ds = ds[path]
            deeper_walk = False
            if isinstance(sub_ds, h5py.Group):
                branch = tree.add(
                    f"[bold magenta]:open_file_folder:({self.dsCount}) {escape(path)}"
                )
                branch._nodeType = "group"
                tree._subgroupList.append(self.dsCount)
                branch._datasetList = []
                deeper_walk = True

            elif isinstance(sub_ds, h5py.Dataset):
                branch = tree.add(
                    f"[green]:page_with_curl:({self.dsCount}) {escape(path)}"
                )
                branch._nodeType = "dataset"
                branch._datasetList = [self.dsCount]
                tree._datasetList.append(self.dsCount)

            else:
                branch = tree.add(
                    f"[red]:question_mark:({self.dsCount}) {escape(path)}"
                )
                branch._nodeType = "unknown"
                tree._unknowndataList.append(self.dsCount)

            branch._dsPath = "/".join((tree._dsPath, path))
            branch._parentGroup = tree
            branch._node_id = self.dsCount
            branch._node_attrs = {}
            branch._subgroupList = []
            branch._unknowndataList = []

            self.idTable[branch._node_id] = branch
            self.pathTable[branch._dsPath] = branch._node_id

            self.dsCount += 1

            for attr in sub_ds.attrs:
                branch._node_attrs[attr] = sub_ds.attrs.get(attr)

            if deeper_walk:
                self.__walk_ds(sub_ds, branch)

    def update_dstree(self):
        try:
            with h5py.File(self.fname, "r") as fhandle:
                self.dsTree = Tree(
                    f":open_file_folder: (0) {escape(self.fname)}",
                    guide_style="bold bright_blue",
                )
                self.dsTree._nodeType = "root"
                self.dsTree._dsPath = ""
                self.dsTree._subgroupList = []
                self.dsTree._datasetList = []
                self.dsTree._unknowndataList = []
                self.dsTree._node_attrs = {}
                ds = fhandle["/"]
                for attr in ds.attrs:
                    self.dsTree._node_attrs[attr] = ds.attrs.get(attr)

                self.dsCount = 1
                self.idTable = {0: self.dsTree}
                self.pathTable = {"/": 0}

                self.__walk_ds(fhandle, self.dsTree)
                self.dsTree._dsPath = (
                    "/"  # this is set last to avoid double // for all sub-directories
                )

        except Exception:
            print("Error encountered when opening hdf file for access.")
            print(f"File name: {self.fname}")
            self.console.print_exception(max_frames=20)

    def __getitem__(self, key: Union[int, str]) -> Datanode:
        try:
            if isinstance(key, int):
                id = key
            elif isinstance(key, str):
                id = self.get_id(key)
                if id < 0:
                    raise KeyError
            else:
                id = -1
                raise KeyError

            node = self.idTable[id]
            d = Datanode(
                fname=self.fname,
                path=node._dsPath,
                type=node._nodeType,
                id=id,
                data=self.get_data_by_id(id),
                attrs=node._node_attrs,
                subgroup_list=node._subgroupList,
                dataset_list=node._datasetList,
                unknown_data_list=node._unknowndataList,
            )
        except KeyError:
            self.console.print(
                f"Key '{escape(key)}' cannot be found in the Dataset '{escape(self.fname)}'"
            )
            d = Datanode(
                fname=self.fname,
                path="",
                type="",
                id=-1,
                data=None,
                attrs=None,
                subgroup_list=[],
                dataset_list=[],
                unknown_data_list=[],
            )
        except Exception:
            self.console.print_exception(max_frames=20)
        return d

    def __setitem__(self, key, value):
        raise Exception(
            f"Dataset '{escape(self.fname)}' is read-only. item#'{escape(key)}' will not be changed to value [{value}]"
        )
        return None

    def __repr__(self):
        return f"<HDF5Dataset file:{self.fname}>"

    def get_path(self, id: int) -> str:
        try:
            node = self.idTable[id]
            if node:
                return node._dsPath
            else:
                return ""
        except KeyError:
            return None
        except Exception:
            self.console.print_exception(max_frames=20)

    def get_type(self, id: int) -> str:
        try:
            node = self.idTable[id]
            if node:
                return node._nodeType
            else:
                return ""
        except KeyError:
            return None
        except Exception:
            self.console.print_exception(max_frames=20)

    def get_data_by_id(self, id: int) -> np.array:
        try:
            node = self.idTable[id]
            data = None
            if node and node._nodeType == "dataset":
                with h5py.File(self.fname, "r") as fhandle:
                    dataset = fhandle[node._dsPath]
                    if dataset:
                        data = dataset[()]
        except KeyError:
            print(
                f"Dataset {self.fname} looks for ID {id} but this node cannot be found."
            )
            return None
        except Exception:
            self.console.print_exception(max_frames=20)

        return data

    def get_attrs(self, id: int) -> dict:
        try:
            node = self.idTable[id]
            if node:
                return node._node_attrs
        except KeyError:
            return None
        except Exception:
            self.console.print_exception(max_frames=20)

    def print_tree(self) -> None:
        self.console.print(self.dsTree)

    def get_subgroup_list(self, id: int) -> list:
        try:
            node = self.idTable[id]
        except KeyError:
            return []

        return node._subgroupList

    def get_dataset_list(self, id: int) -> list:
        try:
            node = self.idTable[id]
        except KeyError:
            return []

        return node._datasetList

    def get_unknown_data_list(self, id: int) -> list:
        try:
            node = self.idTable[id]
        except KeyError:
            return []

        return node._unknowndataList

    def get_id(self, dsPath: str) -> int:
        try:
            id = self.pathTable[dsPath]
        except KeyError:
            return -1
        return id
