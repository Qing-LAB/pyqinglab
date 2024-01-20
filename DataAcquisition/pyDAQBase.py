"""
This is a prototype class to define necessary functions that all data acquisition classes should have.
"""
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Lock
import numpy
import uuid

class pyDAQTask:    
    def __init__(self) -> None:
        self.mem_manager = SharedMemoryManager()
        self.mem_manager.start()
    
    def __del__(self):
        self.mem_manager.shutdown()
        
    def ConfigureTask(self):
        pass


class DAQBaseClass:
    _DAQClass_init_state = False

    def __init__(self):
        _error_code = 0
        _error_msg = ""
        pass

    def __del__(self):
        pass

    def ConfigBoard(self, board_params: dict) -> int:
        return 0

    def GetBoardInfo(self) -> dict:
        return {}

    def InitBoard(self) -> int:
        return 0

    def ShutdownBoard(self) -> int:
        return 0

    def SingleRead(self, param: dict) -> dict:
        return None

    def SingleWrite(self, param: dict) -> None:
        return None

    def ConfigTask(self, task_params: dict, task_manager: pyDAQTask) -> int:
        return 0

    def InitTask(self, task_manager: pyDAQTask) -> int:
        return 0

    def StartTask(self, task_manager: pyDAQTask) -> int:
        return 0

    def StopTask(self, task_manager: pyDAQTask) -> int:
        return 0

    def GetErrorCode(self) -> int:
        return 0

    def GetErrorMsg(self) -> str:
        return ""

if __name__ == '__main__':
    task = pyDAQTask()
    print("OK")
