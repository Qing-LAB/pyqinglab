"""
This is a prototype class to define necessary functions that all data acquisition classes should have.
"""


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

    def ConfigTask(self, task_params: dict) -> int:
        return 0

    def InitTask(self) -> int:
        return 0

    def StartTask(self) -> int:
        return 0

    def StopTask(self) -> int:
        return 0

    def GetErrorCode(self) -> int:
        return 0

    def GetErrorMsg(self) -> str:
        return ""
