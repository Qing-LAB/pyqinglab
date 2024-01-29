# This is a prototype class to define necessary functions that all data
# acquisition classes should have.

import logging
import logging.handlers
import os
import sys
import time
import uuid
from enum import Enum
from multiprocessing import Lock, Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import psutil


def listener_configurer(logfilename: str):
    root = logging.getLogger()
    file_handler = logging.handlers.RotatingFileHandler(logfilename, "a", 300, 10)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG)


def listener_process(queue):
    listener_configurer()
    while True:
        while not queue.empty():
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        sleep(1)


def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)


def worker_process(queue):
    worker_configurer(queue)
    for i in range(3):
        sleep(random())
        innerlogger = logging.getLogger("worker")
        innerlogger.info(f"Logging a random number {randint(0, 10)}")

# Forward declaration for class DAQTaskManager
class DAQTaskManager:
    class TaskContext:
        pass

    pass


class DAQBaseClass:
    """The base class for all DAQ handlers.
    This base class will provide a unified
    interface to the outside (mainly the task manager) to interact with instruments.
    This should inlcude the following:
    (1) configuration and initialization of the board/equipment
    (2) shutdown of the board/equipment
    (3) basic read/write operation (single read and write operation)
    (4) configure a task
    (5) initialization of a task
    (6) start a task
    (7) stop a task
    (8) provide error code/message

    The DAQ task would require a task manager which is a separate class, which provides
    shared memory and other objects that will be used for communication and data handling
    between working process that acquires data, and process within the user space to
    store and process the data.

    Returns:
        N/A
    """

    # The is a class property indicating whether the board is initialized
    board_init_state: bool = False
    board_name: str = "DAQBaseClass"
    _error_code: int = 0
    _error_msg: str = ""
    _task_manager: DAQTaskManager = None

    @classmethod
    def ConfigBoard(cls, board_params: dict) -> int:
        print("Base ConfigBoard called")
        return 0

    @classmethod
    def GetBoardInfo(cls) -> dict:
        print("Base GetBoardInfo called")
        return {}

    @classmethod
    def InitBoard(cls) -> int:
        if cls.board_init_state:
            raise Exception("DAQBoard can only be intialized once at a time.")
        else:
            print("Base InitBoard called")
            cls.board_init_state = True
        return 0

    @classmethod
    def ShutdownBoard(cls) -> int:
        if not cls.board_init_state:
            raise Exception("DAQBoard not initialized when calling ShutdownBoard")
        else:
            print("Base ShutdownBoard called")
            cls.board_init_state = False
        return 0

    @classmethod
    def SingleRead(cls, param: dict) -> dict:
        print("Base SingleRead called")
        return None

    @classmethod
    def SingleWrite(cls, param: dict) -> None:
        print("Base SingleWrite called")
        return None

    @classmethod
    def ConfigTask(cls, lock: Lock, task_manager: DAQTaskManager) -> int:
        print("Base ConfigTask called")
        return 0

    @classmethod
    def InitTask(cls, lock: Lock, task_context: DAQTaskManager.TaskContext) -> int:
        print("Base InitTask called")
        return 0

    @classmethod
    def StartTask(cls, lock: Lock, task_context: DAQTaskManager.TaskContext) -> int:
        print("Base StartTask called")
        return 0

    @classmethod
    def ContinueTask(cls, lock: Lock, task_context: DAQTaskManager.TaskContext) -> int:
        print("Continue Task")
        return 0
    
    @classmethod
    def StopTask(cls, lock: Lock, task_manager: DAQTaskManager.TaskContext) -> int:
        print("Base StopTask called")
        return 0

    @classmethod
    def GetErrorCode(cls) -> int:
        return 0

    @classmethod
    def GetErrorMsg(cls) -> str:
        return ""


class DAQTASK_STATUS(Enum):
    UNINITIALIZED = -1
    INITIALIZED = 0
    RUNNING = 1
    SHUTTINGDOWN = 2
    STOPPED = 3
    CRITICAL_ERROR = 255

    def __int__(self):
        return self.value

    def __eq__(self, other):
        return int(other) == int(self.value)


# by default, 32kByte memory buffer is allocated for each instrument instance
DEFAULT_DAQ_BUFFER_SIZE = 32768
# by default, the parameter list contains a 256-byte binary string, 16 integer
# values, and 16 float values
DEFAULT_DAQ_PARAM_LIST_SIZE = 64


def default_daq_process_prototype(task_manager: DAQTaskManager):
    pass


class DAQTaskManager:
    """DAQ Task manager class
    This is the task manager for handling DAQ tasks associatd with a DAQBase instance.
    The DAQBase instance will be provided with the task manager so that the instance
    can perform tasks such as registering additional parameter/information to share
    and allocate additional buffer space for data exchange and handling among
    user processes and the data acquisition process.
    This manager also provides several basic items for the purpose of tracking
    status and exchange quick command/information between user process and
    data aquisition process, such as the status of acquisition, the intention
    to start/stop the data acquisition task, and request to process data in
    real time.
    """

    class TaskContext:
        """_summary_

        Args:
            shared_mem_ID (dict): _description_
        """

        def __init__(
            self, lock: Lock, shared_mem_info: dict, master_process: bool = False
        ):
            """_summary_

            Args:
                lock (Lock): _description_ lock has to be passed as
                            function parameter across processes,
                            cannot be used with Pool
                shared_mem_info (dict): _description_
                master_process (bool, optional): _description_.
                                            Defaults to True.
            """
            param_list_length = shared_mem_info["__param_list_length"]
            buf_size = shared_mem_info["__buf_size"]

            self.task_info = {"shared_memory_info": shared_mem_info}
            self.lock = lock

            self._pid = os.getpid()

            if master_process:
                # Allocate the shared memory for parameter lists
                # Three lists will be provided with different data types
                # (1) byte - potential use for flags/messages
                # (2) int64 - potential use for index, size info
                # (3) float64 - potential use for sampling parameters
                self._unlink_shared_mem = True

                tmp_param_byte_array = np.zeros(
                    shape=(param_list_length,), dtype=np.byte
                )
                tmp_param_int64_array = np.zeros(
                    shape=(param_list_length,), dtype=np.int64
                )
                tmp_param_float64_array = np.zeros(
                    shape=(param_list_length,), dtype=np.float64
                )
                tmp_int64_singlevalue = np.zeros(shape=(1,), dtype=np.int64)

                self._param_byte_membase = SharedMemory(
                    name=shared_mem_info["__param.byte"],
                    create=True,
                    size=tmp_param_byte_array.nbytes,
                )
                self._param_int64_membase = SharedMemory(
                    name=shared_mem_info["__param.int64"],
                    create=True,
                    size=tmp_param_int64_array.nbytes,
                )
                self._param_double_membase = SharedMemory(
                    name=shared_mem_info["__param.double"],
                    create=True,
                    size=tmp_param_float64_array.nbytes,
                )

                self._data_buf_membase = SharedMemory(
                    name=shared_mem_info["__buf"],
                    create=True,
                    size=buf_size,
                )

                self._status_membase = SharedMemory(
                    name=shared_mem_info["__status"],
                    create=True,
                    size=tmp_int64_singlevalue.nbytes,
                )

                self._command_membase = SharedMemory(
                    name=shared_mem_info["__command"],
                    create=True,
                    size=tmp_int64_singlevalue.nbytes,
                )
                print(f"PID {self._pid} creates shared memory.")
                print("DAQTask information as following:")
                print(self.task_info)
            else:
                # act as a child process, will attach to previousely created
                # shared memory
                self._unlink_shared_mem = False

                self._param_byte_membase = SharedMemory(
                    name=shared_mem_info["__param.byte"], create=False
                )
                self._param_int64_membase = SharedMemory(
                    name=shared_mem_info["__param.int64"], create=False
                )
                self._param_double_membase = SharedMemory(
                    name=shared_mem_info["__param.double"], create=False
                )

                self._data_buf_membase = SharedMemory(
                    name=shared_mem_info["__buf"], create=False
                )

                self._status_membase = SharedMemory(
                    name=shared_mem_info["__status"], create=False
                )

                self._command_membase = SharedMemory(
                    name=shared_mem_info["__command"],
                    create=False,
                )
                print(f"PID {self._pid} attaches to previously created shared memory.")
                print("DAQTask information as following:")
                print(self.task_info)

            # Now create the properties associated with the shared memory
            self._param_byte = np.ndarray(
                shape=(param_list_length,),
                dtype=np.byte,
                buffer=self._param_byte_membase.buf,
            )

            self._param_int64 = np.ndarray(
                shape=(param_list_length,),
                dtype=np.int64,
                buffer=self._param_int64_membase.buf,
            )

            self._param_float64 = np.ndarray(
                shape=(param_list_length,),
                dtype=np.float64,
                buffer=self._param_double_membase.buf,
            )

            self.data_buf = np.ndarray(
                shape=(buf_size,), dtype=np.byte, buffer=self._data_buf_membase.buf
            )

            self._status = np.ndarray(
                shape=(1,), dtype=np.int64, buffer=self._status_membase.buf
            )

            self._command = np.ndarray(
                shape=(1,), dtype=np.int64, buffer=self._command_membase.buf
            )
            if master_process:
                self._status[0] = DAQTASK_STATUS.UNINITIALIZED
                self._command[0] = 0

        def __del__(self):
            print(f"PID {self._pid} closes shared memory.")
            self._param_byte_membase.close()
            self._param_int64_membase.close()
            self._param_double_membase.close()
            self._data_buf_membase.close()
            self._status_membase.close()
            self._command_membase.close()

            if self._unlink_shared_mem:
                print(f"PID {self._pid} unlinks the shared memory.")
                print("DAQTask information as following:")
                print(self.task_info)
                self._param_byte_membase.unlink()
                self._param_int64_membase.unlink()
                self._param_double_membase.unlink()
                self._data_buf_membase.unlink()
                self._status_membase.unlink()
                self._command_membase.unlink()

        def get_status(self) -> np.int64:
            return self._status[0]

        def set_status(self, s: np.int64):
            with self.lock:
                self._status[0] = s

        def get_command(self) -> np.int64:
            return self._command[0]

        def set_command(self, c: np.int64):
            with self.lock:
                self._command[0] = c

        def get_param_byte(self, idx: int) -> np.byte:
            return self._param_byte[idx]

        def set_param_byte(
            self,
            idx: int,
            v: np.byte,
            test_and_set: bool = False,
            mask: np.byte = 0xFF,
            test_value: np.byte = 0,
        ):
            with self.lock:
                if test_and_set:
                    if self._param_byte[idx] & mask == test_value:
                        self._param_byte[idx] = self._param_byte[idx] & ~mask | v & mask
                else:
                    self._param_byte[idx] = v

        def get_param_int(self, idx: int) -> np.int64:
            return self._param_int64[idx]

        def set_param_int(self, idx: int, v: np.int64):
            with self.lock:
                self._param_int64[idx] = v

        def get_param_double(self, idx: int) -> np.float64:
            return self._param_float64[idx]

        def set_param_double(self, idx: int, v: np.float64):
            with self.lock:
                self._param_float64[idx] = v

    def __init__(
        self,
        instrument: DAQBaseClass,
        board_params: dict,
        param_list_length: int = DEFAULT_DAQ_PARAM_LIST_SIZE,
        buffer_size: int = DEFAULT_DAQ_BUFFER_SIZE,
    ) -> None:
        """
        The initialization of the task manager involves the following items:

        (1) allocate the shared memory for parameter lists (including byte type, int64 type and double type)
        (2) allocate the shared memory for buffer for data storage/exchange

        """
        self.uuid_postfix = uuid.uuid4().hex
        # Generate unique names for all the shared memory objects

        self.shared_mem_info = {
            "__board_name": instrument.board_name,
            "__param.byte": "daq_param_" + self.uuid_postfix + ".byte",
            "__param.int64": "daq_param_" + self.uuid_postfix + ".int64",
            "__param.double": "daq_param_" + self.uuid_postfix + ".double",
            "__buf": "daq_buf_" + self.uuid_postfix,
            "__status": "daq_status_" + self.uuid_postfix,
            "__command": "daq_command_" + self.uuid_postfix,
            "__buf_size": buffer_size,
            "__param_list_length": param_list_length,
        }

        self.lock = Lock()
        # create shared memory for the master process
        self.context = DAQTaskManager.TaskContext(
            lock=self.lock, shared_mem_info=self.shared_mem_info, master_process=True
        )
        self.DAQBoard = instrument
        self._daq_proc = default_daq_process_prototype
        self._task_status = DAQTASK_STATUS.UNINITIALIZED

        if not self.DAQBoard.board_init_state:
            self.DAQBoard.ConfigBoard(board_params=board_params)
            self.DAQBoard.InitBoard()

    def __del__(self):
        if self._task_status == DAQTASK_STATUS.RUNNING:
            self.stop_task()
        if self.DAQBoard.board_init_state:
            self.DAQBoard.ShutdownBoard()

    @property
    def shared_memory_info(self) -> dict:
        return self.context.task_info["shared_memory_info"]
    
    @property
    def task_params(self) -> dict:
        if self.context.task_info.has_key("task_params"):
            return self.context.task_info["task_params"]
        else:
            return {}

    def config_task(self, task_params: dict) -> int:
        self.context.task_info["task_params"] = task_params
        self.DAQBoard.ConfigTask(task_params=task_params, task_manager=self)
        

    def init_task(self) -> int:
        self.DAQBoard.InitTask(self.lock, self.context)
        self.daq_proc = Process(
            target=DAQTaskManager.daq_job,
            args=(self.lock, self.DAQBoard, self.context.task_info),
        )
        self.daq_proc.start()

    @staticmethod
    def daq_job(l: Lock, board: DAQBaseClass, info: dict):
        p = psutil.Process()
        if sys.platform == "win32":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10)

        task_context = DAQTaskManager.TaskContext(
            lock=l, shared_mem_info=info["shared_memory_info"], master_process=False
        )
        task_context.set_status(DAQTASK_STATUS.INITIALIZED)
        print("child process started...", flush=True)
        print(task_context.lock, flush=True)
        while True:
            c = task_context.get_command()
            match c:
                case 1:
                    board.StartTask(task_context)
                    task_context.set_status(DAQTASK_STATUS.RUNNING)
                    print(
                        f"child proc: task started...command {c}, status {task_context.get_status()}",
                        flush=True,
                    )
                    task_context.set_command(0)
                case -1:
                    print(
                        f"child proc: task quitting...command {c}, status {task_context.get_status()}",
                        flush=True,
                    )
                    board.StopTask(task_context)
                    task_context.set_status(DAQTASK_STATUS.STOPPED)
                    break
                case 0:
                    pass
                case _:
                    task_context.set_status(DAQTASK_STATUS.CRITICAL_ERROR)
                    print(
                        f"child process: got unknown value for command: command {c}, status {task_context.get_status()}",
                        flush=True,
                    )
                    task_context.set_command(0)

            time.sleep(0)
        return 0

    def start_task(self) -> int:
        while True:
            if self.context.get_status() == DAQTASK_STATUS.UNINITIALIZED:
                self.context.set_command(1)
                time.sleep(0.1)
                print(
                    f"waiting for child process to initialize... command{self.context.get_command()}, status{self.context.get_status()}",
                    flush=True,
                )
            else:
                print("child process left UNINITIALIZED state...", flush=True)
                break

    def stop_task(self) -> int:
        self.context.set_command(-1)
        self.daq_proc.join()


if __name__ == "__main__":
    task = DAQTaskManager(DAQBaseClass, {})
    print("OK")
    task.DAQBoard.SingleRead({})
    task.DAQBoard.SingleWrite({})
    task.config_task({})
    task.init_task()
    task.context.set_command(1)
    print(f"main proc set command as {task.context.get_command()}", flush=True)
    print("main lock:", flush=True)
    print(task.lock, flush=True)
    print("main proc tries to start the task...", flush=True)
    task.start_task()
    i = 0
    while True:
        """
        print(
            f"main proc waiting...command: {task.context.get_command()}, status: {task.context.get_status()}",
            flush=True,
        )
        """
        if task.context.get_status() != DAQTASK_STATUS.RUNNING:
            print(f"main proc detected that child process is no longer in running state... will quit now... {task.context.get_command()}, status: {task.context.get_status()}, i: {i}")
            break
        if i == 100000:
            task.context.set_command(2)
            print(
                f"main proc set command to an unknown value command: {task.context.get_command()}, status: {task.context.get_status()}, i: {i}",
                flush=True,
            )
        i += 1
        time.sleep(0)
    task.stop_task()
