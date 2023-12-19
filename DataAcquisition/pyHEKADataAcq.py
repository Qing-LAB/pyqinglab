import sys
import os
import ctypes
import numpy as np
from enum import Enum
from multiprocessing import Process


class LIH_BoardType(Enum):
    LIH_NONE = -1
    LIH_ITC16Board = 0
    LIH_ITC18Board = 1
    LIH_LIH1600Board = 2
    LIH_LIH88Board = 3
    LIH_ITC16USB = 10
    LIH_ITC18USB = 11


class LIH_AdcRange(Enum):
    LIH_InvalidRange = -1
    LIH_AdcRange10V = 0
    LIH_AdcRange5V = 1
    LIH_AdcRange2V = 2
    LIH_AdcRange1V = 3


class LIH_OptionsType(ctypes.Structure):
    _fields_ = [
        ("UseUSB", ctypes.c_int32),
        ("BoardNumber", ctypes.c_int32),
        ("FIFOSamples", ctypes.c_int32),
        ("MaxProbes", ctypes.c_int32),
        ("DeviceNumber", ctypes.c_char * 16),
        ("SerialNumber", ctypes.c_char * 16),
        ("ExternalScaling", ctypes.c_int32),
        ("DacScaling", ctypes.c_void_p),
        ("AdcScaling", ctypes.c_void_p),
    ]


class HEKADataAcq:
    """
    The HEKADataAcq wraps the EpcDLL.dll/EpcDLL64.dll for python.
    The functions from the DLL is saved as static class function starting with _LIH_
    The class provides a group of Python functions for each instance of the class
    """

    _dll = None
    _LIH_InitializeInterface = None
    _LIH_Shutdown = None
    _LIH_PhysicalChannels = None
    _LIH_StartStimAndSample = None
    _LIH_AvailableStimAndSample = None
    _LIH_ReadStimAndSample = None
    _LIH_AppendToFIFO = None
    _LIH_Halt = None
    _LIH_ForceHalt = None
    _LIH_ReadAdc = None
    _LIH_VoltsToDacUnits = None
    _LIH_AdcUnitsToVolts = None
    _LIH_ReadDigital = None
    _LIH_ReadAll = None
    _LIH_SetDac = None
    _LIH_GetDigitalOutState = None
    _LIH_SetDigital = None
    _LIH_CheckSampleInterval = None
    _LIH_SetInputRange = None
    _LIH_GetBoardType = None
    _LIH_GetErrorText = None
    _LIH_GetBoardInfo = None

    def __init__(
        self,
        path_to_dll: str,
    ):
        self._dacscale = np.ones(8, dtype=np.float64, order="C")
        self._adcscale = np.ones(16, dtype=np.float64, order="C")
        self._errmessage = np.zeros(256, dtype=np.byte, order="C")

        self._board_type = LIH_BoardType.LIH_NONE
        self._sec_per_tick = ctypes.c_double(np.NaN)
        self._min_sampling_time = ctypes.c_double(np.NaN)
        self._max_sampling_time = ctypes.c_double(np.NaN)
        self._fifo_length = ctypes.c_int32(-1)
        self._number_of_dacs = ctypes.c_int32(0)
        self._number_of_adcs = ctypes.c_int32(0)

        self._init_state = False

        try:
            if HEKADataAcq._dll is None:
                HEKADataAcq._dll = ctypes.cdll.LoadLibrary(path_to_dll)

            assert HEKADataAcq._dll is not None
            HEKADataAcq._LIH_InitializeInterface = (
                HEKADataAcq._dll.LIH_InitializeInterface
            )
            assert HEKADataAcq._LIH_InitializeInterface is not None
            HEKADataAcq._LIH_InitializeInterface.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.c_int32,
            ]
            HEKADataAcq._LIH_InitializeInterface.restype = ctypes.c_int32

            HEKADataAcq._LIH_Shutdown = HEKADataAcq._dll.LIH_Shutdown
            assert HEKADataAcq._LIH_Shutdown is not None
            HEKADataAcq._LIH_Shutdown.argtypes = []
            HEKADataAcq._LIH_Shutdown.restype = None

            HEKADataAcq._LIH_Halt = HEKADataAcq._dll.LIH_Halt
            assert HEKADataAcq._LIH_Halt is not None
            HEKADataAcq._LIH_Halt.argtypes = []
            HEKADataAcq._LIH_Halt.restype = None

            HEKADataAcq._LIH_ForceHalt = HEKADataAcq._dll.LIH_ForceHalt
            assert HEKADataAcq._LIH_ForceHalt is not None
            HEKADataAcq._LIH_ForceHalt.argtypes = []
            HEKADataAcq._LIH_ForceHalt.restype = None

            HEKADataAcq._LIH_CheckSampleInterval = (
                HEKADataAcq._dll.LIH_CheckSampleInterval
            )
            assert HEKADataAcq._LIH_CheckSampleInterval is not None
            HEKADataAcq._LIH_CheckSampleInterval.argtypes = [ctypes.c_double]
            HEKADataAcq._LIH_CheckSampleInterval.restype = ctypes.c_double

            HEKADataAcq._LIH_SetInputRange = HEKADataAcq._dll.LIH_SetInputRange
            assert HEKADataAcq._LIH_SetInputRange is not None
            HEKADataAcq._LIH_SetInputRange.argtypes = [ctypes.c_int32, ctypes.c_int32]
            HEKADataAcq._LIH_SetInputRange.restype = ctypes.c_int32

            HEKADataAcq._LIH_GetBoardType = HEKADataAcq._dll.LIH_GetBoardType
            assert HEKADataAcq._LIH_GetBoardType is not None
            HEKADataAcq._LIH_GetBoardType.argtypes = []
            HEKADataAcq._LIH_GetBoardType.restype = ctypes.c_int32

            HEKADataAcq._LIH_GetErrorText = HEKADataAcq._dll.LIH_GetErrorText
            assert HEKADataAcq._LIH_GetErrorText is not None
            HEKADataAcq._LIH_GetErrorText.argtypes = [ctypes.c_void_p]
            HEKADataAcq._LIH_GetErrorText.restype = None

            HEKADataAcq._LIH_GetBoardInfo = HEKADataAcq._dll.LIH_GetBoardInfo
            assert HEKADataAcq._LIH_GetBoardInfo is not None
            HEKADataAcq._LIH_GetBoardInfo.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
            ]
            HEKADataAcq._LIH_GetBoardInfo.restype = None

            HEKADataAcq._LIH_ReadAdc = HEKADataAcq._dll.LIH_ReadAdc
            assert HEKADataAcq._LIH_ReadAdc is not None
            HEKADataAcq._LIH_ReadAdc.argtypes = [ctypes.c_int32]
            HEKADataAcq._LIH_ReadAdc.restype = ctypes.c_int32

            HEKADataAcq._LIH_AdcUnitsToVolts = HEKADataAcq._dll.LIH_AdcUnitsToVolts
            assert HEKADataAcq._LIH_AdcUnitsToVolts is not None
            HEKADataAcq._LIH_AdcUnitsToVolts.argtypes = [ctypes.c_int32, ctypes.c_int32]
            HEKADataAcq._LIH_AdcUnitsToVolts.restype = ctypes.c_double

            HEKADataAcq._LIH_VoltsToDacUnits = HEKADataAcq._dll.LIH_VoltsToDacUnits
            assert HEKADataAcq._LIH_VoltsToDacUnits is not None
            HEKADataAcq._LIH_VoltsToDacUnits.argtypes = [
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_double),
            ]
            HEKADataAcq._LIH_VoltsToDacUnits.restype = ctypes.c_int32

            HEKADataAcq._LIH_SetDac = HEKADataAcq._dll.LIH_SetDac
            assert HEKADataAcq._LIH_SetDac is not None
            HEKADataAcq._LIH_SetDac.argtypes = [ctypes.c_int32, ctypes.c_int32]
            HEKADataAcq._LIH_SetDac.restype = ctypes.c_int32

            HEKADataAcq._LIH_ReadDigital = HEKADataAcq._dll.LIH_ReadDigital
            assert HEKADataAcq._LIH_ReadDigital is not None
            HEKADataAcq._LIH_ReadDigital.argtypes = []
            HEKADataAcq._LIH_ReadDigital.restype = ctypes.c_int16

            HEKADataAcq._LIH_ReadAll = HEKADataAcq._dll.LIH_ReadAll
            assert HEKADataAcq._LIH_ReadAll is not None
            HEKADataAcq._LIH_ReadAll.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_double,
            ]
            HEKADataAcq._LIH_ReadAll.restype = ctypes.c_ubyte

            HEKADataAcq._LIH_GetDigitalOutState = (
                HEKADataAcq._dll.LIH_GetDigitalOutState
            )
            assert HEKADataAcq._LIH_GetDigitalOutState is not None
            HEKADataAcq._LIH_GetDigitalOutState.argtypes = []
            HEKADataAcq._LIH_GetDigitalOutState.restype = ctypes.c_int16

            HEKADataAcq._LIH_SetDigital = HEKADataAcq._dll.LIH_SetDigital
            assert HEKADataAcq._LIH_SetDigital is not None
            HEKADataAcq._LIH_SetDigital.argtypes = [ctypes.c_int16]
            HEKADataAcq._LIH_SetDigital.restype = None

            HEKADataAcq._LIH_StartStimAndSample = (
                HEKADataAcq._dll.LIH_StartStimAndSample
            )
            assert HEKADataAcq._LIH_StartStimAndSample is not None
            HEKADataAcq._LIH_StartStimAndSample.argtypes = [
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int16,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_ubyte,
                ctypes.c_ubyte,
            ]
            HEKADataAcq._LIH_StartStimAndSample.restype = ctypes.c_ubyte

            HEKADataAcq._LIH_AvailableStimAndSample = (
                HEKADataAcq._dll.LIH_AvailableStimAndSample
            )
            assert HEKADataAcq._LIH_AvailableStimAndSample is not None
            HEKADataAcq._LIH_AvailableStimAndSample.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte)
            ]
            HEKADataAcq._LIH_AvailableStimAndSample.restype = ctypes.c_int32

            HEKADataAcq._LIH_ReadStimAndSample = HEKADataAcq._dll.LIH_ReadStimAndSample
            assert HEKADataAcq._LIH_ReadStimAndSample is not None
            HEKADataAcq._LIH_ReadStimAndSample.argtypes = [
                ctypes.c_int32,
                ctypes.c_ubyte,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            HEKADataAcq._LIH_ReadStimAndSample.restype = None

            HEKADataAcq._LIH_AppendToFIFO = HEKADataAcq._dll.LIH_AppendToFIFO
            assert HEKADataAcq._LIH_AppendToFIFO is not None
            HEKADataAcq._LIH_AppendToFIFO.argtypes = [
                ctypes.c_int32,
                ctypes.c_ubyte,
                ctypes.c_void_p,
            ]
            HEKADataAcq._LIH_AppendToFIFO.restype = ctypes.c_ubyte

        except Exception as ex:
            print("Error when initializing HEKADataAcq library.")
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def InitBoard(
        self,
        board_type: LIH_BoardType = LIH_BoardType.LIH_ITC18USB,
        BoardNumber: int = 0,
        FIFOSamples: int = 0,
        EPCAmplifier: int = 1,
    ) -> int:
        """
        Initialize the HEKA data acquisition board
        Parameters:
            board_type : default to LIH_ITC18USB, other models can be see in the class LIH_BoardType
            BoardNumber : only useful when dealing with PCI cards, check the EpcDLL.h file for details
            FIFOSamples : default to zero
            EPCAmplifier: not used in most functions, default to 1. check EpcDLL.h for details
        Return:
            if succeed, returns 0, otherwise, returns the error code. Error message will be printed to
            indicate reasons for the failure.
        """
        retVal = -1
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_InitializeInterface is not None

            self._options = LIH_OptionsType(
                0, BoardNumber, FIFOSamples, 0, b"", b"", 0, None, None
            )

            match board_type:
                case LIH_BoardType.LIH_ITC16Board:
                    board = 0
                case LIH_BoardType.LIH_ITC18Board:
                    board = 1
                case LIH_BoardType.LIH_LIH1600Board:
                    board = 2
                case LIH_BoardType.LIH_LIH88Board:
                    board = 3
                case LIH_BoardType.LIH_ITC16USB:
                    board = 0
                    self._options.UseUSB = 1
                case LIH_BoardType.LIH_ITC18USB:
                    board = 1
                    self._options.UseUSB = 1
                case _:
                    raise ValueError("Unknown type of Board")

            retVal = HEKADataAcq._LIH_InitializeInterface(
                self._errmessage.ctypes.data_as(ctypes.c_char_p),
                EPCAmplifier,
                board,
                ctypes.pointer(self._options),
                ctypes.sizeof(self._options),
            )

            if retVal != 0:
                print(self._errmessage.tobytes().decode("utf-8"))
                raise Exception("Initialization of board failed.")

            retVale = HEKADataAcq._LIH_GetBoardInfo(
                ctypes.byref(self._sec_per_tick),
                ctypes.byref(self._min_sampling_time),
                ctypes.byref(self._max_sampling_time),
                ctypes.byref(self._fifo_length),
                ctypes.byref(self._number_of_dacs),
                ctypes.byref(self._number_of_adcs),
            )
            if retVal != 0:
                self._init_state = False
                print(GetErrorText())
                raise Exception("Cannot get board information.")
            else:
                self._board_type = board_type
                self._init_state = True
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

        return retVal

    def __del__(self):
        if (
            HEKADataAcq._dll is not None
            and HEKADataAcq._LIH_ForceHalt is not None
            and HEKADataAcq._LIH_Shutdown is not None
        ):
            HEKADataAcq._LIH_ForceHalt()
            HEKADataAcq._LIH_Shutdown()

    def ShutdownBoard(self):
        """
        Shutdown the board. Takes no parameter. No return value.
        """
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_ForceHalt is not None
            assert HEKADataAcq._LIH_Shutdown is not None

            HEKADataAcq._LIH_ForceHalt()
            HEKADataAcq._LIH_Shutdown()

            self._board_type = LIH_BoardType.LIH_NONE
            self._sec_per_tick = ctypes.c_double(np.NaN)
            self._min_sampling_time = ctypes.c_double(np.NaN)
            self._max_sampling_time = ctypes.c_double(np.NaN)
            self._fifo_length = ctypes.c_int32(-1)
            self._number_of_dacs = ctypes.c_int32(0)
            self._number_of_adcs = ctypes.c_int32(0)

        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def CheckSamplingInterval(self, requested_sampling_interval: float) -> float:
        if not self._init_state:
            return np.NaN

        true_sampling_interval = np.NaN
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_CheckSamplingInterval is not None

            true_sampling_interval = HEKADataAcq._LIH_CheckSamplingInterval(
                requested_sampling_interval
            )

        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

        return true_sampling_interval

    def GetErrorText(self) -> str:
        errtext = np.zeros(256, dtype=np.byte, order="C")
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_GetErrorText is not None

            HEKADataAcq._LIH_GetErrorText(errtext.ctypes.data_as(ctypes.c_void_p))

        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

        return errtext.tobytes().decode("utf-8")

    def GetBoardInfo(self) -> dict:
        return {
            "Board type": self._board_type,
            "Second per tick": self._sec_per_tick,
            "Min sampling time": self._min_sampling_time,
            "Max sampling time": self._max_sampling_time,
            "FIFO length": self._fifo_length,
            "Number of DACs": self._number_of_dacs,
            "Number of ADCs": self._number_of_adcs,
        }

    def SetInputRange(self, channel: int, input_range: LIH_AdcRange) -> LIH_AdcRange:
        if not self._init_state or (
            channel < 0 or channel >= self._number_of_adcs.value
        ):
            return np.NaN
        actual_range = LIH_AdcRange(LIH_InvalidRange)
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_SetInputRange is not None

            actual_range = LIH_AdcRange(
                HEKADataAcq._LIH_SetInputRange(channel, input_range.value)
            )
        except Exception as ex:
            actual_range = LIH_AdcRange(LIH_InvalidRange)
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return actual_range

    def ReadADCChannel(self, channel: int) -> float:
        if not self._init_state or (
            channel < 0 or channel >= self._number_of_adcs.value
        ):
            return np.NaN
        read_value = np.NaN
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_ReadAdc is not None
            assert HEKADataAcq._LIH_AdcUnitsToVolts is not None

            raw_value = HEKADataAcq._LIH_ReadAdc(channel)
            read_value = HEKADataAcq._LIH_AdcUnitsToVolts(channel, raw_value)
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return read_value

    def SetDACChannel(self, channel: int, output: float) -> float:
        if not self._init_state or (
            channel < 0 or channel >= self._number_of_dacs.value
        ):
            return np.NaN
        actual_output = ctypes.c_double(np.NaN)
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_SetDac is not None
            assert HEKADataAcq._LIH_VoltsToDacUnits is not None

            actual_output = ctypes.c_double(output)
            raw_value = HEKADataAcq._LIH_VoltsToDacUnits(
                channel, ctypes.byref(actual_output)
            )
            HEKADataAcq._LIH_SetDac(channel, raw_value)
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return actual_output.value

    def ReadDigital(self) -> int:
        if not self._init_state:
            return 0
        digit_input = 0
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_ReadDigital is not None

            digit_input = HEKADataAcq._LIH_ReadDigital()
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return digit_input

    def SetDigitalOutput(self, output: int):
        if not self._init_state:
            return None
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_SetDigital is not None

            digi_out = ctypes.c_int16(output & 0xFF)
            HEKADataAcq._LIH_SetDigital(digi_out)
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return None

    def GetDigitalOutputState(self) -> int:
        if not self._init_state:
            return 0
        digit_out = 0
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_GetDigitalOutState is not None

            digit_out = HEKADataAcq._LIH_GetDigitalOutState()
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return digit_out

    def ReadAll(self, interval: float) -> dict:
        if not self._init_state:
            return None
        try:
            assert HEKADataAcq._dll is not None
            assert HEKADataAcq._LIH_ReadAll is not None
            adc_volts = np.zeros(self._number_of_adcs)
            digital_port = ctypes.c_int16(0)
            if interval < self._min_sampling_time.value:
                interval = self._min_sampling_time.value
            if interval > self._max_sampling_time.value:
                interval = self._max_sampling_time.value
            sample_interval = ctypes.c_double(interval)
            retVal = HEKADataAcq._LIH_ReadAll(adc_volts.ctypes.dataas(ctypes.c_void_p), byref(digital_port), sample_interval)
            if retVal:
                all_input = {
                    "ADCs": adc_volts,
                    "DigitalInputs": digital_port,
                    "Interval": sample_interval.value
                }
            else:
                all_input = None
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return all_input

