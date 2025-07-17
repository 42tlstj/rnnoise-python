import ctypes
import numpy as np
import os

class RNNoise:
    def __init__(self):
        # DLL 로드
        dll_path = os.path.join(os.path.dirname(__file__), "librnnoise-0.dll")
        self._rnnoise = ctypes.cdll.LoadLibrary(dll_path)

        # DenoiseState 타입 정의
        class DenoiseState(ctypes.Structure):
            pass
        self._DenoiseState = DenoiseState

        # 함수 시그니처 정확히 지정
        self._rnnoise.rnnoise_create.restype = ctypes.POINTER(DenoiseState)
        self._rnnoise.rnnoise_destroy.argtypes = [ctypes.POINTER(DenoiseState)]
        self._rnnoise.rnnoise_process_frame.restype = ctypes.c_float
        self._rnnoise.rnnoise_process_frame.argtypes = [
            ctypes.POINTER(DenoiseState),
            ctypes.POINTER(ctypes.c_float),  # out
            ctypes.POINTER(ctypes.c_float),  # in
        ]

        # 상태 생성
        self.state = self._rnnoise.rnnoise_create(None)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # 480 샘플이 아니라면 패딩
        if frame.ndim != 1 or frame.shape[0] != 480:
            frame = frame.flatten()[:480]
            frame = np.pad(frame, (0, 480 - len(frame)), mode='constant')

        frame = frame.astype(np.float32)

        # 정확한 포인터 타입으로 변환
        in_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_buf = (ctypes.c_float * 480)()
        out_ptr = ctypes.cast(out_buf, ctypes.POINTER(ctypes.c_float))

        # 호출 순서: (state, out, in)
        self._rnnoise.rnnoise_process_frame(self.state, out_ptr, in_ptr)

        # 반환
        return np.ctypeslib.as_array(out_buf)

    def __del__(self):
        if hasattr(self, "state") and self.state:
            self._rnnoise.rnnoise_destroy(self.state)
