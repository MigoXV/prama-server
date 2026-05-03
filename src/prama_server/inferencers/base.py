from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Inferencer(ABC):
    @abstractmethod
    def infer(self, audio: np.ndarray) -> str:
        """对单条音频执行推理并返回识别文本。"""

    def close(self) -> None:
        pass

    def __enter__(self) -> Inferencer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
