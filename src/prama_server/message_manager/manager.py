from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MESSAGE_SENTINEL = object()
_CLOSE_EVENT = "__close__"


@dataclass(frozen=True)
class ManagedMessage:
    job_id: str
    event_name: str
    payload: dict[str, Any]


class MessageManager:
    def __init__(self) -> None:
        self.input_queue: queue.Queue[ManagedMessage] = queue.Queue()
        self._job_queues: dict[str, queue.Queue[ManagedMessage | object]] = {}
        self._lock = threading.Lock()
        self._dispatcher = threading.Thread(
            target=self._dispatch_messages,
            daemon=True,
            name="message-manager-dispatcher",
        )
        self._dispatcher.start()

    def register_job(self, job_id: str) -> queue.Queue[ManagedMessage | object]:
        with self._lock:
            job_queue: queue.Queue[ManagedMessage | object] = queue.Queue()
            self._job_queues[job_id] = job_queue
            return job_queue

    def put(self, message: ManagedMessage) -> None:
        self.input_queue.put(message)

    def subscribe(self, job_id: str) -> queue.Queue[ManagedMessage | object]:
        with self._lock:
            job_queue = self._job_queues.get(job_id)
            if job_queue is None:
                job_queue = queue.Queue()
                self._job_queues[job_id] = job_queue
            return job_queue

    def close_job(self, job_id: str) -> None:
        self.input_queue.put(
            ManagedMessage(job_id=job_id, event_name=_CLOSE_EVENT, payload={})
        )

    def _dispatch_messages(self) -> None:
        while True:
            message = self.input_queue.get()
            with self._lock:
                job_queue = self._job_queues.get(message.job_id)

            if job_queue is None:
                logger.debug("消息目标任务不存在: job_id=%s", message.job_id)
                continue

            if message.event_name == _CLOSE_EVENT:
                job_queue.put(MESSAGE_SENTINEL)
                with self._lock:
                    self._job_queues.pop(message.job_id, None)
                continue

            job_queue.put(message)
