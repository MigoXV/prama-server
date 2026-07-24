from __future__ import annotations

import grpc

GRPC_MAX_MESSAGE_LENGTH_BYTES = 500 * 1024 * 1024

GRPC_CHANNEL_OPTIONS = (
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH_BYTES),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH_BYTES),
)


def create_insecure_channel(target: str) -> grpc.Channel:
    return grpc.insecure_channel(target, options=GRPC_CHANNEL_OPTIONS)
