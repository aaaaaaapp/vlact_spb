# StarVLA RobotWin eval client compatibility layer.
#
# Starvla/examples/Robotwin/eval_files/model2robotwin_interface.py expects websocket
# metadata fields such as action_chunk_size and response["data"]["actions"].
# The default server_policy.py keeps the original minimal metadata for other
# benchmarks (LIBERO, SimplerEnv, etc.).
#
# Use this entry only when pairing VLAct checkpoints with Starvla RobotWin eval.sh.

import logging
import os
import socket

import torch
from accelerate import PartialState

from deployment.model_server.server_policy import build_argparser, start_debugpy_once
from deployment.model_server.starvla_robotwin_serve_wrapper import (
    StarvlaRobotwinServeWrapper,
)
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


def main(args) -> None:
    PartialState()

    vla = baseframework.from_pretrained(args.ckpt_path)
    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    policy = StarvlaRobotwinServeWrapper(
        framework=vla,
        ckpt_path=args.ckpt_path,
        default_unnorm_key=os.environ.get("ROBOTWIN_UNNORM_KEY"),
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating StarVLA-compat server (host: %s, ip: %s)", hostname, local_ip)

    metadata = policy.metadata
    if args.action_chunk_size is not None:
        metadata = dict(metadata)
        metadata["action_chunk_size"] = int(args.action_chunk_size)
    logging.info("Server metadata: %s", metadata)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    logging.info("server running ...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
