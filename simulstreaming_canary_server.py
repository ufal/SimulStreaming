#!/usr/bin/env python3
from simulstreaming_canary import simulcanary_args, simul_asr_factory
from simulstreaming.whisper.whisper_streaming.whisper_server import main_server

if __name__ == "__main__":
    main_server(simul_asr_factory, add_args=simulcanary_args)
