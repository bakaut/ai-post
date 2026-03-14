#!/usr/bin/env python3
from __future__ import annotations

import json

from common.audio_capture import list_input_devices


def main() -> None:
    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return

    print("Input devices:\n")
    for dev in devices:
        print(
            f'[{dev["index"]}] {dev["name"]} | '
            f'inputs={dev["max_input_channels"]} | '
            f'default_samplerate={dev["default_samplerate"]}'
        )

    print("\nJSON:\n")
    print(json.dumps(devices, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
