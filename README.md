# README

Install **poetry**.

The program is configured to run the default device "0" (usually _/dev/video0_) as a source.
You should be able to visit the stream on [Page](https://localhost:8888).

Please check the **STUN** server box.
And press **start**.

The stream should start, and you will be able to examine the debug log in the console window.

# Setup

Install dependecies with `poetry update` or `poetry install`

# Running

Run with

```bash
poetry run python webrtc_async_webcam_opencv/main.py
```

# Other

## Requirements

Get requirements with `poetry export -o requirements.txt --without-hashes`.

## Build wheel

Create a python bundle which can be installed by **pip** with:

```bash
poetry build
```

You will be able to find the packages in **dist**
