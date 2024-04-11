#!/usr/bin/env python3

import asyncio
import fractions
import json
import queue
import ssl
import time
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import (MediaPlayer, MediaRelay, MediaStreamTrack,
                                  PlayerStreamTrack)
from aiortc.rtcrtpsender import RTCRtpSender
from av import AVError, VideoFrame
from loguru import logger
import aiohttp_cors

ROOT = Path(__file__).parent.parent.resolve()

VIDEO_CLOCK_RATE = 600
VIDEO_PTIME = 1 / 30
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)

relay = None
player = None
pcs = set()
input_queue = None


class OpenCVPlayerStreamTrack(PlayerStreamTrack):

    def __init__(self, player, kind):
        super().__init__(player, kind)
        self.in_q: Queue[numpy.ndarray] = None


def opencv_player_worker(loop: asyncio.AbstractEventLoop,
                         video_track: OpenCVPlayerStreamTrack,
                         quit_event: Event, throttle_playback: bool):
    logger.info("Starting worker")
    video_first_pts: Optional[int] = None
    frame_time: Optional[int] = None
    start_time = time.time()
    _start: float = time.time()
    _timestamp: int = 0

    while not quit_event.is_set():
        try:
            frame = video_track.in_q.get(timeout=0.5)
        except (AVError, StopIteration):
            if video_track:
                asyncio.run_coroutine_threadsafe(video_track.__queue.put(None),
                                                 loop)
            break
        except queue.Empty:
            logger.warning("Empty queue")
            break

        _timestamp = int((time.time()-start_time)/30*600)*30
        # wait = max(0, _start + (_timestamp / VIDEO_CLOCK_RATE) - time.time())
        # time.sleep(wait)
        # logger.info(f"{_timestamp=}")

        if throttle_playback:
            elapsed_time = time.time() - start_time
            if frame_time and frame_time > elapsed_time + 1:
                time.sleep(0.1)
        if isinstance(frame, numpy.ndarray):
            # Convert to VideoFrame
            vf = VideoFrame.from_ndarray(frame, format="bgr24")
            vf.pts = _timestamp
            vf.time_base = VIDEO_TIME_BASE
            asyncio.run_coroutine_threadsafe(video_track._queue.put(vf), loop)
    logger.info("Worker stop event recieved")


class OpenCVMediaPlayer(MediaPlayer):

    def __init__(self, queue: Queue[numpy.ndarray]):
        self.__thread: Optional[Thread] = None
        self.__thread_quit: Optional[Event] = None
        self.__started: Set[OpenCVPlayerStreamTrack] = set()
        self.__streams: List[int] = []
        self.__video: OpenCVPlayerStreamTrack = OpenCVPlayerStreamTrack(
            self, kind="video")
        self.__streams.append(1)
        self._throttle_playback = False

        self.__video.in_q = queue

    def videoResponse(self):
        self.__video.response()

    @property
    def video(self) -> MediaStreamTrack:
        return self.__video

    def _start(self, track: OpenCVPlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            logger.info("Starting video trak.")
            self.__thread_quit = Event()
            self.__thread = Thread(name="media-player",
                                   target=opencv_player_worker,
                                   args=(asyncio.get_event_loop(),
                                         self.__video, self.__thread_quit,
                                         self._throttle_playback))
            self.__thread.start()

    def _stop(self, track: OpenCVPlayerStreamTrack) -> None:
        logger.info("Stopping video trak.")
        self.__started.discard(track)
        if not self.__started and self.__thread is not None:
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None


def create_local_tracks(
    queue: Queue[numpy.ndarray]
) -> Tuple[Optional[MediaStreamTrack], Optional[MediaStreamTrack]]:
    global relay, player
    player = OpenCVMediaPlayer(queue)
    if relay is None:
        relay = MediaRelay()
    return None, relay.subscribe(player.video, buffered=False)


def force_codec(pc: RTCPeerConnection, sender: RTCRtpSender,
                forced_codec: str) -> None:
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transciever = next(t for t in pc.getTransceivers() if t.sender == sender)
    transciever.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec])


async def index(request: web.Request):
    logger.info("Serving index.html")
    content = open(ROOT / "templates" / "index.html", "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request: web.Request):
    logger.info("serving client.js")
    content = open(ROOT / "static" / "client.js", "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request: web.Request):
    logger.info(f"Request: {type(request)}, {request}")
    logger.info("Getting params")
    params = await request.json()
    logger.info("Creaating offer")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params['type'])
    logger.info("Createing Peer Connection")
    pc = RTCPeerConnection()
    logger.info("Adding Peer connection to list")
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionsstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            logger.info("Connection failed, stopping")
            for vt in pc.getSenders():
                logger.info("Stopping senders")
                logger.info(f"Sender track is {type(vt.track)}")
            await pc.close()
            pcs.discard(pc)

    logger.info("Creating local video tracks.")
    _, video = create_local_tracks(input_queue)

    if video:
        logger.info("Adding video track to peer connection")
        video_sender = pc.addTrack(video)
        logger.info("Forcing video track codec")
        force_codec(pc, video_sender, "video/H264")
    logger.info("Setting remote description to offer")
    await pc.setRemoteDescription(offer)
    logger.info("Waiting for answer")
    answer = await pc.createAnswer()
    logger.info("Setting local description to answer")
    await pc.setLocalDescription(answer)
    logger.info("Returning response")
    return web.Response(content_type="application/json",
                        text=json.dumps({
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type
                        }))


async def on_shutdown(app: web.Application) -> None:
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


class Camera:

    def __init__(self, device: Union[str, int], output: Queue[numpy.ndarray]):
        self.__device = device
        self.__cap: cv2.VideoCapture = cv2.VideoCapture(self.__device)
        self.__thread: Thread = None
        self.__thread_quit: Event = None
        self.__frame: numpy.ndarray
        self.__q: Queue[numpy.ndarray] = output

    @property
    def device(self) -> str:
        return self.__device

    @device.setter
    def set_device(self, device: str) -> None:
        self.__device = device
        if self.__cap is not None:
            self.__cap.release()
        self.__cap = cv2.VideoCapture(device)
        if not self.__cap.isOpened():
            raise RuntimeError(f"Cannot open device {self.__device}")

    @property
    def frame(self) -> numpy.ndarray:
        return self.__frame

    @property
    def options(self) -> Dict[str, Any]:
        if not self.__cap:
            raise RuntimeError("Device not present")
        width = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.__cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.__cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return {"width": width, "height": height, "fps": fps, "fourcc": fourcc}

    def set_options(self, settings: Dict) -> None:
        if not self.__cap:
            raise RuntimeError("Device not present")
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
        self.__cap.set(cv2.CAP_PROP_FPS, settings['fps'])
        self.__cap.set(cv2.CAP_PROP_FOURCC,
                       cv2.VideoWriter_fourcc(*settings['fourcc']))

    def start(self):
        logger.info("Starting camera thread")
        self.__thread_quit = Event()
        self.__thread = Thread(name="camera_worker",
                               target=Camera.__run,
                               args=(self, ))
        self.__thread.start()

    def stop(self):
        logger.info("Stopping camera thread")
        self.__thread_quit.set()
        self.__thread.join()
        self.__thread_quit.clear()

    def __run(self):
        logger.info("Camera started")
        while not self.__thread_quit.is_set():
            ret, frame = self.__cap.read()
            if not ret:
                raise RuntimeError(f"Error during frame reading")
            self.__frame = frame
            try:
                self.__q.put(frame, block=False, timeout=0.1)
            except queue.Full:
                self.__q.get_nowait()
            else:
                time.sleep(0.01)
        logger.info("Camera stopped")


if __name__ == '__main__':
    logger.info(f"ROOT: {ROOT}")

    input_queue = queue.Queue(5)

    camera = Camera(0, input_queue)
    logger.info(f'Options: {camera.options}')
    logger.info(f"Camera :{camera.device}")
    camera.set_options({
        "width": 1920,
        "height": 1280,
        "fps": 30,
        "fourcc": "H264"
    })
    logger.info(f'Options: {camera.options}')

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
    ssl_context.load_cert_chain("cert.crt", "cert.key")

    app = web.Application()

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
    })

    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    cors.add(app.router.add_post("/offer", offer))

    camera.start()
    web.run_app(app=app, host="0.0.0.0", port=8888, ssl_context=None)
    camera.stop()
