import asyncio
import io
import os
import shutil
import tempfile
import traceback
import zipfile
from enum import Enum
from fractions import Fraction
from functools import wraps
from pathlib import Path
from typing import AsyncGenerator
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import aiofiles
import httpx
import typer
from aiostream import stream

FramesType = Dict[str, int]

app = typer.Typer()

ffmpeg_sema = asyncio.Semaphore(os.cpu_count())


class OutFormat(Enum):
    GIF = "gif"
    WEBM = "webm"

    @property
    def create_subprocess(self) -> Callable[[Path, str], asyncio.subprocess.Process]:
        return {
            self.GIF: create_ffmpeg_subprocess,
            self.WEBM: create_ffmpeg_webm_subprocess,
        }[self]

    def make_filename(self, illust_id: int):
        return f"{illust_id}.{self.value}"


def run_coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


async def copyfileobj(fsrc, fdst, length=0):
    if not length:
        length = shutil.COPY_BUFSIZE

    iscoro = asyncio.iscoroutine

    fsrc_read = fsrc.read
    fdst_write = fdst.write

    fsrc_read_async = None
    fdst_write_async = None

    while True:
        buf = fsrc_read(length)
        if fsrc_read_async is None:
            fsrc_read_async = iscoro(buf)
        if fsrc_read_async is True:
            buf = await buf

        if not buf:
            break

        write = fdst_write(buf)
        if fdst_write_async is None:
            fdst_write_async = iscoro(write)
        if fdst_write_async is True:
            write = await write


async def create_ffmpeg_subprocess(
    directory: Path,
    fps: str,
) -> asyncio.subprocess.Process:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        "ffconcat.txt",
        "-vf",
        "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop",
        "0",
        "-r",
        fps,
        "-f",
        "gif",
        "out",
        cwd=directory,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    return process


async def create_ffmpeg_webm_subprocess(
    directory: Path,
    fps: str,
) -> asyncio.subprocess.Process:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        "ffconcat.txt",
        "-c:v",
        "libvpx-vp9",
        "-lossless",
        "1",
        "-r",
        fps,
        "-f",
        "webm",
        "out",
        cwd=directory,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    return process


def get_fps(frames: FramesType) -> str:
    return str(Fraction(len(frames) * 1000, sum(frames.values())))


async def create_ffconcat_file(directory: Path, frames: FramesType) -> None:
    filepath = directory.joinpath("ffconcat.txt")

    async with aiofiles.open(filepath, "w") as f:
        await f.write("ffconcat version 1.0\n\n")

        for i in range(1):
            for filename, duration in frames.items():
                await f.write(f"file {filename}\n" f"duration {duration/1000}\n\n")


def get_metadata_url(illust_id: int) -> str:
    return f"https://www.pixiv.net/ajax/illust/{illust_id}/ugoira_meta"


def get_artist_url(artist_id: int) -> str:
    return f"https://www.pixiv.net/ajax/user/{artist_id}/profile/all"


async def get_artist_illustrations(
    client: httpx.AsyncClient, artist_id: int
) -> AsyncGenerator[int, None]:
    url = get_artist_url(artist_id)
    response = await client.get(url)
    data = response.json()

    if data["error"]:
        raise ValueError("invalid artist_id")

    for illust_id in data["body"]["illusts"].keys():
        yield illust_id


async def download_zip(
    client: httpx.AsyncClient, illust_id: int
) -> Tuple[bytes, FramesType]:
    url = get_metadata_url(illust_id)
    response = await client.get(url)
    data = response.json()

    if data["error"]:
        raise ValueError("invalid illust_id")

    url = data["body"]["originalSrc"]
    response = await client.get(url, headers={"Referer": "https://www.pixiv.net/"})
    response.raise_for_status()

    frames = {f["file"]: f["delay"] for f in data["body"]["frames"]}
    return response.content, frames


async def unpack_zip(content: bytes, directory: Path) -> None:
    zf = zipfile.ZipFile(io.BytesIO(content))

    for name in zf.namelist():
        src = zf.open(name)
        async with aiofiles.open(directory.joinpath(name), "wb") as dst:
            await copyfileobj(src, dst)


async def process_illust(
    client: httpx.AsyncClient, format: OutFormat, illust_id: int, out_dir: Path
) -> None:
    zf, frames = await download_zip(client, illust_id)

    with tempfile.TemporaryDirectory(dir=out_dir) as directory:
        directory = Path(directory)
        await asyncio.gather(
            create_ffconcat_file(directory, frames),
            unpack_zip(zf, directory),
        )
        fps = get_fps(frames)
        async with ffmpeg_sema:
            ff = await format.create_subprocess(directory, fps)
            ff_code = await ff.wait()
        if ff_code == 0:
            out_fn = format.make_filename(illust_id)
            shutil.move(
                directory.joinpath("out"),
                out_dir.joinpath(out_fn),
            )
        else:
            raise RuntimeError(f"ffmpeg exited with non-zero status {ff_code}")


@app.command()
@run_coro
async def main(
    out_dir: Path,
    format: OutFormat = typer.Option(OutFormat.GIF, "--format", "-f"),
    artist_ids: List[int] = typer.Option(list, "--artist-id", "-a"),
    illust_ids: List[int] = typer.Option(list, "--illust-id", "-i"),
) -> None:
    out_dir.mkdir(exist_ok=True)

    async with httpx.AsyncClient(timeout=None) as client:
        illust_ids = stream.merge(
            stream.iterate(illust_ids),
            *[get_artist_illustrations(client, artist_id) for artist_id in artist_ids],
        )

        tasks = []
        async with illust_ids.stream() as illust_ids:
            async for illust_id in illust_ids:
                task = asyncio.create_task(
                    process_illust(client, format, illust_id, out_dir)
                )
                tasks.append((illust_id, task))
                typer.echo(f"Started processing {illust_id}")

        for illust_id, task in tasks:
            try:
                await task
            except Exception:
                typer.echo(
                    f"Could not fetch {illust_id}:\n{traceback.format_exc()}",
                    err=True,
                )
            else:
                typer.echo(
                    f"Finished processing {illust_id}",
                )
