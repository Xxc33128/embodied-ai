"""Unix socket -> one stdio app-server per connection (Python 3.8+, Unix)."""
import argparse
import asyncio
import contextlib
import os
from pathlib import Path
import signal
import socket
import sys


async def copy_stream(reader, writer):
    while True:
        data = await reader.read(65536)
        if not data:
            break
        writer.write(data)
        await writer.drain()


def signal_group(proc, sig):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, sig)


async def relay(reader, writer, command):
    proc = None
    pumps = []
    try:
        proc = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, start_new_session=True)
        upload = asyncio.create_task(copy_stream(reader, proc.stdin))
        download = asyncio.create_task(copy_stream(proc.stdout, writer))
        pumps = [upload, download]
        done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        if upload in done:
            # Preserve a final response after the client half-closes its socket.
            proc.stdin.close()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(download, timeout=3)
    except (OSError, ConnectionError) as exc:
        print("bridge connection failed: {}".format(exc), file=sys.stderr)
    finally:
        for task in pumps:
            task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        if proc is not None:
            if proc.stdin is not None:
                proc.stdin.close()
            signal_group(proc, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                signal_group(proc, signal.SIGKILL)
                await proc.wait()
            # Also retire descendants if their parent exited first.
            signal_group(proc, signal.SIGKILL)
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


async def serve(path, command):
    if os.path.lexists(path):
        raise FileExistsError("socket path already exists: {}".format(path))
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise PermissionError("socket parent must be owned by you with mode 700")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server = None
    identity = None
    connections = set()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def accept(reader, writer):
        task = asyncio.create_task(relay(reader, writer, command))
        connections.add(task)
        task.add_done_callback(connections.discard)

    try:
        listener.bind(str(path))
        identity = path.stat().st_ino
        os.chmod(path, 0o600)
        listener.setblocking(False)
        server = await asyncio.start_unix_server(accept, sock=listener)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        print("READY {}".format(path), flush=True)
        await stop.wait()
    finally:
        if server is not None:
            server.close()
        listener.close()
        pending = list(connections)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # Python 3.12+ wait_closed also waits for client connections. Retire
        # the relays first so server shutdown cannot deadlock on its clients.
        if server is not None:
            await server.wait_closed()
        if identity is not None:
            with contextlib.suppress(FileNotFoundError):
                if path.lstat().st_ino == identity:
                    path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="after --: app-server executable and arguments")
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        command = ["codex", "app-server", "--listen", "stdio://"]
    try:
        asyncio.run(serve(args.socket.expanduser().absolute(), command))
    except OSError as exc:
        parser.exit(1, "{}\n".format(exc))


if __name__ == "__main__":
    main()
