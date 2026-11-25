"""Exercise the real socket/child-process boundary without model calls."""
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest


LISTENER = Path(__file__).resolve().parents[1] / "tools/mac_bridge/listen.py"


class ListenerTests(unittest.TestCase):
    def setUp(self):
        # Short paths also fit macOS's Unix socket path limit.
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "bridge.sock")

    def start(self, child):
        proc = subprocess.Popen(
            [sys.executable, str(LISTENER), "--socket", self.path,
             "--", sys.executable, "-u", "-c", child],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(self.stop, proc)
        ready, _, _ = select.select([proc.stdout], [], [], 10)
        self.assertTrue(ready, "listener did not become ready")
        line = proc.stdout.readline()
        self.assertTrue(line.startswith(b"READY "), line)
        return proc

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()

    def connect(self):
        conn = socket.socket(socket.AF_UNIX)
        self.addCleanup(conn.close)
        conn.settimeout(10)
        conn.connect(self.path)
        return conn

    def test_large_fragmented_data_and_half_close_preserve_final_response(self):
        self.start("import sys; data=sys.stdin.buffer.read(); "
                   "sys.stdout.buffer.write(data[::-1]); sys.stdout.buffer.flush()")
        conn = self.connect()
        payload = ('双视角' * 50000).encode() + b'\n'
        for offset in range(0, len(payload), 7919):
            conn.sendall(payload[offset:offset + 7919])
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        self.assertEqual(b''.join(chunks), payload[::-1])

    def test_connections_have_separate_children_and_shutdown_reaps_them(self):
        proc = self.start("import os,time; print(os.getpid(), flush=True); time.sleep(60)")
        one, two = self.connect(), self.connect()
        pids = [int(one.recv(100)), int(two.recv(100))]
        self.assertNotEqual(*pids)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
        self.assertFalse(os.path.exists(self.path))
        for pid in pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_disconnect_reaps_child_and_listener_accepts_again(self):
        self.start("import os,time; print(os.getpid(), flush=True); time.sleep(60)")
        conn = self.connect()
        pid = int(conn.recv(100))
        conn.close()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("disconnected child is still alive")
        self.assertNotEqual(int(self.connect().recv(100)), pid)

    def test_existing_file_is_preserved(self):
        Path(self.path).write_text("keep me")
        proc = subprocess.run(
            [sys.executable, str(LISTENER), "--socket", self.path],
            capture_output=True, timeout=10)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"already exists", proc.stderr)
        self.assertEqual(Path(self.path).read_text(), "keep me")

    def test_server_stdio_bridge_round_trip(self):
        self.start("import sys;\nfor line in sys.stdin.buffer: "
                   "sys.stdout.buffer.write(line); sys.stdout.buffer.flush()")
        bridge = subprocess.Popen(
            [sys.executable, str(LISTENER.with_name('stdio_socket.py')), self.path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(self.stop, bridge)
        message = json.dumps({"id": 1, "method": "initialize"}).encode() + b'\n'
        bridge.stdin.write(message)
        bridge.stdin.flush()
        ready, _, _ = select.select([bridge.stdout], [], [], 10)
        self.assertTrue(ready)
        self.assertEqual(bridge.stdout.readline(), message)


if __name__ == '__main__':
    unittest.main()
