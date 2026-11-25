"""Bridge newline JSON-RPC stdio to a local Unix socket; no credentials."""
import socket
import os
import sys
import threading


def main():
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(sys.argv[1])

        def incoming():
            try:
                while data := client.recv(65536):
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
            finally:
                os._exit(0)

        threading.Thread(target=incoming, daemon=True).start()
        while data := sys.stdin.buffer.read1(65536):
            client.sendall(data)


if __name__ == "__main__":
    main()
