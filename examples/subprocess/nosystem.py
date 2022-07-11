import subprocess
from typing import List


def echo_hello() -> None:
    subprocess.run(["echo", "hello"], check=True)


def ls(path: str, args: str = "") -> List[str]:
    popen_args = ["ls", path]
    if len(args) > 0:
        popen_args.append(args)

    with subprocess.Popen(popen_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as pipe:
        stdout, stderr = pipe.communicate()

    if len(stderr):
        raise RuntimeError(f"Command {popen_args} failed: {stderr}")

    return [p for p in stdout.decode("utf-8").split("\n") if len(p) > 0]
