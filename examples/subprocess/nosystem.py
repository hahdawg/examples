"""
Use subprocess to run commands and capture output instead of os. Both functions should mimic
os.system, but they capture the output as a string.
"""
import subprocess


def ls_run(path: str, args: str = "") -> str:
    """
    Use subprocess.run to do an ls command.
    """
    popen_args = ["ls", path]
    if len(args) > 0:
        popen_args.append(args)

    try:
        output = subprocess.run(popen_args, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # NOTE: We have to catch the error in order to log it. Otherwise, we just get
        # CalledProcessError, which doesn't say what happened.
        stderr = e.stderr.decode("utf-8")
        raise ValueError(f"Command {popen_args} failed: {stderr}") from e

    return output.stdout.decode("utf-8")


def ls_popen(path: str, args: str = "") -> str:
    """
    Use subprocess.popen to do an ls command.
    """
    popen_args = ["ls", path]
    if len(args) > 0:
        popen_args.append(args)

    with subprocess.Popen(popen_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as pipe:
        stdout, stderr = pipe.communicate()

    if len(stderr):
        raise ValueError(f"Command {popen_args} failed: {stderr}")

    return stdout.decode("utf-8")
