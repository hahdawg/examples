import asyncio
import time


async def load_chunk(path):
    await asyncio.sleep(1.0)
    return f"loaded data from {path}"


async def load_data():
    print("In load_data")
    paths = map(str, list(range(20)))
    output = asyncio.gather(*[asyncio.create_task(load_chunk(p)) for p in paths])
    await output
    return output


def normal_fcn():
    return "normal-output"


def main():
    async_out = asyncio.run(load_data()).result()
    normal_out = normal_fcn()
    print(async_out)
    print(normal_out)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"elapsed = {time.time() - start_time:0.2f}")
