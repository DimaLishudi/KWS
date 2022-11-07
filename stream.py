import logging
import multiprocessing as mp

import torch
from torchaudio.io import StreamReader

logger = logging.getLogger(__file__)


def audio_stream(queue: mp.Queue, frames_per_chunk):
    """
    Learn more about how to install and use streaming audio here
    https://pytorch.org/audio/stable/tutorials/streaming_api2_tutorial.html
    """

    streamer = StreamReader(src="hw:0", format="alsa") # src=":0", format=avfoundation
    print(frames_per_chunk)
    streamer.add_basic_audio_stream(frames_per_chunk=frames_per_chunk, sample_rate=16000)
    stream_iterator = streamer.stream(-1, 1)

    logger.info("Start audio streaming")
    while True:
        (chunk_,) = next(stream_iterator)
        logger.info("Put chunk to queue")
        queue.put(chunk_)


if __name__ == "__main__":
    model = torch.jit.load("streaming_model.pth").eval()

    ctx = mp.get_context("spawn")
    chunk_queue = ctx.Queue()
    streaming_process = ctx.Process(target=audio_stream, args=(chunk_queue, model.chunk_size))

    streaming_process.start()
    while True:
        try:
            chunk = chunk_queue.get().sum(dim=1)
            # chunk = chunk.view(1, -1)
            # print(f"{chunk.shape=}")

            with torch.inference_mode():
                result = model(chunk)


            if result > 0.6: # > 0.7
                print('ПРИВЕТ')
                #print("DETECTED KEY WORD")

        except KeyboardInterrupt:
            break
        except Exception as exc:
            raise exc

    streaming_process.join()
