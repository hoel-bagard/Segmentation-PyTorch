import multiprocessing as mp
from multiprocessing import shared_memory
from typing import (
    Callable,
    Optional,
    Final,
    Any
)
from time import time
from math import ceil

import numpy as np


class BatchGenerator:

    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int,
                 nb_workers: int = 1,
                 data_preprocessing_fn: Optional[Callable[[np.ndarray], Any]] = None,
                 labels_preprocessing_fn: Optional[Callable[[np.ndarray], Any]] = None,
                 shuffle: bool = False):
        """
        Args:
            data: Numpy array with the data. It can be be only path to the datapoints to load (or other forms of data)
                  if the loading function is given as data_preprocessing_fn.
            labels: Numpy array with the labels, as for data it can be not yet fully ready labels.
            nb_workers: Number of workers to use for multiprocessing (>=1).
            data_preprocessing_fn: If not None, data will be passed through this function.
            labels_preprocessing_fn: If not None, labels will be passed through this function.
            shuffle: If True, then dataset is shuffled for each epoch
        """

        self.data: Final[np.ndarray] = data
        self.labels: Final[np.ndarray] = labels

        self.batch_size: Final[int] = batch_size
        self.shuffle: Final[bool] = shuffle
        self.nb_workers: Final[int] = nb_workers
        self.data_preprocessing_fn = data_preprocessing_fn
        self.labels_preprocessing_fn = labels_preprocessing_fn

        # TODOLIST
        # TODO: Add possibility to save dataset as hdf5
        # TODO: add possibility to drop last batch
        # TODO: Make an Iterator to allow enumerating the dataset

        self.nb_datapoints: Final[int] = len(self.data)

        index_list: np.ndarray = np.arange(self.nb_datapoints)
        data_batch: np.ndarray = np.asarray([data_preprocessing_fn(entry) if data_preprocessing_fn else entry
                                             for entry in data[:batch_size]])
        labels_batch: np.ndarray = np.asarray([labels_preprocessing_fn(entry) if data_preprocessing_fn else entry
                                               for entry in labels[:batch_size]])

        self.step_per_epoch = (self.nb_datapoints + (batch_size-1)) // self.batch_size
        self.last_batch_size = self.nb_datapoints % self.batch_size
        if self.last_batch_size == 0:
            self.last_batch_size = self.batch_size

        self.epoch = 0
        self.global_step = 0
        self.step = 0

        # Create shared memories for indices, data and labels.
        self.memories_released = mp.Event()
        # For data and labels, 2 memories / caches are required for prefetch to work.
        # (One for the main process to read from, one for the workers to write in)
        self.current_cache = 0
        # Indices
        self.cache_memory_indices = shared_memory.SharedMemory(create=True, size=index_list.nbytes)
        self.cache_indices = np.ndarray(self.nb_datapoints, dtype=int, buffer=self.cache_memory_indices.buf)
        self.cache_indices[:] = index_list
        # Data
        self.cache_memory_data = [
            shared_memory.SharedMemory(create=True, size=data_batch.nbytes),
            shared_memory.SharedMemory(create=True, size=data_batch.nbytes)]
        self.cache_data = [
            np.ndarray(data_batch.shape, dtype=data_batch.dtype, buffer=self.cache_memory_data[0].buf),
            np.ndarray(data_batch.shape, dtype=data_batch.dtype, buffer=self.cache_memory_data[1].buf)]
        # Labels
        self.cache_memory_labels = [
            shared_memory.SharedMemory(create=True, size=labels_batch.nbytes),
            shared_memory.SharedMemory(create=True, size=labels_batch.nbytes)]
        self.cache_labels = [
            np.ndarray(labels_batch.shape, dtype=labels_batch.dtype, buffer=self.cache_memory_labels[0].buf),
            np.ndarray(labels_batch.shape, dtype=labels_batch.dtype, buffer=self.cache_memory_labels[1].buf)]

        # Create workers
        self.process_id = "NA"
        self._init_workers()
        self.process_id = "main"

    def _init_workers(self):
        """Create workers and pipes / events used to communicate with them"""
        self.stop_event = mp.Event()
        self.worker_pipes = [mp.Pipe() for _ in range(self.nb_workers)]
        self.worker_processes = []
        for worker_index in range(self.nb_workers):
            self.worker_processes.append(mp.Process(target=self._worker_fn, args=(worker_index,)))
            self.worker_processes[-1].start()

    def _worker_fn(self, worker_index: int):
        self.process_id = f"worker_{worker_index}"
        pipe = self.worker_pipes[worker_index][1]

        while not self.stop_event.is_set():
            try:
                # Check if there is a message to be received. (prevents process from getting stuck)
                if pipe.poll(0.05):
                    current_cache, cache_start_index, indices_start_index, nb_elts = pipe.recv()
                else:
                    continue

                indices_to_process = self.cache_indices[indices_start_index:indices_start_index+nb_elts]

                # Get the data (and process it)
                if self.data_preprocessing_fn:  # TODO: one liner with # noqa:E501 ?
                    processed_data = self.data_preprocessing_fn(self.data[indices_to_process])
                else:
                    processed_data = self.data[indices_to_process]
                # Put the data into the shared memory
                self.cache_data[current_cache][cache_start_index:cache_start_index+nb_elts] = processed_data

                # Do the same for labels
                if self.labels_preprocessing_fn:
                    processed_labels = self.labels_preprocessing_fn(self.labels[indices_to_process])
                else:
                    processed_labels = self.labels[indices_to_process]
                self.cache_labels[current_cache][cache_start_index:cache_start_index+nb_elts] = processed_labels

                # Send signal to the main process to say that everything is ready
                pipe.send(True)
            except (KeyboardInterrupt, ValueError):
                break

    def next_batch(self):
        self.global_step += 1
        self.step += 1   # Step starts at 1

        # Check if the current epoch is finished. If it is then start a new one.
        if (self.step-1) * self.batch_size > self.nb_datapoints:
            self.epoch += 1
            self.step = 1
            np.random.shuffle(self.cache_indices)

        # Prepare arguments for workers and send them
        current_batch_size = self.batch_size if self.step != self.step_per_epoch else self.last_batch_size
        self.current_cache = (self.current_cache+1) % 2
        nb_elts_per_worker = current_batch_size // self.nb_workers
        for worker_index in range(self.nb_workers):
            cache_start_index = worker_index * nb_elts_per_worker
            indices_start_index = (self.step-1) * self.batch_size + cache_start_index
            nb_elts = nb_elts_per_worker if worker_index != (self.nb_workers-1) else ceil(current_batch_size / self.nb_workers)  # noqa:E501
            self.worker_pipes[worker_index][0].send((self.current_cache, cache_start_index, indices_start_index, nb_elts))  # noqa:E501

        # Wait for everyworker to have finished preparing its mini-batch
        for pipe, _ in self.worker_pipes:
            pipe.recv()

        data_batch = self.cache_data[self.current_cache][:current_batch_size]
        labels_batch = self.cache_data[self.current_cache][:current_batch_size]

        return data_batch, labels_batch

    def __del__(self):
        self.release()

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.release()

    def __enter__(self):
        return self

    def release(self):
        """Terminates all workers and releases all the shared ressources"""
        # Closes acces to the shared memories
        for shared_mem in self.cache_memory_data + self.cache_memory_labels + [self.cache_memory_indices]:
            shared_mem.close()

        if self.process_id == "main":
            self.stop_event.set()   # Sends signal to stop to all the workers

            # Terminates all the workers
            end_time = time() + 5  # Maximum waiting time (for all processes)
            for pipe, worker in zip(self.worker_pipes, self.worker_processes):
                pipe[0].close()
                worker.join(timeout=max(0.0, end_time - time()))
                self.worker_pipes.remove(pipe)
                self.worker_processes.remove(worker)

            if not self.memories_released.is_set():
                # Requests for all the shared memories to be destroyed
                print("Releasing shared memories")
                for shared_mem in self.cache_memory_data + self.cache_memory_labels + [self.cache_memory_indices]:
                    shared_mem.unlink()
                self.memories_released.set()


if __name__ == '__main__':
    # TODO: Make the function into an actual test with asserts
    def test():
        data = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.uint8)
        label = np.array(
            [.1, .2, .3, .4, .5, .6, .7, .8, .9, .10, .11, .12, .13, .14, .15, .16, .17, .18], dtype=np.uint8)

        data_preprocessing_fn = None

        for nb_workers in [2]:
            print(f'{nb_workers=}')
            with BatchGenerator(data, label, 5, data_preprocessing_fn=data_preprocessing_fn,
                                nb_workers=nb_workers) as batch_generator:
                for _ in range(19):
                    data_batch, labels_batch = batch_generator.next_batch()
                    # print(f"{batch_generator.epoch=}, {batch_generator.step=}")
                    print(f"{data_batch=}, {labels_batch=}")

    test()
