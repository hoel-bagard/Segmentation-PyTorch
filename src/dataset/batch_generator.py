import multiprocessing as mp
from multiprocessing import shared_memory
import os
from typing import (
    Callable,
    Optional,
    Tuple
)

import numpy as np


class BatchGenerator:

    def __init__(self, data: np.ndarray, label: np.ndarray, batch_size: int,
                 num_workers: int,
                 data_preprocessing_fn: Optional[Callable] = None, label_preprocessing_fn: Optional[Callable] = None,
                 shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.data = data
        self.label = label

        # TODO: Add possibility to save dataset as hdf5
        self.data_preprocessing_fn = data_preprocessing_fn
        self.label_preprocessing_fn = label_preprocessing_fn

        self.dataset_size = len(self.data)
        self.index_list = np.arange(self.dataset_size)
        if shuffle:
            np.random.shuffle(self.index_list)

        # TODO: add possibility to drop last batch
        self.step_per_epoch = (self.dataset_size + (batch_size-1)) // self.batch_size
        self.last_batch_size = len(self.index_list) % self.batch_size
        if self.last_batch_size == 0:
            self.last_batch_size = self.batch_size

        self.epoch = 0
        self.step = 0

        first_data = np.array([data_processor(entry) if data_processor else entry
                               for entry in self.data[self.index_list[:batch_size]]])
        first_label = np.array([label_processor(entry) if label_processor else entry
                                for entry in self.label[self.index_list[:batch_size]]])
        self.batch_data = first_data
        self.batch_label = first_label

        self.process_id = 'NA'
        if self.prefetch or self.num_workers > 1:
            self.cache_memory_indices = shared_memory.SharedMemory(create=True, size=self.index_list.nbytes)
            self.cache_indices = np.ndarray(
                self.index_list.shape, dtype=self.index_list.dtype, buffer=self.cache_memory_indices.buf)
            self.cache_indices[:] = self.index_list
            self.cache_memory_data = [
                shared_memory.SharedMemory(create=True, size=first_data.nbytes),
                shared_memory.SharedMemory(create=True, size=first_data.nbytes)]
            self.cache_data = [
                np.ndarray(first_data.shape, dtype=first_data.dtype, buffer=self.cache_memory_data[0].buf),
                np.ndarray(first_data.shape, dtype=first_data.dtype, buffer=self.cache_memory_data[1].buf)]
            self.cache_memory_label = [
                shared_memory.SharedMemory(create=True, size=first_label.nbytes),
                shared_memory.SharedMemory(create=True, size=first_label.nbytes)]
            self.cache_label = [
                np.ndarray(first_label.shape, dtype=first_label.dtype, buffer=self.cache_memory_label[0].buf),
                np.ndarray(first_label.shape, dtype=first_label.dtype, buffer=self.cache_memory_label[1].buf)]
        else:
            self.cache_memory_indices = None
            self.cache_data = [first_data]
            self.cache_label = [first_label]

        if self.prefetch:
            self.prefetch_pipe_parent, self.prefetch_pipe_child = mp.Pipe()
            self.prefetch_stop = shared_memory.SharedMemory(create=True, size=1)
            self.prefetch_stop.buf[0] = 0
            self.prefetch_skip = shared_memory.SharedMemory(create=True, size=1)
            self.prefetch_skip.buf[0] = 0
            self.prefetch_process = mp.Process(target=self._prefetch_worker)
            self.prefetch_process.start()
            self.num_workers = 0
        self._init_workers()
        self.current_cache = 0
        self.process_id = 'main'

    def _init_workers(self):
        if self.num_workers > 1:
            self.worker_stop = shared_memory.SharedMemory(create=True, size=1)
            self.worker_stop.buf[0] = 0
            self.worker_pipes = []
            self.worker_processes = []
            for _ in range(self.num_workers):
                self.worker_pipes.append(mp.Pipe())
            for worker_index in range(self.num_workers):
                self.worker_processes.append(mp.Process(target=self._worker, args=(worker_index,)))
                self.worker_processes[-1].start()

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.release()

    def release(self):
        """Terminates all subprocesses and releases ressources"""
        pass

    def _worker(self, worker_index: int):
        self.process_id = f'worker_{worker_index}'
        self.num_workers = 0
        self.current_cache = 0
        parent_cache_data = self.cache_data
        parent_cache_label = self.cache_label
        cache_data = np.ndarray(self.cache_data[0].shape, dtype=self.cache_data[0].dtype)
        cache_label = np.ndarray(self.cache_label[0].shape, dtype=self.cache_label[0].dtype)
        self.cache_data = [cache_data]
        self.cache_label = [cache_label]
        self.index_list[:] = self.cache_indices
        pipe = self.worker_pipes[worker_index][1]

        while self.worker_stop.buf is not None and self.worker_stop.buf[0] == 0:
            try:
                current_cache, batch_index, start_index, self.batch_size = pipe.recv()
                if self.batch_size == 0:
                    continue
                self.index_list = self.cache_indices[start_index:start_index + self.batch_size].copy()

                self.cache_data[0] = cache_data[:self.batch_size]
                self.cache_label[0] = cache_label[:self.batch_size]
                self._next_batch()
                parent_cache_data[current_cache][batch_index:batch_index + self.batch_size] = self.cache_data[
                    self.current_cache][:self.batch_size]
                parent_cache_label[current_cache][batch_index:batch_index + self.batch_size] = self.cache_label[
                    self.current_cache][:self.batch_size]
                pipe.send(True)
            except KeyboardInterrupt:
                break
            except ValueError:
                break

    def _worker_next_batch(self):
        index_list = self.index_list[self.step * self.batch_size:(self.step + 1) * self.batch_size]
        batch_len = len(index_list)
        indices_per_worker = batch_len // self.num_workers
        if indices_per_worker == 0:
            indices_per_worker = 1

        worker_params = []
        batch_index = 0
        start_index = self.step * self.batch_size
        for _worker_index in range(self.num_workers - 1):
            worker_params.append([self.current_cache, batch_index, start_index, indices_per_worker])
            batch_index += indices_per_worker
            start_index += indices_per_worker
        worker_params.append([
            self.current_cache, batch_index, start_index,
            batch_len - ((self.num_workers - 1) * indices_per_worker)])

        if indices_per_worker == 1 and batch_len < self.num_workers:
            worker_params = worker_params[:batch_len]

        for params, (pipe, _) in zip(worker_params, self.worker_pipes):
            pipe.send(params)
        for _, (pipe, _) in zip(worker_params, self.worker_pipes):
            pipe.recv()

    def next_bath(self):
        pass


if __name__ == '__main__':
    def test():
        data = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.uint8)
        label = np.array(
            [.1, .2, .3, .4, .5, .6, .7, .8, .9, .10, .11, .12, .13, .14, .15, .16, .17, .18], dtype=np.uint8)

        for data_processor in [None, lambda x:x]:
            for prefetch in [True, False]:
                for num_workers in [3, 1]:
                    print(f'{data_processor=} {prefetch=} {num_workers=}')
                    with BatchGenerator(data, label, 5, data_processor=data_processor,
                                        prefetch=prefetch, num_workers=num_workers) as batch_generator:
                        for _ in range(19):
                            print(batch_generator.batch_data, batch_generator.epoch, batch_generator.step)
                            batch_generator.next_batch()
                        raise KeyboardInterrupt
                    print()

    test()
