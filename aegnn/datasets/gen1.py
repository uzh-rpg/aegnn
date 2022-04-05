import glob
import logging
import os
import numpy as np
import pickle
import sys
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from tqdm import tqdm
from typing import Callable, List, Optional

from aegnn.utils.multiprocessing import TaskManager
from .utils.normalization import normalize_time
from .base.event_dm import EventDataModule


class Gen1(EventDataModule):

    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 8, pin_memory: bool = False,
                 transform: Optional[Callable[[Data], Data]] = None):
        super(Gen1, self).__init__(img_shape=(304, 240), batch_size=batch_size, shuffle=shuffle,
                                   num_workers=num_workers, pin_memory=pin_memory, transform=transform)
        pre_processing_params = {"r": 3.0, "d_max": 128, "n_samples": 25000, "sampling": True}
        self.save_hyperparameters({"preprocessing": pre_processing_params})

    def _prepare_dataset(self, mode: str):
        raw_files = self.raw_files(mode)
        logging.debug(f"Found {len(raw_files)} raw files in dataset (mode = {mode})")

        total_bbox_count = self._total_bbox_count(raw_files)
        logging.debug(f"Total count of (filtered) bounding boxes = {total_bbox_count}")

        task_manager = TaskManager(self.num_workers, queue_size=self.num_workers)
        for rf in tqdm(raw_files):
            task_manager.queue(self._processing, rf=rf, root=self.root, read_annotations=self._read_annotations,
                               read_label=self._read_label, buffer_to_data=self._buffer_to_data)
        task_manager.join()

    def _processing(self, rf: str, root: str, read_annotations, read_label, buffer_to_data):
        data_loader = PSEELoader(rf)
        params = self.hparams.preprocessing
        bounding_boxes = read_annotations(rf)
        labels = np.array(read_label(bounding_boxes))

        for i, bbox in enumerate(bounding_boxes):
            processed_dir = os.path.join(root, "processed")
            processed_file = rf.replace(root, processed_dir).replace(".dat", f"{i}.pkl")
            if os.path.exists(processed_file):
                continue

            # Determine temporal window around the current bounding box [t_start, t_end]. Add all of the
            # bounding boxes within this window to the sample.
            sample_dict = dict()

            t_bbox = bbox[0]
            t_start = t_bbox - 100000  # 100 ms
            t_end = t_bbox + 300000  # 300 ms
            bbox_mask = np.logical_and(t_start < bounding_boxes['ts'], bounding_boxes['ts'] < t_end)

            sample_dict['bbox'] = torch.tensor(bounding_boxes[bbox_mask].tolist())
            sample_dict['label'] = labels[bbox_mask]
            sample_dict['raw_file'] = rf

            # Load raw data around bounding box.
            idx_start = data_loader.seek_time(t_start)
            data = data_loader.load_delta_t(t_end - t_start)
            sample_dict['raw'] = (idx_start, data.size)  # offset and number of events
            if data.size < 4000:
                continue

            # Normalize data to same number of events per sample. Therefore, either use subsampling to get
            # a larger temporal window or do not to get a more connected graph.
            num_samples = params["n_samples"]
            if data.size <= num_samples:
                sample_idx = np.arange(data.size)
            elif params["sampling"]:  # sub-sampling -> random N events
                sample_idx = np.random.choice(np.arange(data.size), size=num_samples, replace=False)
            else:  # no sub-sampling -> first N events
                sample_idx = np.arange(num_samples)
            sample = buffer_to_data(data[sample_idx])
            sample_dict['sample_idx'] = sample_idx

            # Graph generation (edges w.r.t. sub-sampled graph).
            device = torch.device(torch.cuda.current_device())
            sample.pos[:, 2] = normalize_time(sample.pos[:, 2])
            edge_index = radius_graph(sample.pos.to(device), r=params["r"], max_num_neighbors=params["d_max"])
            sample_dict['edge_index'] = edge_index.cpu()

            # Store resulting dictionary in file, however, the data only contains the data necessary
            # to re-create the graph, not the raw data itself.
            os.makedirs(os.path.dirname(processed_file), exist_ok=True)
            with open(processed_file, 'wb') as f:
                pickle.dump(sample_dict, f)

    #########################################################################################################
    # Processing ############################################################################################
    #########################################################################################################
    @staticmethod
    def _read_annotations(raw_file: str, skip_ts=int(5e5), min_box_diagonal=60, min_box_side=20) -> np.ndarray:
        boxes = np.load(raw_file.replace("_td.dat", "_bbox.npy"))

        # Bounding box filtering to avoid dealing with too small or initial bounding boxes.
        # See https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/io/box_filtering.py
        # Format: (ts, x, y, w, h, class_id, confidence = 1, track_id)
        ts = boxes['ts']
        width = boxes['w']
        height = boxes['h']
        diagonal_square = width**2+height**2
        mask = (ts > skip_ts)*(diagonal_square >= min_box_diagonal**2)*(width >= min_box_side)*(height >= min_box_side)
        return boxes[mask]

    @staticmethod
    def _read_label(bounding_boxes: np.ndarray) -> List[str]:
        class_id = bounding_boxes['class_id'].tolist()
        label_dict = {0: "car", 1: "pedestrian"}
        return [label_dict.get(cid, None) for cid in class_id]

    @staticmethod
    def _buffer_to_data(buffer: np.ndarray, **kwargs) -> Data:
        x = torch.from_numpy(buffer['x'].astype(np.float32))
        y = torch.from_numpy(buffer['y'].astype(np.float32))
        t = torch.from_numpy(buffer['t'].astype(np.float32))
        p = torch.from_numpy(buffer['p'].astype(np.float32)).view(-1, 1)
        pos = torch.stack([x, y, t], dim=1)
        return Data(x=p, pos=pos, **kwargs)

    def _load_processed_file(self, f_path: str) -> Data:
        with open(f_path, 'rb') as f:
            data_dict = pickle.load(f)
        data_loader = PSEELoader(data_dict['raw_file'])

        raw_start, raw_num_events = data_dict['raw']
        data_loader.seek_event(raw_start)
        data = data_loader.load_n_events(raw_num_events)
        data = data[data_dict['sample_idx']]

        data = self._buffer_to_data(data, label=data_dict['label'], file_id=f_path)
        data.bbox = data_dict['bbox'][:, 1:6].long()  # (x, y, w, h, class_id)
        data.y = data.bbox[:, -1]
        data.pos[:, 2] = normalize_time(data.pos[:, 2])  # time normalization
        data.edge_index = data_dict['edge_index']
        return data

    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    def raw_files(self, mode: str) -> List[str]:
        return glob.glob(os.path.join(self.root, mode, "*_td.dat"))

    def processed_files(self, mode: str) -> List[str]:
        processed_dir = os.path.join(self.root, "processed")
        return glob.glob(os.path.join(processed_dir, mode, "*.pkl"))

    def _total_bbox_count(self, raw_files: List[str]) -> int:
        num_bbox = 0
        for rf in raw_files:
            bounding_boxes = self._read_annotations(rf)
            num_bbox += bounding_boxes.size
        return num_bbox

    @property
    def classes(self) -> List[str]:
        return ["car", "pedestrian"]


#########################################################################################################
# Gen1 specific loader function from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
#########################################################################################################
EV_TYPE = [('t', 'u4'), ('_', 'i4')]  # Event2D
EV_STRING = 'Event2D'


class PSEELoader(object):
    """
    PSEELoader loads a dat or npy file and stream events
    """

    def __init__(self, datfile):
        """
        ctor
        :param datfile: binary dat or npy file
        """
        self._extension = datfile.split('.')[-1]
        assert self._extension in ["dat", "npy"], 'input file path = {}'.format(datfile)
        self._file = open(datfile, "rb")
        self._start, self.ev_type, self._ev_size, self._size = _parse_header(self._file)
        assert self._ev_size != 0
        self._dtype = EV_TYPE

        self._decode_dtype = []
        for dtype in self._dtype:
            if dtype[0] == '_':
                self._decode_dtype += [('x', 'u2'), ('y', 'u2'), ('p', 'u1')]
            else:
                self._decode_dtype.append(dtype)

        # size
        self._file.seek(0, os.SEEK_END)
        self._end = self._file.tell()
        self._ev_count = (self._end - self._start) // self._ev_size
        self.done = False
        self._file.seek(self._start)
        # If the current time is t, it means that next event that will be loaded has a
        # timestamp superior or equal to t (event with timestamp exactly t is not loaded yet)
        self.current_time = 0
        self.duration_s = self.total_time() * 1e-6

    def reset(self):
        """reset at beginning of file"""
        self._file.seek(self._start)
        self.done = False
        self.current_time = 0

    def event_count(self):
        """
        getter on event_count
        :return:
        """
        return self._ev_count

    def get_size(self):
        """"(height, width) of the imager might be (None, None)"""
        return self._size

    def __repr__(self):
        """
        prints properties
        :return:
        """
        wrd = ''
        wrd += 'PSEELoader:' + '\n'
        wrd += '-----------' + '\n'
        if self._extension == 'dat':
            wrd += 'Event Type: ' + str(EV_STRING) + '\n'
        elif self._extension == 'npy':
            wrd += 'Event Type: numpy array element\n'
        wrd += 'Event Size: ' + str(self._ev_size) + ' bytes\n'
        wrd += 'Event Count: ' + str(self._ev_count) + '\n'
        wrd += 'Duration: ' + str(self.duration_s) + ' s \n'
        wrd += '-----------' + '\n'
        return wrd

    def load_n_events(self, ev_count):
        """
        load batch of n events
        :param ev_count: number of events that will be loaded
        :return: events
        Note that current time will be incremented to reach the timestamp of the first event not loaded yet
        """
        event_buffer = np.empty((ev_count + 1,), dtype=self._decode_dtype)

        pos = self._file.tell()
        count = (self._end - pos) // self._ev_size
        if ev_count >= count:
            self.done = True
            ev_count = count
            _stream_td_data(self._file, event_buffer, self._dtype, ev_count)
            self.current_time = event_buffer['t'][ev_count - 1] + 1
        else:
            _stream_td_data(self._file, event_buffer, self._dtype, ev_count + 1)
            self.current_time = event_buffer['t'][ev_count]
            self._file.seek(pos + ev_count * self._ev_size)

        return event_buffer[:ev_count]

    def load_delta_t(self, delta_t):
        """
        loads a slice of time.
        :param delta_t: (us) slice thickness
        :return: events
        Note that current time will be incremented by delta_t.
        If an event is timestamped at exactly current_time it will not be loaded.
        """
        if delta_t < 1:
            raise ValueError("load_delta_t(): delta_t must be at least 1 micro-second: {}".format(delta_t))

        if self.done or (self._file.tell() >= self._end):
            self.done = True
            return np.empty((0,), dtype=self._decode_dtype)

        final_time = self.current_time + delta_t
        tmp_time = self.current_time
        start = self._file.tell()
        pos = start
        nevs = 0
        batch = 100000
        event_buffer = []
        # data is read by buffers until enough events are read or until the end of the file
        while tmp_time < final_time and pos < self._end:
            count = (min(self._end, pos + batch * self._ev_size) - pos) // self._ev_size
            buffer = np.empty((count,), dtype=self._decode_dtype)
            _stream_td_data(self._file, buffer, self._dtype, count)
            tmp_time = buffer['t'][-1]
            event_buffer.append(buffer)
            nevs += count
            pos = self._file.tell()
        if tmp_time >= final_time:
            self.current_time = final_time
        else:
            self.current_time = tmp_time + 1
        assert len(event_buffer) > 0
        idx = np.searchsorted(event_buffer[-1]['t'], final_time)
        event_buffer[-1] = event_buffer[-1][:idx]
        event_buffer = np.concatenate(event_buffer)
        idx = len(event_buffer)
        self._file.seek(start + idx * self._ev_size)
        self.done = self._file.tell() >= self._end
        return event_buffer

    def seek_event(self, ev_count):
        """
        seek in the file by ev_count events
        :param ev_count: seek in the file after ev_count events
        Note that current time will be set to the timestamp of the next event.
        """
        if ev_count <= 0:
            self._file.seek(self._start)
            self.current_time = 0
        elif ev_count >= self._ev_count:
            # we put the cursor one event before and read the last event
            # which puts the file cursor at the right place
            # current_time is set to the last event timestamp + 1
            self._file.seek(self._start + (self._ev_count - 1) * self._ev_size)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0] + 1
        else:
            # we put the cursor at the *ev_count*nth event
            self._file.seek(self._start + ev_count * self._ev_size)
            # we read the timestamp of the following event (this change the position in the file)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]
            # this is why we go back at the right position here
            self._file.seek(self._start + ev_count * self._ev_size)
        self.done = self._file.tell() >= self._end

    def seek_time(self, final_time, term_criterion: int = 100000):
        """
        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_criterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        if final_time > self.total_time():
            self._file.seek(self._end)
            self.done = True
            self.current_time = self.total_time() + 1
            return

        if final_time <= 0:
            self.reset()
            return

        low = 0
        high = self._ev_count

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2

            self.seek_event(middle)
            mid = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]

            if mid > final_time:
                high = middle
            elif mid < final_time:
                low = middle + 1
            else:
                self.current_time = final_time
                self.done = self._file.tell() >= self._end
                return
        # we now know that it is between low and high
        self.seek_event(low)
        final_buffer = np.fromfile(self._file, dtype=self._dtype, count=high - low)['t']
        final_index = np.searchsorted(final_buffer, final_time)

        self.seek_event(low + final_index)
        self.current_time = final_time
        self.done = self._file.tell() >= self._end
        return low + final_index

    def total_time(self):
        """
        get total duration of video in mus, providing there is no overflow
        :return:
        """
        if not self._ev_count:
            return 0
        # save the state of the class
        pos = self._file.tell()
        current_time = self.current_time
        done = self.done
        # read the last event's timestamp
        self.seek_event(self._ev_count - 1)
        time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]
        # restore the state
        self._file.seek(pos)
        self.current_time = current_time
        self.done = done

        return time

    def __del__(self):
        self._file.close()


def _parse_header(f):
    """
    Parses the header of a dat file
    Args:
        - f file handle to a dat file
    return :
        - int position of the file cursor after the header
        - int type of event
        - int size of event in bytes
        - size (height, width) tuple of int or None
    """
    f.seek(0, os.SEEK_SET)
    bod = None
    end_of_header = False
    header = []
    num_comment_line = 0
    size = [None, None]
    # parse header
    while not end_of_header:
        bod = f.tell()
        line = f.readline()
        if sys.version_info > (3, 0):
            first_item = line.decode("latin-1")[:2]
        else:
            first_item = line[:2]

        if first_item != '% ':
            end_of_header = True
        else:
            words = line.split()
            if len(words) > 1:
                if words[1] == 'Date':
                    header += ['Date', words[2] + ' ' + words[3]]
                if words[1] == 'Height' or words[1] == b'Height':  # compliant with python 3 (and python2)
                    size[0] = int(words[2])
                    header += ['Height', words[2]]
                if words[1] == 'Width' or words[1] == b'Width':  # compliant with python 3 (and python2)
                    size[1] = int(words[2])
                    header += ['Width', words[2]]
            else:
                header += words[1:3]
            num_comment_line += 1
    # parse data
    f.seek(bod, os.SEEK_SET)

    if num_comment_line > 0:  # Ensure compatibility with previous files.
        # Read event type
        ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
        # Read event size
        ev_size = np.frombuffer(f.read(1), dtype=np.uint8)[0]
    else:
        ev_type = 0
        ev_size = sum([int(n[-1]) for _, n in EV_TYPE])

    bod = f.tell()
    return bod, ev_type, ev_size, size


def _stream_td_data(file_handle, buffer, dtype, ev_count=-1):
    """
    Streams data from opened file_handle
    args :
        - file_handle: file object
        - buffer: pre-allocated buffer to fill with events
        - dtype:  expected fields
        - ev_count: number of events
    """

    dat = np.fromfile(file_handle, dtype=dtype, count=ev_count)
    count = len(dat['t'])
    for name, _ in dtype:
        if name == '_':
            buffer['x'][:count] = np.bitwise_and(dat["_"], 16383)
            buffer['y'][:count] = np.right_shift(np.bitwise_and(dat["_"], 268419072), 14)
            buffer['p'][:count] = np.right_shift(np.bitwise_and(dat["_"], 268435456), 28)
        else:
            buffer[name][:count] = dat[name]
