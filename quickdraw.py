import os, random, pdb
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import struct
from struct import unpack

def unpack_drawing(file_handle):
    '''
    Utility function from QuickDraw project. Originally taken from ..
    https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py
    '''
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))
    
    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }

class QuickDraw(Dataset):

    # Two modes of working
    SKETCH = 0
    STROKE = 1

    def __init__(self, root, *, categories=[], max_samples=80000, normalize_xy=True, dtype=np.float32, verbose=False,
            cache=None, # not to be used
            filter_func=None, # subset sketches based on a function
            mode=SKETCH, # default in sketch mode
            seperate_p_tensor=False,
            shifted_seq_as_supevision=False
        ):
        super().__init__()

        # Track the parameters
        if os.path.exists(root):
            self.root = root
        
        if len(categories) == 0:
            self.categories = os.listdir(self.root)
        else:
            self.categories = [cat + '.bin' for cat in categories]
        
        self.normalize_xy = normalize_xy
        self.dtype = dtype
        self.verbose = verbose
        self.max_samples = max_samples
        if filter_func == None:
            self.filter_func = lambda s: (True, s) # passes every sketch
        else:
            self.filter_func = filter_func
        self.mode = mode
        self.seperate_p_tensor = seperate_p_tensor
        self.shifted_seq_as_supevision = shifted_seq_as_supevision

        # The cached data
        if cache != None:
            self.cache = cache
        else:
            self.cache = []
            for cat_idx, category in enumerate(self.categories):
                bin_file_path = os.path.join(self.root, category)
                n_samples = 0
                with open(bin_file_path, 'rb') as file:
                    while True:
                        try:
                            drawing = unpack_drawing(file)['image']
                            # breakpoint()

                            # Passes all sketches/strokes through 'filter_func'. It returns either
                            # (True, modified_sketch) OR (False, <anything>). If the first return
                            # object is True, it adds the 'modified_sketch' into cache. If False,
                            # it just skips that sample (the 2nd argument is useless then).
                            if self.mode == QuickDraw.SKETCH:
                                filter_check, _drawing = self.filter_func(drawing)
                                if filter_check:
                                    drawing = _drawing
                                else:
                                    continue
                                
                                # Append the whole sketch (with category ID)
                                self.cache.append((drawing, cat_idx))
                                n_samples += 1

                            elif self.mode == QuickDraw.STROKE:
                                stroke_drawings = [([d,], cat_idx) for d in drawing]
                                for j, (sd, _) in enumerate(stroke_drawings):
                                    filter_check, _sd = self.filter_func(sd)
                                    if filter_check:
                                        stroke_drawings[j] = (_sd, cat_idx)
                                    else:
                                        continue

                                # Append the stroke sketchs (with category ID each)
                                self.cache.extend(stroke_drawings)
                                n_samples += len(stroke_drawings)
                            
                            if n_samples >= max_samples:
                                break
                        except struct.error:
                            break
                if self.verbose:
                    print('[Info] {} sketches/strokes read from {}'.format(n_samples, bin_file_path))

            random.shuffle(self.cache)

        if self.verbose:
            print('[Info] Dataset with {} samples created'.format(len(self)))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i_sketch):
        qd_sketch, c_id = self.cache[i_sketch]
        n_strokes = len(qd_sketch) # sketch contains 'n_strokes' strokes
        sketch = np.empty((0, 3), dtype=self.dtype) # 2 for X-Y pair, 1 for pen state P

        for i_stroke in range(n_strokes):
            stroke = np.array(qd_sketch[i_stroke], dtype=self.dtype).T

            # The pen states. Only the one at the end of stroke has 1, rest 0
            p = np.zeros((stroke.shape[0], 1), dtype=self.dtype); p[-1, 0] = 1.
            
            # stack up strokes to make sketch
            sketch = np.vstack((sketch, np.hstack((stroke, p))))

        if self.normalize_xy:
            norm_factor = np.sqrt((sketch[:,:2]**2).sum(1)).max()
            sketch[:,:2] = sketch[:,:2] / (norm_factor + np.finfo(self.dtype).eps)

        if self.seperate_p_tensor:
            if self.shifted_seq_as_supevision:
                return (sketch[:-1,:-1], sketch[:-1,-1]), (sketch[1:,:-1], sketch[1:,-1]), c_id
            else:
                return (sketch[:,:-1], sketch[:,-1]), c_id
        else:
            if self.shifted_seq_as_supevision:
                return sketch[:-1,:], sketch[1:,:], c_id
            else:
                return sketch, c_id

    def collate(self, batch):
        if self.seperate_p_tensor:
            if self.shifted_seq_as_supevision:
                lengths = torch.tensor([x.shape[0] for (x, _), (_, _), _ in batch])
                padded_seq_inp = pad_sequence([torch.tensor(x) for (x, _), (_, _), _ in batch])
                padded_seq_inp_p = pad_sequence([torch.tensor(p) for (_, p), (_, _), _ in batch])
                padded_seq_shifted = pad_sequence([torch.tensor(x) for (_, _), (x, _), _ in batch])
                padded_seq_shifted_p = pad_sequence([torch.tensor(p) for (_, _), (_, p), _ in batch])
                labels = torch.tensor([c for (_, _), (_, _), c in batch])
                return (pack_padded_sequence(padded_seq_inp, lengths, enforce_sorted=False),\
                        pack_padded_sequence(padded_seq_inp_p, lengths, enforce_sorted=False)),\
                       (pack_padded_sequence(padded_seq_shifted, lengths, enforce_sorted=False), \
                        pack_padded_sequence(padded_seq_shifted_p, lengths, enforce_sorted=False)), labels
            else:
                lengths = torch.tensor([x.shape[0] for (x, _), _ in batch])
                padded_seq_inp = pad_sequence([torch.tensor(x) for (x, _), _ in batch])
                padded_seq_inp_p = pad_sequence([torch.tensor(p) for (_, p), _ in batch])
                labels = torch.tensor([c for (_, _), c in batch])
                return (pack_padded_sequence(padded_seq_inp, lengths, enforce_sorted=False),\
                        pack_padded_sequence(padded_seq_inp_p, lengths, enforce_sorted=False)), labels
        else:
            if self.shifted_seq_as_supevision:
                lengths = torch.tensor([x.shape[0] for (x, _, _) in batch])
                padded_seq_inp = pad_sequence([torch.tensor(x) for (x, _, _) in batch])
                padded_seq_shifted = pad_sequence([torch.tensor(x) for (_, x, _) in batch])
                labels = torch.tensor([c for (_, _, c) in batch])
                return pack_padded_sequence(padded_seq_inp, lengths, enforce_sorted=False),\
                        pack_padded_sequence(padded_seq_shifted, lengths, enforce_sorted=False), labels
            else:
                lengths = torch.tensor([x.shape[0] for (x, _) in batch])
                padded_seq_inp = pad_sequence([torch.tensor(x) for (x, _) in batch])
                labels = torch.tensor([c for (_, c) in batch])
                return pack_padded_sequence(padded_seq_inp, lengths, enforce_sorted=False), labels

    def get_dataloader(self, batch_size, shuffle = True, pin_memory = True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate, shuffle=shuffle, pin_memory=pin_memory)

    def split(self, proportion=0.8):
        train_samples = int(len(self) * proportion)
        qd_test = QuickDraw(self.root, categories=self.categories, max_samples=self.max_samples, normalize_xy=self.normalize_xy,
            dtype=self.dtype, verbose=self.verbose, filter_func=self.filter_func, mode=self.mode,
            seperate_p_tensor=self.seperate_p_tensor, shifted_seq_as_supevision=self.shifted_seq_as_supevision,
            cache=self.cache[train_samples:])
        self.cache = self.cache[:train_samples]

        if self.verbose:
            print('[Info] Dataset with {} samples created'.format(len(self)))
        
        return self, qd_test

if __name__ == '__main__':
    import sys

    qd = QuickDraw(sys.argv[1], categories=['airplane', 'bus'], max_samples=2, verbose=True, mode=QuickDraw.SKETCH,
        seperate_p_tensor=False, shifted_seq_as_supevision=True)
    qdl = qd.get_dataloader(4)
    for X, Y, _ in qdl:
        breakpoint()