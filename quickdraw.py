import ndjson
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

    FULLSEQ = 0
    ENCDEC = 1
    STROKESET = 2

    def __init__(self, root, *, categories=[], normalize_xy=True, start_from_zero=False, dtype=np.float32, verbose=False,
            npz=False, raw=False,
            max_sketches_each_cat=1000, # maximum sketches from each category
            cache=None, # not to be used
            filter_func=None, # subset sketches based on a function
            mode=SKETCH, # default in sketch mode
            problem=FULLSEQ # organisation of returned data
        ):
        super().__init__()

        # Track the parameters
        if os.path.exists(root):
            self.root = root
        
        if len(categories) == 0:
            self.categories = os.listdir(self.root)
        else:
            if npz:
                self.categories = [cat + '.npz' for cat in categories]
            elif raw:
                self.categories = [cat + '.ndjson' for cat in categories]
            else:
                self.categories = [cat + '.bin' for cat in categories]
        
        self.normalize_xy = normalize_xy
        self.start_from_zero = start_from_zero
        self.dtype = dtype
        self.verbose = verbose
        self.npz = npz
        self.ndjson = raw
        self.max_sketches_each_cat = max_sketches_each_cat
        self.max_sketches = self.max_sketches_each_cat * len(self.categories)
        if filter_func == None:
            self.filter_func = lambda s: (True, s) # passes every sketch
        else:
            self.filter_func = filter_func
        self.mode = mode
        self.problem = problem
        # self.seperate_p_tensor = seperate_p_tensor
        # self.shifted_seq_as_supevision = shifted_seq_as_supevision

        # The cached data
        if cache != None:
            self.cache = cache
        else:
            self.cache = []
            n_sketches = 0
            for cat_idx, category in enumerate(self.categories):
                bin_file_path = os.path.join(self.root, category)
                n_sketches_each_cat = 0
                with open(bin_file_path, 'rb') as file:
                    if self.npz:
                        file = np.load(bin_file_path, allow_pickle=True, encoding='bytes')['train']
                    if self.ndjson:
                        with open(bin_file_path, 'r') as f:
                            J = ndjson.load(f)

                    while True:
                        try:
                            if self.npz:
                                drawing = []
                                
                                # acquire the next sketch
                                sketch = file[0]
                                file = file[1:] # replace the original structure by the rest of it
                                sketch[:,:2] = np.cumsum(sketch[:,:2], axis=0)

                                # range normalization
                                xmin, xmax = sketch[:,0].min(), sketch[:,0].max()
                                ymin, ymax = sketch[:,1].min(), sketch[:,1].max()

                                sketch[:,0] = ((sketch[:,0] - xmin) / float(xmax - xmin)) * 255.
                                sketch[:,1] = ((sketch[:,1] - ymin) / float(ymax - ymin)) * 255.

                                sketch = sketch.astype(np.int64)

                                stroke_list = np.split(sketch[:,:2], np.where(sketch[:,2])[0] + 1, axis=0)[:-1]
                                for stroke in stroke_list:
                                    drawing.append(stroke.T.tolist())
                            elif self.ndjson:
                                drawing = []

                                sketch = J[0]['drawing']
                                J = J[1:] # replace the original structure by the rest of it
                                sketch = [np.array(s[:-1]).T for s in sketch]

                                xmin = min([stroke[:,0].min() for stroke in sketch])
                                xmax = max([stroke[:,0].max() for stroke in sketch])
                                ymin = min([stroke[:,1].min() for stroke in sketch])
                                ymax = max([stroke[:,1].max() for stroke in sketch])

                                for i_stroke in range(len(sketch)):
                                    # breakpoint()
                                    sketch[i_stroke][:,0] = ((sketch[i_stroke][:,0] - xmin) / float(xmax - xmin)) * 255.
                                    sketch[i_stroke][:,1] = 255. - (((sketch[i_stroke][:,1] - ymin) / float(ymax - ymin)) * 255.)
                                    sketch[i_stroke] = sketch[i_stroke].astype(np.int64)

                                for stroke in sketch:
                                    drawing.append(stroke.T.tolist())
                            else:
                                drawing = unpack_drawing(file)['image']
                        
                        except (struct.error, IndexError) as e:
                            break

                        # Passes all sketches/strokes through 'filter_func'. It returns either
                        # (True, modified_sketch) OR (False, <anything>). If the first return
                        # object is True, it adds the 'modified_sketch' into cache. If False,
                        # it just skips that sample (the 2nd argument is useless then).
                        if self.mode == QuickDraw.SKETCH or self.mode == QuickDraw.STROKESET:
                            filter_check, _drawing = self.filter_func(drawing)
                            if filter_check:
                                drawing = _drawing
                            else:
                                continue
                            
                            # Append the whole sketch (with category ID)
                            # breakpoint()
                            self.cache.append((drawing, cat_idx))

                        elif self.mode == QuickDraw.STROKE:
                            stroke_drawings = [([d,], cat_idx) for d in drawing]
                            for j, (sd, _) in enumerate(stroke_drawings):
                                filter_check, _sd = self.filter_func(sd)
                                if filter_check:
                                    stroke_drawings[j] = (_sd, cat_idx)
                                else:
                                    continue

                            # Append the stroke sketchs (with category ID each)
                            # breakpoint()
                            self.cache.extend(stroke_drawings)
                        
                        n_sketches += 1
                        n_sketches_each_cat += 1
                        
                        # enough sketches collected per category
                        if n_sketches_each_cat >= self.max_sketches_each_cat:
                            break
                    
                    if self.verbose:
                        print('[Info] {} sketches read from {}'.format(n_sketches_each_cat, bin_file_path))
                
                # enough sketches collected
                if n_sketches >= self.max_sketches:
                    break
                
            random.shuffle(self.cache)

        if self.verbose:
            print('[Info] Dataset with {} samples created'.format(len(self)))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i_sketch):
        qd_sketch, c_id = self.cache[i_sketch]
        n_strokes = len(qd_sketch) # sketch contains 'n_strokes' strokes
        if self.mode != QuickDraw.STROKESET:
            sketch = np.empty((0, 3), dtype=self.dtype) # 2 for X-Y pair, 1 for pen state P
        else:
            sketch = []

        for i_stroke in range(n_strokes):
            stroke = np.array(qd_sketch[i_stroke], dtype=self.dtype).T

            # The pen states. Only the one at the end of stroke has 1, rest 0
            p = np.zeros((stroke.shape[0], 1), dtype=self.dtype); p[-1, 0] = 1.
            
            if self.mode != QuickDraw.STROKESET:
                # stack up strokes to make sketch
                sketch = np.vstack((sketch, np.hstack((stroke, p))))
            else:
                sketch.append(np.hstack((stroke, p)))

        if self.normalize_xy:
            if self.mode != QuickDraw.STROKESET:
                norm_factor = np.sqrt((sketch[:,:2]**2).sum(1)).max()
                sketch[:,:2] = sketch[:,:2] / (norm_factor + np.finfo(self.dtype).eps)
            else:
                norm_factor = 0
                for stroke in sketch:
                    stroke_max_mag = np.sqrt((stroke[:,:2]**2).sum(1)).max()
                    # breakpoint()
                    if stroke_max_mag > norm_factor:
                        norm_factor = stroke_max_mag
                for j in range(len(sketch)):
                    sketch[j][:,:2] = sketch[j][:,:2] / (norm_factor + np.finfo(self.dtype).eps)

        if self.start_from_zero:
            if self.mode != QuickDraw.STROKESET:
                sketch[:,:2] -= sketch[0,:2]
            else:
                first_xy = sketch[0][0,:2]
                for j in range(len(sketch)):
                    sketch[j][:,:2] = sketch[j][:,:2] - first_xy

        if self.mode == QuickDraw.STROKESET:
            # For now, ignore 'problem' if 'mode' is QuickDraw.STROKESET
            return sketch, c_id # A premature return

        if self.problem == QuickDraw.FULLSEQ:
            return sketch, c_id
        elif self.problem == QuickDraw.ENCDEC:
            return sketch[:,:-1], c_id

    def collate(self, batch):
        if self.mode == QuickDraw.STROKESET:
            return batch # premature return
        
        if self.problem == QuickDraw.FULLSEQ:
            lengths = torch.tensor([x.shape[0] for x, _ in batch])
            padded_seq_inp = pad_sequence([torch.tensor(x) for x, _ in batch])
            labels = torch.tensor([c for _, c in batch])
            return pack_padded_sequence(padded_seq_inp, lengths, enforce_sorted=False), labels
        elif self.problem == QuickDraw.ENCDEC:
            lengths_enc = torch.tensor([x.shape[0] for x, _ in batch])
            padded_seq_inp_enc = pad_sequence([torch.tensor(x) for x, _ in batch])
            # lengths_dec = torch.tensor([x.shape[0] for _, (x, _, _), _ in batch])
            # padded_seq_inp_dec = pad_sequence([torch.tensor(x) for _, (x, _, _), _ in batch])
            # padded_seq_out_dec = pad_sequence([torch.tensor(y) for _, (_, y, _), _ in batch])
            # padded_seq_out_pen = pad_sequence([torch.tensor(p) for _, (_, _, p), _ in batch])
            labels = torch.tensor([c for _, c in batch])
            return pack_padded_sequence(padded_seq_inp_enc, lengths_enc, enforce_sorted=False), labels

    def get_dataloader(self, batch_size, shuffle = True, pin_memory = True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate, shuffle=shuffle, pin_memory=pin_memory, drop_last=True)

    def split(self, proportion=0.8):
        train_samples = int(len(self) * proportion)
        qd_test = QuickDraw(self.root, categories=self.categories, max_sketches_each_cat=self.max_sketches_each_cat, normalize_xy=self.normalize_xy,
            dtype=self.dtype, verbose=self.verbose, filter_func=self.filter_func, mode=self.mode, start_from_zero=self.start_from_zero,
            problem=self.problem,
            cache=self.cache[train_samples:])
        self.cache = self.cache[:train_samples]

        if self.verbose:
            print('[Info] Dataset with {} samples created'.format(len(self)))
        
        return self, qd_test

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    qd = QuickDraw(sys.argv[1], raw=True, categories=['cat'], max_sketches_each_cat=10, verbose=True, mode=QuickDraw.STROKESET)
    qdl = qd.get_dataloader(4)
    for S in qdl:
        for sketch, c in S:
            fig = plt.figure()
            for stroke in sketch:
                plt.plot(stroke[:,0], stroke[:,1])
            plt.show()
            plt.close()