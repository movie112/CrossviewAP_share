import logging as log
import h5py
import json
import collections
import numpy as np
import torch
import torch.utils.data as tud

import utils.stateful_dataset
import utils.speech_utils


# help to unify all the vocabs
class CountDict:
    def __init__(self):
        self.dict = {}

    def insert(self, x):
        if x in self.dict:
            self.dict[x] += 1
        else:
            self.dict[x] = 1
        return

    def elt_set(self):
        return set(self.dict.keys())

    def idx_dict(self):
        sorted_d = sorted(self.dict.items(), key=lambda x: x[1])
        sorted_d.reverse()
        idx = 0
        result_dict = {}
        for k, v in sorted_d:
            result_dict[k] = idx
            idx += 1
        return result_dict


def combine_subwords_to_ids(vocab_fns, subwords):
    vocab_dict = CountDict()
    for vocab_fn in vocab_fns:
        with open(vocab_fn, "r") as f:
            vocab = json.load(f)
        for k, v in vocab[f"{subwords}_to_ids"].items():
            vocab_dict.insert(k)
    return vocab_dict.idx_dict()


class MultilangDataset(utils.stateful_dataset.StatefulDataset):

    def __init__(self, feats_fns, align_fns, vocab_fns,  subwords, subwords_to_ids,
                 min_occ_count=0, min_seg_dur=6, stack_frames=False,
                 batch_size=1, shuffle=None, cache=None):

        super().__init__()

        if cache is not None:
            feats_fns = [cache(fn) for fn in feats_fns]
            align_fns = [cache(fn) for fn in align_fns]

        self.stack_frames = stack_frames

        # number of training langauges
        self.n_languages = len(feats_fns)
        log.info(f"Using {self.n_languages} training languages")

        log.info(f"Using {feats_fns}; stacked={stack_frames}")
        featss = [h5py.File(fn, "r") for fn in feats_fns]

        log.info(f"Using {align_fns}")
        aligns = [h5py.File(fn, "r") for fn in align_fns]

        # using different embeddings
        log.info(f"Using {vocab_fns}")

        self.subwords_to_ids = subwords_to_ids
        self.n_subwords = len(self.subwords_to_ids)

        log.info(f"Using {self.n_subwords} tokens in subwords")

        # all training_data
        self.langs_data = []

        # training samples
        for lang_id, vocab_fn in enumerate(vocab_fns):
            with open(vocab_fn, "r") as f:
                vocab = json.load(f)
            lang_dict = {"words_to_ids": vocab["words_to_ids"], "word_to_subwords": vocab[f"word_to_{subwords}"]}

            # counting words
            word_counts = collections.defaultdict(int)
            align = aligns[lang_id]
            feats = featss[lang_id]
            for cs in align:
                for uid, g in align[cs].items():
                    words = g["words"][()]
                    durs = g["ends"][()] - g["starts"][()] + 1
                    for i, word in enumerate(words):
                        # if word == "<unk>":
                        if word[0] == '<' and word[-1] == '>':
                            continue
                        if durs[i] < min_seg_dur or durs[i] > 500:
                            continue
                        word_counts[word] += 1

            lang_dict['feats'] = feats
            lang_dict['stack_frames'] = stack_frames
            lang_dict['align'] = align
            lang_dict['word_counts'] = word_counts
            lang_dict['ids_to_words'] = {v: k for k, v in lang_dict["words_to_ids"].items()}

            # total training frames
            total_frame = 0.0
            total_train_instances= 0

            # example of this language
            examples = {}
            for cs in align:
                for uid, g in align[cs].items():
                    words = g["words"][()]
                    durs = g["ends"][()] - g["starts"][()] + 1
                    for i, word in enumerate(words):
                        # if word == "<unk>":
                        if word[0] == '<' and word[-1] == '>':
                            continue
                        if durs[i] < min_seg_dur or durs[i] > 500:
                            continue
                        if word_counts[word] < min_occ_count:
                            if word in lang_dict["words_to_ids"]:
                                del lang_dict["words_to_ids"][word]
                            if word in lang_dict["word_to_subwords"]:
                                del lang_dict["word_to_subwords"][word]
                            continue
                        examples[(cs, uid, i)] = durs[i]
                        total_frame += durs[i]
                        total_train_instances += 1

            log.info(f"total train data time {total_frame/60/100} minutes")
            log.info(f"total train data instances {total_train_instances}")

            examples = sorted(examples, key=examples.get, reverse=True)

            lang_dict['examples'] = examples

            self.langs_data.append(lang_dict)

        # all examples
        all_examples = []
        for i in range(self.n_languages):
            all_examples.append(self.langs_data[i]['examples'])

        batch_sampler = utils.stateful_dataset.MultilangStatefulBatchSampler(
            all_examples, batch_size=batch_size, shuffle=shuffle)

        loader = tud.DataLoader(self,
                                batch_sampler=batch_sampler,
                                collate_fn=self.collate_fn,
                                pin_memory=True,
                                num_workers=0)

        self.loader = loader

    @property
    def feat_dim(self):
        for lang_dict in self.langs_data:
            feats = lang_dict['feats']
            for cs in feats:
                for uid, g in feats[cs].items():
                    seg = g["feats"][()]
                    seg = utils.speech_utils.add_deltas(seg)
                    if self.stack_frames:
                        seg = utils.speech_utils.stack(seg)
                    return seg.shape[1]

    @property
    def num_subwords(self):
        return self.n_subwords

    @property
    def unify_subwords_to_ids(self):
        return self.subwords_to_ids

    def __getitem__(self, ex):

        lang_id, cs, uid, i = ex
        lang_dict = self.langs_data[lang_id]
        align = lang_dict['align']
        feats = lang_dict['feats']
        words_to_ids = lang_dict['words_to_ids']
        word_to_subwords = lang_dict['word_to_subwords']

        start = align[cs][uid]["starts"][()][i]
        end = align[cs][uid]["ends"][()][i]
        seg = feats[cs][uid]["feats"][()][start:end + 1]

        if len(seg) == 0:
            return None

        seg = utils.speech_utils.add_deltas(seg)
        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)

        raw_word = align[cs][uid]["words"][()][i]
        if isinstance(raw_word, (bytes, bytearray)):
            word = raw_word.decode('utf-8')
        else:
            word = raw_word

        if word not in words_to_ids:
            return None
        word_id = words_to_ids[word]
 

        # seq = word_to_subwords[word][0]  # Note: uses the first pronunciation
        sub_list = word_to_subwords.get(word, [])
        if not sub_list:
            log.warning(f"No subwords found for word '{word}'. Skipping sample.")
            return None
        seq = sub_list[0]

        for s in seq:
            if s not in self.subwords_to_ids:
                return None
        seq_ids = np.array([ self.subwords_to_ids[s] + 1 for s in seq ])

        if len(seq_ids) == 0:
            return None

        return {"seg": seg, "seq_ids": seq_ids, "word_id": word_id}

    def collate_fn(self, batch):
        # ① __getitem__에서 None 리턴된 샘플 제거
        batch = [ex for ex in batch if ex is not None]
        # ② 남은 샘플이 하나도 없으면 None 리턴 → train loop에서 skip
        if len(batch) == 0:
            return None
        batch_size = 0
        max_seg_dur = 0
        max_seq_len = 0

        for ex in batch:
            if ex is not None:
                max_seg_dur = max(max_seg_dur, len(ex["seg"]))
                max_seq_len = max(max_seq_len, len(ex["seq_ids"]))
                batch_size += 1

        feats = torch.zeros(batch_size, max_seg_dur, self.feat_dim)
        feat_lens = torch.zeros(batch_size, dtype=torch.long)
        seqs = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        seq_lens = torch.zeros(batch_size, dtype=torch.long)
        ids = np.zeros(batch_size, dtype=np.int32)

        i = 0
        for ex in batch:
            if ex is not None:
                seg = ex["seg"]
                feats[i, :len(seg)] = torch.from_numpy(seg)
                feat_lens[i] = len(seg)
                seq = ex["seq_ids"]
                seqs[i, :len(seq)] = torch.from_numpy(seq)
                seq_lens[i] = len(seq)
                ids[i] = ex["word_id"]
                i += 1

        ids, ind, inv = np.unique(ids, return_index=True, return_inverse=True)
        ind = torch.from_numpy(ind)
        inv = torch.from_numpy(inv)

        return {
            "view1": feats, "view1_lens": feat_lens,
            "view2": seqs[ind], "view2_lens": seq_lens[ind],
            "ids": ids, "inv": inv
        }


class DevDataset(utils.stateful_dataset.StatefulDataset):

    def __init__(self, feats, align, vocab, subwords, subwords_to_ids,
                 word_counts=None, min_occ_count=0,
                 min_seg_dur=6, stack_frames=False,
                 batch_size=1, shuffle=None, cache=None):

        super().__init__()

        if cache is not None:
            feats = cache(feats) # fbank_pitch_feats_with_cmvn.sample_dev.hdf5
            align = cache(align) # align.sample_dev.hdf5

        log.info(f"Using {feats}; stacked={stack_frames}")
        feats = h5py.File(feats, "r")

        log.info(f"Using {align}")
        align = h5py.File(align, "r")

        log.info(f"Using {vocab}") # vocab.json
        with open(vocab, "r") as f:
            vocab = json.load(f)
            
        words_to_ids = vocab["words_to_ids"]
        word_to_subwords = vocab[f"word_to_{subwords}"]

        if word_counts is None:
            word_counts = collections.defaultdict(int)
            for cs in align: # cs: '91777-B'
                for uid, g in align[cs].items(): # uid='91777-B_000698_001132, g=<HDF5 group "/91777-B/91777-B_000698_001132" (4 members)>
                    words = g["words"][()] # array([b'\xe5\x96\x82'], dtype=object)
                    durs = g["ends"][()] - g["starts"][()] + 1 # array([52])
                    for i, word in enumerate(words): # word=array([b'\xe5\x96\x82'], dtype=object)
                        # if word == "<unk>":
                        if word[0] == '<' and word[-1] == '>':
                            continue
                        if durs[i] < min_seg_dur or durs[i] > 500:
                            continue
                        word_counts[word] += 1

        examples = {}

        # total time and instance
        total_frame = 0.0
        total_dev_instances = 0

        for cs in align:
            for uid, g in align[cs].items(): # '91777-B_000698_001132'
                words = g["words"][()]
                durs = g["ends"][()] - g["starts"][()] + 1
                for i, word in enumerate(words):
                    # if word == "<unk>":
                    if word.decode('utf-8')[0] == '<' and word.decode('utf-8')[-1] == '>':
                        continue
                    if durs[i] < min_seg_dur or durs[i] > 500: # min_seg_dur: 0 너무 짧거나 길면 제거
                        continue
                    if word_counts[word] < min_occ_count: # min_occ_count보다 적은 경우 제거, 0
                        if word in words_to_ids:
                            del words_to_ids[word]
                        if word in word_to_subwords:
                            del word_to_subwords[word]
                        continue
                    examples[(cs, uid, i)] = durs[i] # cs=91777-B, uid=91777-B_000698_001132, i=0, durs[i]=52
                    total_frame += durs[i]
                    total_dev_instances += 1

        log.info(f"total dev data time {total_frame / 60 / 100} minutes")
        log.info(f"total dev data instances {total_dev_instances}")
        
        examples = sorted(examples, key=examples.get, reverse=True)

        batch_sampler = utils.stateful_dataset.StatefulBatchSampler(
            examples, batch_size=batch_size, shuffle=shuffle)

        loader = tud.DataLoader(self,
                                batch_sampler=batch_sampler,
                                collate_fn=self.collate_fn,
                                pin_memory=True,
                                num_workers=0)

        self.feats = feats
        self.stack_frames = stack_frames
        self.align = align
        self.word_counts = word_counts
        self.words_to_ids = words_to_ids
        self.ids_to_words = {v: k for k, v in words_to_ids.items()}
        self.word_to_subwords = word_to_subwords
        self.subwords_to_ids = subwords_to_ids
        self.loader = loader

    @property
    def feat_dim(self):
        for cs in self.feats:
            for uid, g in self.feats[cs].items():
                seg = g["feats"][()]
                seg = utils.speech_utils.add_deltas(seg)
                if self.stack_frames:
                    seg = utils.speech_utils.stack(seg)
                return seg.shape[1]

    @property
    def num_subwords(self):
        return len(self.subwords_to_ids)

    def __getitem__(self, ex):

        cs, uid, i = ex
        start = self.align[cs][uid]["starts"][()][i]
        end = self.align[cs][uid]["ends"][()][i]
        seg = self.feats[cs][uid]["feats"][()][start:end + 1]

        if len(seg) == 0:
            return None

        seg = utils.speech_utils.add_deltas(seg)
        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)

        word = self.align[cs][uid]["words"][()][i]
        word = word.decode('utf-8')

        if word not in self.words_to_ids:
            return None
        word_id = self.words_to_ids[word]
        
        # seq = self.word_to_subwords[word][0]  # Note: uses the first pronunciation
        sub_list = self.word_to_subwords.get(word, [])
        if not sub_list:
            log.warning(f"No subwords found for dev word '{word}'. Skipping sample.")
            return None
        seq = sub_list[0]
        for s in seq:
            if s not in self.subwords_to_ids:
                return None
        seq_ids = np.array([ self.subwords_to_ids[s] + 1 for s in seq ])

        if len(seq_ids) == 0:
            return None

        return {"seg": seg, "seq_ids": seq_ids, "word_id": word_id}

    def collate_fn(self, batch):
        # ① __getitem__에서 None 리턴된 샘플 제거
        batch = [ex for ex in batch if ex is not None]
        # ② 남은 샘플이 하나도 없으면 None 리턴 → train loop에서 skip
        if len(batch) == 0:
            return None
        batch_size = 0
        max_seg_dur = 0
        max_seq_len = 0

        for ex in batch:
            if ex is not None:
                max_seg_dur = max(max_seg_dur, len(ex["seg"]))
                max_seq_len = max(max_seq_len, len(ex["seq_ids"]))
                batch_size += 1

        feats = torch.zeros(batch_size, max_seg_dur, self.feat_dim)
        feat_lens = torch.zeros(batch_size, dtype=torch.long)
        seqs = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        seq_lens = torch.zeros(batch_size, dtype=torch.long)
        ids = np.zeros(batch_size, dtype=np.int32)

        i = 0
        for ex in batch: # len(batch) = 116
            if ex is not None: # ex.keys() = dict_keys(['seg', 'seq_ids', 'word_id'])
                seg = ex["seg"] # (56, 234) len(seg): 56
                feats[i, :len(seg)] = torch.from_numpy(seg)
                feat_lens[i] = len(seg) # len(feat_lens): 116 tensor([56,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                seq = ex["seq_ids"] # 계속 늘어남array([207,  18,  74,   6, 202, 184,  73])
                seqs[i, :len(seq)] = torch.from_numpy(seq) # seqs.shape: 116, 19 tensor([[207,  20,  71,  ...,   0,   0,   0],
                                                            # [ 87, 225, 170,  ...,   0,   0,   0],
                seq_lens[i] = len(seq) # tensor([13,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                ids[i] = ex["word_id"] # array([10296,  3219,     0,     0,     0,     0,     0,     0,     0,
                i += 1

        ids, ind, inv = np.unique(ids, return_index=True, return_inverse=True)
        # 정렬된 고유 값 77/ 고유 값이 처음 등장한 인덱스 77/원래 배열의 각 요소가 고유 배열에서 차지하는 인덱스 116
        ind = torch.from_numpy(ind)
        inv = torch.from_numpy(inv)

        return {
            "view1": feats, "view1_lens": feat_lens,  # feats.shape: (batch_size, max_seg_dur, self.feat_dim) torch.Size([116, 56, 234])
            # feat_lens.shape: torch.Size([116])
            "view2": seqs[ind], "view2_lens": seq_lens[ind], # seqs.shape: 116, 19, seqs[ind].shape: 77, 19 / seq_lens[ind].shape: 77
            "ids": ids, "inv": inv
        }


class DevMultilangDataset(utils.stateful_dataset.StatefulDataset):

    def __init__(self, subwords_to_ids):

        super().__init__()

        self.subwords_to_ids = subwords_to_ids
        self.n_subwords = len(self.subwords_to_ids)

        log.info(f"Using {self.n_subwords} tokens in subwords")

        batch_sampler = utils.stateful_dataset.MultilangStatefulBatchSampler([])

        loader = tud.DataLoader(self, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)

        self.loader = loader

    @property
    def num_subwords(self):
        return self.n_subwords

    @property
    def unify_subwords_to_ids(self):
        return self.subwords_to_ids
