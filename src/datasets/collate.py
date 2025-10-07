import torch
import torch.nn as nn

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    if not dataset_items:
        return {}

    batch = {}

    for sample in dataset_items:
        for key, val in sample.items():
            if key not in batch:
                batch[key] = []
            batch[key].append(val)

    def pad_tensors(tensors):
        max_len = max(t.shape[-1] for t in tensors)
        padded_tensors = []
        lengths = []
        for t in tensors:
            lengths.append(t.shape[-1])
            padding_size = max_len - t.shape[-1]
            padded = nn.functional.pad(t, (0, padding_size))
            padded_tensors.append(padded)
        padded_tensors = torch.concat(padded_tensors)
        lengths = torch.tensor(lengths)
        return padded_tensors, lengths

    keys = list(batch.keys())
    for key in keys:
        if key in ['audio', 'spectrogram', 'text_encoded']:
            padded, lengths = pad_tensors(batch[key])
            batch[key] = padded
            batch[key + '_length'] = lengths
        elif key in ['text', 'audio_path']:
            pass
        else:
            raise ValueError("Unexpected key `%s` encountered" % (key,))
    
    return batch