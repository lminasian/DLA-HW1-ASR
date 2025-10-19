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

    def pad_tensors(tensors, dim=-1):
        max_len = max(t.shape[dim] for t in tensors)
        padded_tensors = []
        lengths = []
        for t in tensors:
            lengths.append(t.shape[dim])
            padding_size = max_len - t.shape[dim]

            padding = tuple(list(
                [0, 0] * dim + [0, padding_size]
            ))

            padded = nn.functional.pad(t, (0, padding_size))
            padded_tensors.append(padded)
        padded_tensors = torch.concat(padded_tensors)
        lengths = torch.tensor(lengths)
        return padded_tensors, lengths

    keys = list(batch.keys())
    for key in keys:
        if key in ['audio', 'audio_orig', 'spectrogram', 'text_encoded', 'log_probs']:

            # tranpose is needed because pad_tensors pads on dim=-1
            # and log_probs have shape [N, T, C]
            if key == 'log_probs':
                batch[key] = [t.transpose(1, 2) for t in batch[key]]
            padded, lengths = pad_tensors(batch[key])
            batch[key] = padded
            batch[key + '_length'] = lengths
            if key == 'log_probs':
                batch[key] = batch[key].transpose(1, 2)

        elif key in ['text', 'audio_path']:
            pass
        else:
            raise ValueError("Unexpected key `%s` encountered" % (key,))
    
    return batch