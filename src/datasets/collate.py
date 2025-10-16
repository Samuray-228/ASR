import torch
from torch.nn.utils.rnn import pad_sequence


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

    result_batch = dict()

    for key in dataset_items[0].keys():
        if key in ["text", "audio_path"]:
            result_batch[key] = [item[key] for item in dataset_items]
        elif key == "spectrogram":
            freq_length = dataset_items[0]["spectrogram"].shape[1]
            max_time_length = max(item["spectrogram"].shape[2] for item in dataset_items)
            result_batch["spectrogram"] = torch.zeros((len(dataset_items), freq_length, max_time_length))
            for i, item in enumerate(dataset_items):
                current_length = item["spectrogram"].shape[2]
                result_batch["spectrogram"][i, :, :current_length] = item["spectrogram"][0]
            result_batch["spectrogram_length"] = torch.tensor([item["spectrogram"].shape[2] for item in dataset_items], dtype=torch.int32)
        elif key == "audio":
            result_batch[key] = [item[key] for item in dataset_items]
        else:
            sequences = [item[key][0] for item in dataset_items]
            result_batch[key] = pad_sequence(sequences, batch_first=True)
            if key == "text_encoded":
                result_batch["text_encoded_length"] = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int32)

    return result_batch
