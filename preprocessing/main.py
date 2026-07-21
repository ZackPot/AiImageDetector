import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as t
from tqdm import tqdm
import numpy as np

def pre_process_func(x):
    x = x.convert("RGB")
    x = t.functional.to_tensor(x)
    x = t.functional.resize(x, [256, 256])
    return x

def apply_transforms(device, dataset, pre_process, FFT, filepath, max_samples):
    all_signals = []
    samples = 0

    for batch_data in tqdm(dataset, desc=f"Processing {filepath}"):
        imgs = [pre_process(item['image']) for item in batch_data]
        batch = torch.stack(imgs)
        signals = batch[:, 0, :, :].to(device)

        with torch.no_grad():
            fft_signals = FFT(signals).cpu().numpy()

        all_signals.append(np.array(fft_signals, dtype=np.float32))

        samples += batch.size(0)
        if samples >= max_samples:
            break

    all_signals = np.concatenate(all_signals, axis=0)
    np.save(filepath, all_signals)

def FFT(batch):
    f_transform = torch.fft.fft2(batch)
    f_shift = torch.fft.fftshift(f_transform)

    magnitude_spectrum = 20 * torch.log(torch.abs(f_shift) + 1e-9)

    min_vals = torch.amin(magnitude_spectrum, dim=(1, 2), keepdim=True)
    max_vals = torch.amax(magnitude_spectrum, dim=(1, 2), keepdim=True)
    normalized_spectrums = (magnitude_spectrum - min_vals) / (max_vals - min_vals + 1e-9)

    return normalized_spectrums

if __name__ == "__main__":
    device = torch.device('mps')
    batch_size = 64

    dataset_ai = load_dataset(
        "poloclub/diffusiondb",
        split="train",
        streaming=True
    )

    dataset_real = load_dataset(
        "bitmind/MS-COCO",
        split="train",
        streaming=True
    )

    dataloader_ai = DataLoader(
        dataset_ai,
        batch_size=batch_size,
        collate_fn=lambda x: x,
    )

    dataloader_real = DataLoader(
        dataset_real,
        batch_size=batch_size,
        collate_fn=lambda x: x,
    )

    apply_transforms(device, dataloader_ai, pre_process_func, FFT, 'fft_signals_fake.npy', 2500)
    apply_transforms(device, dataloader_real, pre_process_func, FFT, 'fft_signals_real.npy', 2500)