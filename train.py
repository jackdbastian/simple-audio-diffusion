from datasets import load_from_disk
from torchvision.transforms import Compose, Normalize, ToTensor

dataset = load_from_disk('data/audio-diffusion-256')

resolution = dataset[0]["image"].height, dataset[0]["image"].width

augmentations = Compose([
    ToTensor(),
    Normalize([0.5], [0.5]),
])

def transforms(examples):
    images = [augmentations(image) for image in examples["image"]]

print(dataset)