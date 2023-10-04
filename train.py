from datasets import load_from_disk
from torchvision.transforms import Compose, Normalize, ToTensor
import torch
from diffusers import UNet2DModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator


# Accelerator params
GRADIENT_ACCUMULATION_STEPS=1
MIXED_PRECISION='no'
LOGGING_DIR='logs'

# dataset params
DATA_PATH='data/audio-diffusion-256'
BATCH_SIZE=16
NUM_TRAIN_STEPS=100

# optimizer params
LEARNING_RATE=1e-4
ADAM_BETA1=0.95
ADAM_BETA2=0.999
ADAM_WEIGHT_DECAY=1e-6
ADAM_EPSILON=1e-08

# lr_scheduler params
LR_SCHEDULER_TYPE='cosine'
LR_WARMUP_STEPS=500
NUM_EPOCHS=100

# EMAModel Params
EMA_INV_GAMMA=1.0
EMA_POWER=3 /4
EMA_MAX_DECAY=


accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision=MIXED_PRECISION,
    log_with="tensorboard",
    project_dir=LOGGING_DIR,
)

dataset = load_from_disk(DATA_PATH)['train']

resolution = dataset[0]["image"].height, dataset[0]["image"].width

augmentations = Compose([
    ToTensor(),
    Normalize([0.5], [0.5]),
])

def transforms(examples):
    images = [augmentations(image) for image in examples["image"]]

dataset.set_transform(transforms)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet2DModel(
    sample_size=resolution,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

noise_scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_STEPS)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(ADAM_BETA1, ADAM_BETA2),
    weight_decay=ADAM_WEIGHT_DECAY,
    eps=ADAM_EPSILON,
)

lr_scheduler = get_scheduler(
    LR_SCHEDULER_TYPE,
    optimizer=optimizer,
    num_warmup_steps=LR_WARMUP_STEPS,
    num_training_steps=(len(train_dataloader) * NUM_EPOCHS) //
    GRADIENT_ACCUMULATION_STEPS,
)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler)


ema_model = EMAModel(
    getattr(model, "module", model),
    inv_gamma=EMA_INV_GAMMA,
    power=EMA_POWER,
    max_value=EMA_MAX_DECAY,
)