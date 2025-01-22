from analyzing.actlbl_model import train_model
from util.dataset_manager import augment_dataset, history_add_padding, load_dataset


dataset, labels = load_dataset("dataset")
dataset, labels = augment_dataset(dataset, labels)
dataset = history_add_padding(dataset)

model = train_model(dataset, labels)