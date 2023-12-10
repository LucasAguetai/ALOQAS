import json
import os

def load_files_from_directory(directory):
    all_data = []
    all_labels = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:

                articles = json.load(file)
                for article in articles:

                    for label, data in article.items():
                        all_data.append(article)
                        if label not in all_labels:
                            all_labels.append(label)

    return all_data, all_labels

def load_dataset(base_directory, dataset_type):
    dataset_directory = os.path.join(base_directory, dataset_type)
    return load_files_from_directory(dataset_directory)

pathToDataset = "../chunking-dataset"

train_data, train_labels = load_dataset(pathToDataset, "train")
test_data, test_labels = load_dataset(pathToDataset, "test")
val_data, val_labels = load_dataset(pathToDataset, "val")

print(f"Train data: {len(train_data)} articles")
print(f"Test data: {len(test_data)} articles")
print(f"Validation data: {len(val_data)} articles")