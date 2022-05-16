from pathlib import Path
import urllib.request


model_path_to_url = {
    Path("models/vgg/ndcc_stanford_dogs_30.pt"):
    "https://drive.google.com/uc?export=download&id=1gm8GvXkTjgw-1u3YBdqioOGG0onD9si9&confirm=t",

    Path("models/vgg/vgg_classifier_1000.pt"):
    "https://drive.google.com/uc?export=download&id=1bw5pYUPKPg6WlMyowuzG-nfCTBeJvvS0&confirm=t",

    Path("models/vgg/vgg_classifier_1000_2.pt"):
    "https://drive.google.com/uc?export=download&id=1RZG8ZqgYoq5ihbVFq0yaYQjG9qNCGnO_&confirm=t",

    Path("models/vgg/vgg_classifier_1000_3.pt"):
    "https://drive.google.com/uc?export=download&id=1R4gwJsPJRtkDILjpa06tmpkBYs5z-cwD&confirm=t",

    Path("models/vgg/vgg_classifier_1000_4.pt"):
    "https://drive.google.com/uc?export=download&id=1MtPn-D0AP7bgpObT0IKiQoS06YpZcas1&confirm=t",

    Path("models/vgg/vgg_classifier_1000_5.pt"):
    "https://drive.google.com/uc?export=download&id=1HNjKNYXTkFqNR-O6HWAoAIIXxjNbUMtK&confirm=t",
}

if __name__ == "__main__":
    for model_path, url in model_path_to_url.items():
        if not model_path.exists():
            print("Downloading " + str(model_path))
            urllib.request.urlretrieve(url, model_path)
        else:
            print("Found " + str(model_path))
