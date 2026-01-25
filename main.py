import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
from models import PokemonResNet

# ---------- CONFIG ----------
MODEL_PATH = "pokemon_resnet34.pth"
DATASET_DIR = r"D:\PycharmProjects\Pokemon_Dataset_879"
TOP_K = 5
IMAGE_SIZE = 224
# ----------------------------

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(image_path)
    img = transform(img)
    return img.unsqueeze(0)  # add batch dimension

def main(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names i.e. pokemon names
    dummy_dataset = datasets.ImageFolder(DATASET_DIR)
    class_names = dummy_dataset.classes
    num_classes = len(class_names)

    # Load model
    model = PokemonResNet(num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load image
    image = load_image(image_path).to(device)

    # Inference
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)

    # Top-K
    top_probs, top_idxs = probs.topk(TOP_K)

    print("\n Top Pokémon predictions:")
    for i in range(TOP_K):
        idx = top_idxs[0][i].item()
        prob = top_probs[0][i].item()
        print(f"{i+1}. {class_names[idx]} ({prob*100:.2f}%)")


if __name__ == "__main__":
    # image_path = input("Which Pokemon to identify: ")
    image_path = 'assets/em.jpg'
    main(image_path)



