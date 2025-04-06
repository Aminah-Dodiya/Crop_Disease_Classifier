import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from model import get_model
from data_preprocessing import prepare_data
from utils import get_device

def file_to_prediction(model, datadir, filename, transform_pipeline, device, class_names):
    """Process a single test image and return predictions as a DataFrame row."""
    file_path = os.path.join(datadir, filename)
    image = Image.open(file_path)
    
    # Preprocess the image
    transformed = transform_pipeline(image)
    image_cuda = transformed.unsqueeze(0).to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(image_cuda)
        confidence = torch.nn.functional.softmax(output, dim=1)

    # Get the predicted class
    predicted_class_idx = torch.argmax(confidence, dim=1).item()
    predicted_class_name = class_names[predicted_class_idx]

    # Create a DataFrame row with predictions
    conf_list = confidence.tolist()[0]
    conf_df = pd.DataFrame([[filename] + conf_list + [predicted_class_name]])
    conf_df.columns = ["ID"] + class_names + ["predicted_class"]

    return conf_df

def predict_on_test_set(model, test_dir, transform_pipeline, class_names, output_csv="cassava_disease_predictions.csv", device="cpu"):
    """Process all test images and save predictions to a CSV file."""
    prediction_dfs = []

    print("Predicting on test set...")
    for filename in tqdm(os.listdir(test_dir), desc="Processing test images"):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            prediction_dfs.append(file_to_prediction(model, test_dir, filename, transform_pipeline, device, class_names))

    # Combine and save predictions
    predictions_df = pd.concat(prediction_dfs)
    predictions_df = predictions_df.sort_values("ID").reset_index(drop=True)
    predictions_df.to_csv(output_csv, index=False)
    print(f"Test predictions saved to {output_csv}")

if __name__ == "__main__":
    # Setup
    device = get_device()
    test_dir = os.path.join("data_p2", "test")

    # Load model and checkpoint
    model = get_model(num_classes=5, device=device)
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get transformation pipeline and class names
    _, _, train_dataset, _ = prepare_data("data_p2/train")
    transform_pipeline = train_dataset.dataset.transform
    class_names = train_dataset.dataset.classes

    # Run predictions
    predict_on_test_set(model, test_dir, transform_pipeline, class_names, device=device)