# 🌿 CropCure: Plant Disease Classifier using MobileNetV2

📉 **CropCure** is a lightweight deep learning-based web app that can classify plant diseases from leaf images using the MobileNetV2 model. Designed to assist farmers, agriculturists, and researchers in identifying plant diseases early.

---

## 📌 Project Highlights

- 🔍 **Model**: Pretrained MobileNetV2 (transfer learning)
- 🏷️ **Classes**: 71 different plant disease categories
- 📊 **Accuracy**: Achieved 88.89% on the test dataset
- 💻 **Interface**: Built using [Gradio](https://gradio.app)
- 🧠 **Framework**: PyTorch

---

## 📁 Dataset

- Sourced from Kaggle:\
  👉 [Plant Disease Classification (Merged Dataset)](https://www.kaggle.com/datasets/alinedobrovsky/plant-disease-classification-merged-dataset)

This dataset includes thousands of labeled leaf images of various crops with healthy and diseased states.

---

## 🧠 Model Details

- Architecture: **MobileNetV2**
- Strategy: **Transfer Learning** with final classification layer updated to 71 classes
- Optimized using Adam optimizer with CrossEntropyLoss
- Trained with image resizing and normalization

---

## 🚀 Running the App

You can try it locally by cloning the repository and running:

```bash
git clone https://github.com/yourusername/cropcure.git
cd cropcure
pip install -r requirements.txt
python app.py
```

Make sure you have `mobilenetv2_pretrained.pth` and `class_names.txt` in the same directory as `app.py`.

---

## 🖼️ Using the App

Once launched, just upload a leaf image and the app will:

1. Preprocess it (resize, normalize)
2. Predict the disease using MobileNetV2
3. Show the class name and prediction confidence

---

## 📂 Project Structure

```bash
🔹 app.py                 # Gradio web app
🔹 mobilenetv2_pretrained.pth  # Trained weights
🔹 class_names.txt        # List of class names
🔹 requirements.txt
🔹 README.md
🔹 notebook/
    └️ Workshop_MobileNetV2_Pre-Trained.ipynb
```

---

## 🔧 Requirements

See [`requirements.txt`](./requirements.txt) for all dependencies.

---

## 📜 License

MIT License.\
You are free to use, modify, and distribute the code with or without attribution, with minimal restrictions.

---

## 🤝 Acknowledgements

- [Kaggle Dataset](https://www.kaggle.com/datasets/alinedobrovsky/plant-disease-classification-merged-dataset)
- [PyTorch](https://pytorch.org)
- [Gradio](https://www.gradio.app)
- [Hugging Face Spaces](https://huggingface.co/spaces) for deployment

---

## 🔗 Author

Made with ❤️ by **Rohit Gomes**

