# SCT_ML_3
# 🐱🐶 Cats vs Dogs Classifier using SVM + HOG  

## 📌 Project Overview  
This project implements a **Support Vector Machine (SVM)** classifier to distinguish between images of **cats** and **dogs** using the **Kaggle Dogs vs Cats Dataset**.  
Instead of deep learning, we extract **Histogram of Oriented Gradients (HOG)** features from each image and train an SVM model, showing how classical ML techniques can still achieve strong results in image classification.  

---

## 🎯 Task  
**Task 03:** Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.  

---

## 📂 Dataset  
- Source: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)  
- Contains:  
  - **12,500 cat images**  
  - **12,500 dog images**  
- Images are resized to **128x128** before HOG extraction.  

**Folder structure:**
data/
│── cats/
│ ├── cat.1.jpg
│ ├── cat.2.jpg
│ └── ...
│
└── dogs/
├── dog.1.jpg
├── dog.2.jpg
└── ...


---

## ⚙️ Approach  
1. **Preprocessing**  
   - Convert images to grayscale  
   - Resize to `128x128`  
   - Extract **HOG features** (8100 per image)  

2. **Training**  
   - Split dataset → **70% Train / 15% Val / 15% Test**  
   - Train SVM with **Linear** and **RBF kernels**  
   - Tune hyperparameters with **GridSearchCV**  

3. **Evaluation**  
   - Accuracy on validation & test sets  
   - Classification report (Precision, Recall, F1-score)  
   - Confusion Matrix  

---

## 📊 Results  
- Train set: **17,500 images**  
- Validation set: **3,750 images**  
- Test set: **3,750 images**  
- Each image → **8100 HOG features**  
- **Test Accuracy:** ~XX% (update after running)  

---

## 🚀 How to Run  

### 1. Clone Repository
git clone https://github.com/<your-username>/cats-vs-dogs-svm-hog.git
cd cats-vs-dogs-svm-hog

2. Install Dependencies
pip install -r requirements.txt

3. Train Model
 python src/train.py

4. Predict New Images
python src/predict.py --image path/to/image.jpg
cats-vs-dogs-svm-hog/
│
├── data/                # Dataset (not uploaded)
├── notebooks/           # Jupyter/Colab notebooks
│   └── svm_hog.ipynb
├── src/                 
│   ├── train.py         # Training script
│   └── predict.py       # Prediction script
├── outputs/             # Saved models
├── requirements.txt     # Dependencies
└── README.md            # Project description

🛠️ Technologies Used

Python

OpenCV

scikit-image

scikit-learn

NumPy

Matplotlib

tqdm

📜 License

This project is licensed under the MIT License.

