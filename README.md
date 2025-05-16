# Classification-of-Rice-Seeds-Using-Machine-Learning
🌾 Classification of Rice Seeds Using Machine Learning
This project uses deep learning techniques to classify different types of rice seeds based on their physical characteristics. Leveraging convolutional neural networks (CNNs) such as ResNet50 and VGG, the system achieves high accuracy in rice variety classification. A user-friendly web interface is built with Streamlit for easy interaction and real-time prediction.


📁 Project Structure
data/ – Training and testing image datasets 
models/ – Pretrained and fine-tuned ResNet50 and VGG models
notebooks/ – Jupyter notebooks for experimentation and EDA
streamlit_app.py – Streamlit web interface for prediction
train.py – Model training script
requirements.txt – Required Python packages


🖼️ Dataset
Rice seed images with labeled categories (e.g., Basmati, Jasmine, Karacadag). Dataset includes various morphological traits captured through images.
(Specify your data source, e.g., a public dataset or custom image set.)


🧠 Deep Learning Models
ResNet50: A residual neural network known for deep feature extraction and reduced training time via skip connections.
VGG16/VGG19: Deep CNN with uniform architecture, known for performance and simplicity.
Models fine-tuned using transfer learning with pre-trained weights from ImageNet.


🔧 Tools & Libraries
Python
PyTorch – Model training and evaluation
Torchvision – Pretrained models and image transformations
Streamlit – Web app for user interaction and predictions
Matplotlib / Seaborn – Visualization


🧪 Evaluation Metrics
Accuracy
Confusion Matrix
Precision, Recall, F1 Score
Training & validation curves


🚀 Streamlit Web App
Upload an image of a rice seed
Run prediction using trained ResNet50 or VGG model
View predicted variety and model confidence


📌 Future Work
Add more rice seed varieties
Improve UI/UX in Streamlit
Deploy the app via Streamlit Cloud or Docker
