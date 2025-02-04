from tensorflow.keras.models import load_model

# Path ke direktori SavedModel
model_path = 'model/best_model_densenet201_32303a.h5'

# Memuat model
model = load_model(model_path)

# Sekarang model siap digunakan
print("Model loaded successfully!")
