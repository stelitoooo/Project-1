import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")

# -------------------------------
# Model loading
# -------------------------------
@st.cache_resource
def load_model():
    try:
        from sklearn.datasets import load_digits
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split

        digits = load_digits()
        X = digits.images.reshape(len(digits.images), -1) / 16.0
        y = digits.target

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


model = load_model()

if model is None:
    st.warning("Could not load model. Using fallback recognition.")
else:
    st.success("Model loaded successfully!")

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image file", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to grayscale and resize
        img_gray = image.convert("L")
        img_resized = img_gray.resize((8, 8))

        img_array = np.array(img_resized)

        # Invert if background is light
        if np.mean(img_array) > 128:
            img_array = 255 - img_array

        # Normalize to match sklearn digits
        img_array = (img_array / 255.0) * 16.0
        img_flat = img_array.flatten().reshape(1, -1)

        if model is not None:
            prediction = model.predict(img_flat)[0]
            st.write(f"## Prediction: **{prediction}**")

            probs = model.predict_proba(img_flat)[0]
            st.write("### Probabilities:")
            for i, prob in enumerate(probs):
                st.write(f"Digit {i}: {prob:.2%}")

        else:
            st.write("## Using fallback recognition")
            digit_guess = int(np.argmax(np.sum(img_array, axis=0)) % 10)
            st.write(f"Estimated digit: **{digit_guess}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# -------------------------------
# Sidebar Instructions
# -------------------------------
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a handwritten digit (0-9)
2. The image will be resized to 8×8 pixels
3. AI model will predict the digit

**Best results:**
- White background
- Black digit
- Centered digit
- Minimal noise
""")
