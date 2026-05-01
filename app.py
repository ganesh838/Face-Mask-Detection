# ---------------- CUSTOM CSS ----------------
import streamlit as st

st.markdown("""
<style>
/* Main background */
.main {
    # background: linear-gradient(to right, #f8f9fa, #e3f2fd);
    background: linear-gradient(to right, #e8f5e9, #e0f7fa);

}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #2A5442;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Title styling */
h1 {
    color: #0d47a1;
    text-align: center;
}

/* Subheaders */
h2, h3 {
    color: #1565c0;
}

/* Success box */
div[data-testid="stAlert"][role="alert"] {
    border-radius: 10px;
    font-size: 18px;
}

/* Buttons */
button[kind="primary"] {
    background-color: #1565c0;
    color: white;
    border-radius: 8px;
    height: 3em;
}

/* Info box */
div[data-testid="stInfo"] {
    background-color: #e3f2fd;
    border-left: 5px solid #1565c0;
}

/* Warning box */
div[data-testid="stWarning"] {
    background-color: #fff3e0;
    border-left: 5px solid #fb8c00;
}

/* Image border */
img {
    border-radius: 12px;
    border: 2px solid #1565c0;
}
</style>
""", unsafe_allow_html=True)


import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# 2. PAGE CONFIG

st.set_page_config(
    page_title="Face Mask Detection",
    layout="centered"
)



# 3. LOAD MODEL (CACHED)

@st.cache_resource
def load_cnn_model():
    return load_model("face_mask_model.h5")

model = load_cnn_model()
CLASSES = ['With Mask', 'Without Mask']

# 4. HELPER FUNCTIONS

def preprocess_image(image: Image.Image):
    """Resize, normalize and reshape image for prediction"""
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)
    return img


def predict_mask(image: Image.Image):
    """Run model prediction and return class & confidence"""
    img = preprocess_image(image)
    prediction = model.predict(img)
    class_name = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return class_name, confidence


# 5. SIDEBAR NAVIGATION

def sidebar_navigation():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "📘 How It Works",
            "🧠 Concepts Used",
            "🎯 Prediction",
            "📌 Examples & Use Cases",
            "❓ FAQ / Common Issues"
        ]
    )



# 6. PAGE SECTIONS

def home_page():
    st.title("😷 Face Mask Detection App")
    st.markdown("""
### 🎯 Project Overview
This Deep Learning application detects whether a person  
is **wearing a face mask or not** using CNN.

### 🛠 Technologies Used
- Python
- TensorFlow / Keras
- CNN
- Streamlit
""")


def how_it_works_page():
    st.title("📘 How Face Mask Detection Works")

    st.markdown("""
🧭 Complete Working Flow (Step by Step)

This application follows a **Deep Learning pipeline**.  
Let us understand **what happens internally** when a user gives an image.

---

1️⃣ Image Input (User Side)
The user provides an image in **two ways**:
- 📁 Upload an image from device  
- 📷 Capture an image using camera  

👉 This image usually contains a **human face**.

---

2️⃣ Image Preprocessing (Machine Preparation)
Before giving the image to the model, it must be **prepared**.

We perform:
- **Resize** → Convert all images to **128 × 128 pixels**  
- **RGB Conversion** → Ensure 3 color channels  
- **Normalization** → Convert pixel values from **0–255 → 0–1**

📌 Why preprocessing?
> Deep Learning models work better when input data is clean and consistent.

---

 3️⃣ Feature Extraction using CNN
Now the image is passed to the **Convolutional Neural Network (CNN)**.

CNN learns features in stages:
- **First layers** → detect edges and lines  
- **Middle layers** → detect shapes (nose, mouth, mask area)  
- **Deeper layers** → detect complex patterns (mask presence)

📌 CNN automatically learns these features —  
**we do NOT write rules manually**.

---

 4️⃣ Prediction by the Model
After extracting features:
- CNN sends information to **Dense layers**
- Model calculates **probabilities** for each class:
  - With Mask  
  - Without Mask  

The class with **highest probability** is selected.

Example:

👉 Final output = **With Mask**

---

5️⃣ Confidence Score
Along with prediction, the model also provides a **confidence percentage**.

Confidence tells:
> “How sure the model is about its decision”

- High confidence → Model is sure  
- Low confidence → Model is confused  

---

 6️⃣ Output Display (User Side)
Finally, the application shows:
- ✅ Prediction result  
- 🔢 Confidence percentage  

This completes the **end-to-end Deep Learning process**.

---

🔁 Complete Flow Summary
""")


def concepts_page():
    st.title("🧠 Concepts Used")
#     st.markdown("""
# - CNN for image understanding  
# - Convolution & Pooling layers  
# - Softmax for probability output  
# """)

    st.markdown("""
                
     1️⃣ Convolutional Neural Network (CNN)
    - Special deep learning model for images
    - Learns visual patterns automatically

     2️⃣ Convolution Layer
    - Detects edges, textures, shapes

     3️⃣ Pooling Layer
    - Reduces image size
    - Keeps important information

     4️⃣ Flatten Layer
    - Converts image data into 1D format

     5️⃣ Dense Layer
    - Makes final decision  

     6️⃣ Softmax
    - Converts output into probabilities
    """)



def prediction_page():
    st.title("🎯 Face Mask Prediction")

    # Choose input method
    input_method = st.radio(
        "Select Input Method",
        ["📁 Upload Image", "📷 Use Camera"]
    )

    image = None

    # -------- Upload Image --------
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Face Image",
            type=["jpg", "png", "jpeg"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    # -------- Camera Input --------
    elif input_method == "📷 Use Camera":
        camera_image = st.camera_input("Capture Image")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")

    # -------- Prediction --------
    if image is not None:
        st.image(image, caption="Input Image", use_column_width=True)

        if st.button("Predict"):
            class_name, confidence = predict_mask(image)

            if class_name == "With Mask":
                st.success(f"✅ Prediction: {class_name}")
            else:
                st.error(f"❌ Prediction: {class_name}")

            st.info(f"🔢 Confidence: {confidence:.2f}%")


def examples_page():
    st.title("📌  Real-World Use Cases")

    st.markdown("""
---

### 🌍 Real-World Applications

**🏥 Hospitals & Healthcare Centers**  
- Ensure patients and visitors wear masks  
- Improve safety and hygiene compliance  

---

**✈️ Airports & Railway Stations**  
- Monitor mask usage in crowded public places  
- Assist security and safety teams  

---

**🏫 Colleges & Educational Institutions**  
- Entry-gate monitoring  
- Student safety during health emergencies  

---

**🏢 Offices & Workplaces**  
- Enforce safety rules automatically  
- Reduce manual checking  

---

**📹 CCTV & Smart Surveillance Systems**  
- Automatic detection from camera feeds  
- Works without human supervision  

---

### 🏭 Industry Perspective

Similar deep learning systems are used in:
- Smart city surveillance  
- Access control systems  
- Public safety monitoring  
- AI-powered security solutions   
""")



def faq_page():
    st.title("❓ FAQ")
    with st.expander("❓ What does this app do?"):
        st.write("Detects face mask usage using a CNN model.")

    with st.expander("❓ Does camera work online?"):
        st.write("Yes, Streamlit Cloud supports camera access.")


    with st.expander("❓ Why is my prediction sometimes wrong?"):
        st.write("""
        Wrong prediction can happen due to:
        - Poor image quality
        - Mask not worn properly
        - Face angle or lighting issues
        - Small training dataset
        """)

    with st.expander("❓ Can this model be improved?"):
        st.write("""
        Yes, the model can be improved by:
        - Adding more training images
        - Using data augmentation
        - Training for more epochs
        - Using a deeper CNN architecture
        """)



# 7. MAIN CONTROLLER

def main():
    page = sidebar_navigation()

    if page == "🏠 Home":
        home_page()
    elif page == "📘 How It Works":
        how_it_works_page()
    elif page == "🧠 Concepts Used":
        concepts_page()
    elif page == "🎯 Prediction":
        prediction_page()
    elif page == "📌 Examples & Use Cases":
        examples_page()
    elif page == "❓ FAQ / Common Issues":
        faq_page()



# 8. RUN APP
if __name__ == "__main__":
    main()
