import streamlit as st
from inferencing_functions import save_uploaded_img, resemblance_img, find_similarity
from PIL import Image
from mtcnn import MTCNN
import pickle
from keras_vggface.vggface import VGGFace

# --- Configuration ---
st.set_page_config(
    page_title="Celeb Match ",
    page_icon="👬",
    layout="centered"
)

filenames = pickle.load(open("filenames.pkl", "rb"))

senet_feature_list = pickle.load(open("senet_embeddings.pkl", "rb"))
@st.cache(allow_output_mutation=True)
def load_senet_model():
    model = VGGFace(model = "senet50", include_top = False, input_shape=(224, 224, 3), pooling="avg")
    model.load_weights("rcmalli_vggface_tf_notop_senet50.h5")
    return model
senet_model = load_senet_model()

resnet_feature_list = pickle.load(open("embeddings.pkl", "rb"))
@st.cache(allow_output_mutation=True)
def load_resnet_model():
    return VGGFace(model = "resnet50", include_top = False, input_shape=(224, 224, 3), pooling="avg")
resnet_model = load_resnet_model()

def selection(option):
    if option == "ResNet":
        model = resnet_model
        feature_list = resnet_feature_list
    else:
        model = senet_model
        feature_list = senet_feature_list

    return model, feature_list

detector = MTCNN()


st.title("Which Celebrity Resembles You")

option = st.selectbox(
    "Choose a model to find your resemblance with celebrities (Hollywood / Bollywood / Sports / Politics):",
    options=["ResNet", "SENet"]
)

upload_img = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

model, feature_list = selection(option)

if upload_img is not None:
    saved_path = save_uploaded_img(upload_img)

    if saved_path:

        display_img = Image.open(upload_img)
        orig_width, orig_height = display_img.size

        try:
            index, score = find_similarity(saved_path, detector, model, feature_list)


            resembling_celeb = " ".join(filenames[index].split("\\")[1].split()[:2])

            # Get resemblance image with correct color and size
            resemblance_img_array = resemblance_img(index, filenames)
            resemblance_img_pil = Image.fromarray(resemblance_img_array).resize((orig_width, orig_height))

            col1, col2 = st.beta_columns(2)

            with col1:
                st.header("Your Uploaded Image")
                st.image(display_img, width = 300)

            with col2:
                st.header(f"{resembling_celeb}.")
                st.image(resemblance_img_pil, width = 300)
            st.markdown(f"### Your resemblance with {resembling_celeb}: {score*100:.2f}%")
            st.progress(float(score))

        except ValueError as e:
            st.error(str(e))
            st.stop()

