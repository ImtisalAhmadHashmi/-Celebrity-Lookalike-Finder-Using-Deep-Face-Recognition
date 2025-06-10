from keras_vggface.utils import preprocess_input
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
import os
import tempfile

def face_detection(img, face_detector):
    inferencing_img = cv2.imread(img)
    inferencing_img = cv2.cvtColor(inferencing_img, cv2.COLOR_BGR2RGB)
    img_face = face_detector.detect_faces(inferencing_img)

    if len(img_face) == 0:
        raise ValueError("No face detected in the image")

    x,y,width,height = img_face[0]["box"]

    face = inferencing_img[y:y+height, x:x+width]

    return face

def extract_features(img, face_detector, model ):
    face = face_detection(img, face_detector)

    image = Image.fromarray(face)
    image = image.resize((224,224))

    face_array = np.asarray(image)

    face_array = face_array.astype("float32")

    expanded_img = np.expand_dims(face_array, axis = 0)
    preprocess_img = preprocess_input(expanded_img)
    results = model.predict(preprocess_img).flatten()

    return results

def find_similarity(img, face_detector, model, embedding_list):
    embedding_1 = extract_features(img, face_detector, model)
    similarity = []

    embedding_1 = embedding_1.reshape(1, -1)

    for i in range(len(embedding_list)):
        embedding_2 = np.array(embedding_list[i]).reshape(1, -1)

        score = cosine_similarity(embedding_1, embedding_2)
        similarity.append(score[0][0])

    highest_similarity_score = sorted(list(enumerate(similarity)), reverse = True, key =lambda x:x[1])

    index, highest_score = highest_similarity_score[0]

    return index, highest_score

def resemblance_img(img_index, filepaths):

    img = cv2.imread(filepaths[img_index])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def save_uploaded_img(uploaded_img):
    try:
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, uploaded_img.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_img.getbuffer())
        return save_path
    except Exception as e:
        return False






