#intalling dependencies
#pip install tensorflow==2.3.1
#pip install mtcnn==0.1.0
#pip install keras==2.4.3
#pip install keras_vggface==0.6
#pip install keras_applications==1.0.8
# pip install numPy==1.18.5
#pip install protobuf==3.20.*
#pip install scikit-learn==0.23.2 --no-deps
#pip install scipy==1.4.1 joblib>=0.11 threadpoolctl>=2.0.0
#pip install streamlit==0.82.0 --no-deps
# pip install pandas==1.2.5 altair==4.2.2 protobuf==3.20.*
#pip install astor base58 tzlocal validators click==7.1.2

import os
import pickle
actors = os.listdir("Data")
print(actors)

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join("Data", actor)):
        filenames.append(os.path.join("Data",actor, file ))

print(filenames)
print(len(filenames))

pickle.dump(filenames,open("filenames.pkl", 'wb'))


from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open("filenames.pkl", "rb"))

def feature_extractor(img_path, feature_model):
    img = image.load_img(img_path, target_size = (224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = feature_model.predict(preprocessed_img).flatten()

    return result

res_model = VGGFace(model = "resnet50", include_top = False, input_shape=(224, 224, 3), pooling="avg")

res_features = []
for file in tqdm(filenames):
    res_features.append(feature_extractor(file, res_model))
pickle.dump(res_features,open("embeddings.pkl", "wb"))


se_model = VGGFace(model = "senet50", include_top = False, input_shape=(224, 224, 3), pooling="avg")

se_features = []
for file in tqdm(filenames):
    se_features.append(feature_extractor(file, se_model))
pickle.dump(se_features,open("senet_embeddings.pkl", "wb"))




