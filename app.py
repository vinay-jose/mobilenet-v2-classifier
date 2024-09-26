from fasthtml.common import *
from fasthtml.components import *
from fastai.vision.all import *
from PIL import Image
import io

app, rt = fast_app()
learn = load_learner("model.pkl")

def classify_image(img):
    char,idx,probs = learn.predict(img)
    im = Image.open(img).to_thumb(256,256)    
    name = " ".join([s.capitalize() for s in (char).split("_")])
    return name, idx, probs

@rt('/')
def index():
    return Titled("Chair vs Lamp Classifier",
        Div(
            H2("Example Images"),
            Div(
                Img(src="chair1.jpg", hx_trigger="click", hx_get="/classify", hx_target="#result", height=20%, width=10%),
                Img(src="chair2.jpg", hx_trigger="click", hx_get="/classify", hx_target="#result"),
                Img(src="lamp1.jpg", hx_trigger="click", hx_get="/classify", hx_target="#result"),
                Img(src="lamp2.jpg", hx_trigger="click", hx_get="/classify", hx_target="#result"),
                cls="flex flex-wrap justify-center gap-4"
            ),
            H2("Upload an Image"),
            Button("Upload Image", hx_post="/upload", hx_target="#result"),
            Div(id="result")
        )
    )

@rt('/classify')
def classify(img_file: UploadFile):
    img_bytes = img_file.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes))
    
    name, idx, probs = classify_image(img)
    return Div(Div(f"This is {name}."),
               Div(f"Probability it's {name}: {probs[idx]:.4f}"))
    
@rt('/upload', methods=['POST'])
def upload(img_file: UploadFile):
    img_bytes = img_file.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes))
    
    name, idx, probs = classify_image(img)    
    return Div(Div(f"This is {name}."),
               Div(f"Probability it's {name}: {probs[idx]:.4f}"))

serve()