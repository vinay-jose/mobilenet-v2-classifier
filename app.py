from fasthtml.common import *
from fasthtml.components import *
from fastai.vision.all import *
from PIL import Image
import io

style = Style("""
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }
    h2 { color: #2c3e50; }
    img { height: 10%; width: 20%; }
    .image-box { display: flex; }
"""),

app = FastHTML(hdrs=(style))
rt = app.route
learn = load_learner("model.pkl")

def classify_image(img):
    char,idx,probs = learn.predict(img)
    im = Image.open(img).to_thumb(256,256)    
    name = " ".join([s.capitalize() for s in (char).split("_")])
    return name, idx, probs

@rt('/')
def index():
    return Titled("Chair vs Lamp Classifier",
        Body(
            H2("Upload an Image"),
            Button("Upload Image", hx_post="/upload", hx_target="#result"),
            Div(id="result")
            H2("Test Images"),
            Div(
                Img(src="chair1.jpg", hx_trigger="click", hx_get="/classify", hx_target="#result"),
                Img(src="chair2.jpg", hx_trigger="click", hx_get="/classify", hx_target="#result"),
                Img(src="lamp1.jpg", hx_trigger="click", hx_get="/classify", hx_target="#result"),
                Img(src="lamp2.jpg", hx_trigger="click", hx_get="/classify", hx_target="#result"),
                cls="image-box"
            ),
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