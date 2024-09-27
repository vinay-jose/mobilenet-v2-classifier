from fasthtml.common import *
from fasthtml.components import *
from fastai.vision.all import *
import pathlib

style = Style("""
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }
    h2 { color: #2c3e50; }
    img { height: 100px; width: auto; }
    .image-box { display: flex; }
"""),

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

app = FastHTML(hdrs=(style), )
rt = app.route

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner("model.pkl")

def classify_image(image_path):
    img = load_image(image_path)
    char,idx,probs = learn.predict(img)   
    name = " ".join([s.capitalize() for s in (char).split("_")])
    return name, idx, probs

@rt('/')
def index():
    
    return (Titled("Chair vs Lamp Classifier"), 
            Main(H2("Upload an Image"),
                Form(
                    Input(type="file", name="img_file", accept="image/*", required=True),
                    Button("Classify"),
                    enctype="multipart/form-data",
                    hx_post="/classify",
                    hx_target="#result"
                ),
                Br(),
                Div(id="result"),
                H2("Test Images"),
                Div(
                    Img(src="chair1.jpg", hx_trigger="click", hx_get="/classify", 
                        hx_target="#result", hx_vals='{"image_path": "chair1.jpg"}'),
                    Img(src="chair2.jpg", hx_trigger="click", hx_get="/classify", 
                        hx_target="#result", hx_vals='{"image_path": "chair2.jpg"}'),
                    Img(src="lamp1.jpg", hx_trigger="click", hx_get="/classify", 
                        hx_target="#result", hx_vals='{"image_path": "lamp1.jpg"}'),
                    Img(src="lamp2.jpg", hx_trigger="click", hx_get="/classify", 
                        hx_target="#result", hx_vals='{"image_path": "lamp2.jpg"}'),
                    cls="image-box"
                )
            ))
    
@rt('/classify', methods=['GET', 'POST'])
async def classify(img_file: UploadFile|None = None, 
                   image_path:Str|None = None):
    
    if img_file:
        # Save the uploaded image
        image_path = f"uploads/{img_file.filename}"
        with open(image_path, "wb") as f:
            f.write(await img_file.read())
            
    name, idx, probs = classify_image(image_path)    
    return Img(src=image_path), Div(P(f"This is a {name}."),
               P(f"Probability that it's a {name}: {probs[idx]:.4f}"))

serve()