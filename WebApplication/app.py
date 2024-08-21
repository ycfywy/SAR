from flask import Flask,request,jsonify,render_template,send_file,send_from_directory
import os
import io
from PIL import Image
from Sam import segment
import numpy as np
import cv2
from VIT.predict import recognize,recognize2
import base64
import sys



app = Flask(__name__)


masks = []


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER =os.path.join(BASE_DIR, 'static/upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


for path in sys.path:
    print(path)


if not os.path.exists(UPLOAD_FOLDER):
    print("有没有")
    os.makedirs(UPLOAD_FOLDER)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
 
@app.route("/returnimgs",methods = ['GET'])
def returnimgs():
    print(os.getcwd())
    mask = encode_image(os.path.join(os.getcwd(),'static/segment/mask.jpg'))
    cropped = encode_image(os.path.join(os.getcwd(),'static/segment/cropped.jpg'))
    
    return jsonify({
        'mask':mask,
        'cropped':cropped
    })
@app.route("/showimgs",methods = ['GET'])
def showimgs():
    return render_template("showimgs.html")

@app.route("/button",methods = ['GET'])
def button():
    return render_template("send.html")
@app.route("/loadbar",methods = ['GET'])
def loadbar():
    return render_template("loadbar.html")

@app.route("/predict",methods = ['GET'])
def predict():
    recognize2()
    return render_template("predict.html")

@app.route("/",methods = ['GET']) 
def index():
    return render_template("index.html")

@app.route("/component",methods = ['GET'])
def component():
    return render_template("component.html")    

@app.route("/process",methods = ['POST'])
def process():
    data = request.json
    print(data)
    response_data = f"I have receiced the data {data['data']}"
    return jsonify({'message':response_data})

@app.route("/upload",methods = ["POST"])
def upload():
    print(request.files)
    if 'image' not in request.files:
        return jsonify({"errro":"image not exist"})
    file = request.files['image']
    if file:
        filename = 'user.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(filepath)
        file.save(filepath)
        image = Image.open(file)
        print("=================")
        image = image.resize((1000,800))
        print("=================")
        image_ndarray = np.array(image.convert('RGB'))
        print(image_ndarray.shape)
        global masks
        masks = segment(image_ndarray)
        print("length of masks: ",len(masks))
        return send_from_directory('static/segment', 'segment.jpg')


@app.route("/click_image",methods = ["POST"])
def click_image():
    coordinate = request.json
    print(coordinate)
    x = int(coordinate['x'])
    y = int(coordinate['y'])
    print(x,y)
    
    image = Image.open('static/upload/user.jpg')
    image = image.resize((1000,800))
    image_ndarray = np.array(image) 
    image_ndarray = cv2.cvtColor(image_ndarray, cv2.COLOR_RGB2BGR)
    for index,mask in enumerate(masks):
        print(mask['segmentation'])
        print(mask['segmentation'].shape)
        if mask['segmentation'][y][x] == True:
            print(f"find index {index}")
            mask = mask['segmentation']
            mask_ndarray = mask.astype(np.uint8)*255
            cv2.imwrite('static/segment/mask.jpg',mask_ndarray)
            contours, _ = cv2.findContours(mask_ndarray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(contours[0])
            cropped = image_ndarray[y:y+h,x:x+w]
            cropped = cv2.resize(cropped,(224,224))
            cv2.imwrite('static/segment/cropped.jpg',cropped)


            mask = encode_image(os.path.join(os.getcwd(),'static/segment/mask.jpg'))
            cropped = encode_image(os.path.join(os.getcwd(),'static/segment/cropped.jpg'))
            
            return jsonify({
                'mask':mask,
                'cropped':cropped
            })
            
    return jsonify({"message":"fail"})




@app.route("/click_classify",methods = ["POST","GET"])
def classify():
    result =  recognize()
    print("get result")
    print(result)
    return jsonify(result)

    

if __name__ == '__main__':
    app.run(debug=True)