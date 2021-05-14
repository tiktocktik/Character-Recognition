from scipy import io
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from flask import Flask, render_template, request , url_for , redirect
from segment import word_Segmentation , character_Segmentation, singleCharacterSegmentation
import os , sys

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

MAIN_FOLDER = os.getcwd()
train_data = io.loadmat(MAIN_FOLDER+'\emnist-bymerge.mat', struct_as_record=False, squeeze_me=True)

data_dict = _todict(train_data['dataset'])

UPLOAD_FOLDER = '/static/uploads/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','jfif','bmp'])

print(MAIN_FOLDER)

model = tf.keras.models.load_model(MAIN_FOLDER+'\model')

app = Flask(__name__)

def predictImage(dirInput):
    WordList = []
    imageFiles = os.listdir(dirInput)
    for x in imageFiles:
        innerFiles = os.listdir(dirInput+'%s'%(x))
        for a in innerFiles:
            im = Image.open(dirInput+'%s\\%s'%(x,a)).convert('L')
            width = float(im.size[0])
            height = float(im.size[1])
            newImage = Image.new('L', (28, 28), (255))

            if width > height:
                nheight = int(round((20.0 / width * height), 0))
                if (nheight == 0):
                    nheight = 1
                img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wtop = int(round(((28 - nheight) / 2), 0))
                newImage.paste(img, (4, wtop))
            else:
                nwidth = int(round((20.0 / height * width), 0))
                if (nwidth == 0):
                    nwidth = 1
                img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wleft = int(round(((28 - nwidth) / 2), 0))
                newImage.paste(img, (wleft, 4))

            tv = list(newImage.getdata())

            tva = [(255 - x) * 1.0 / 255.0 for x in tv]

            newx=np.array(tva).reshape(28,28)
            newx=newx.T
            pred = model.predict(newx.reshape(1, 28, 28, 1))
            myDict = {idx : data_dict['mapping'][idx] for idx in range(len(data_dict['mapping']))}
            WordList.append(chr(myDict.get(pred.argmax())[1]))
            print(WordList)
        WordList.append(' ')
    f = open('output.txt','a')
    for x in range(len(WordList)):
        f.write(str(WordList[x]))
    f.write('\n')
    f.close()
    return WordList

def predictCharImage(dirInput):
    Charlist = []
    imageFiles = os.listdir(dirInput)
    for x in imageFiles:
        im = Image.open(dirInput+'\\%s'%(x)).convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L', (28, 28), (255))

        if width > height:
            nheight = int(round((20.0 / width * height), 0))
            if (nheight == 0):
                nheight = 1
            img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight) / 2), 0))
            newImage.paste(img, (4, wtop))
        else:
            nwidth = int(round((20.0 / height * width), 0))
            if (nwidth == 0):
                nwidth = 1
            img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth) / 2), 0))
            newImage.paste(img, (wleft, 4))

        tv = list(newImage.getdata())

        tva = [(255 - x) * 1.0 / 255.0 for x in tv]

        newx=np.array(tva).reshape(28,28)
        newx=newx.T
        pred = model.predict(newx.reshape(1, 28, 28, 1))
        myDict = {idx : data_dict['mapping'][idx] for idx in range(len(data_dict['mapping']))}
        Charlist.append(chr(myDict.get(pred.argmax())[1]))
    Charlist.append(' ')
    mystr = ''
    print(Charlist)
    f = open('outputChar.txt','a')
    for x in range(len(Charlist)):
        f.write(str(Charlist[x]))
        # mystr + str(Charlist[x])
    f.write('\n')
    f.close()   
    # print(mystr) 
    return Charlist

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('upload.html')

    
@app.route('/multi', methods = ['GET','POST'])
def multi():
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template(('multi.html'), msg='No file selected')
        file = request.files['file']
        test = file.filename
        test = test.split('.')
        if test[1] not in ALLOWED_EXTENSIONS:
            return render_template(('multi.html'), msg='file not supported!')
        # if no file is selected
        if file.filename == '':
            return render_template(('multi.html'), msg='No file selected')

        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))
            filename = file.filename
            myFile = os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename)
            segmentedWord = word_Segmentation(myFile,filename)
            segmentedChar = character_Segmentation(segmentedWord,filename)
            # extract the text and display it
            return render_template(('multi.html'),
                                   msg='Successfully processed',
                                   extracted_text=predictImage(segmentedChar),
                                   img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template(('multi.html'))

@app.route('/single', methods = ['GET','POST'])
def single():
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template(('single.html'), msg='No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template(('single.html'), msg='No file selected')
        file = request.files['file']
        test = file.filename
        test = test.split('.')
        if test[1] not in ALLOWED_EXTENSIONS:
            return render_template(('multi.html'), msg='file not supported!')
        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))
            filename = file.filename
            myFile = os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename)
            extracted_text = singleCharacterSegmentation(myFile,filename)
            
            # extract the text and display it
            return render_template(('single.html'),
                                   msg='Successfully processed',
                                   extracted_text=predictCharImage(extracted_text),
                                   img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template(('single.html'))
if __name__ == '__main__':
    app.run(debug=True)