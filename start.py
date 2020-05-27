# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from flask import Flask , render_template, request, redirect, make_response
# from werkzeug import secure_filename
from flask_dropzone import Dropzone
app = Flask(__name__)

from flask_dropzone import Dropzone
dropzone = Dropzone()
dropzone.init_app(app)

from os.path import join, dirname, realpath, abspath

from convert import transfer

import time

import random

from os import remove, listdir

# 保证安全，只接受图片
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF', 'jpeg', 'GPEG','pdf', 'PDF','bmp','jpg','png','tif','gif','pcx','tga','exif','fpx','svg','psd','cdr','pcd','dxf','ufo','eps','ai','raw','WMF','webp','jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def base():
    return redirect('/entry')

@app.route('/favicon.ico', methods = ['GET'])
def favicon():
    if request.method == 'GET':
        ico = open('favicon.ico','rb')
        response = make_response(ico.read())
        response.headers['Content-Type'] = 'image/png'
        ico.close()
        return response

@app.route('/entry')
def entry():
    return redirect('/index')

@app.route('/index', methods = ['GET', 'POST'])
def entry_page() -> 'html':
    return render_template('index.html',rand=str(random.randint(1,(1<<31))))

@app.route('/image/upload/content/<string:filename>', methods = ['GET', 'POST'])
def upload_content(filename):
    # 上传到服务器保存
    if request.method == 'POST':
        f = request.files.get('image')
        if allowed_file(f.filename):
            try:
                f.save('image/upload/content/'+filename)
            except BaseException:
                return 'Sorry, save error, Please upload a figure, try again.'
            return render_template('index.html',rand=filename[7:-4])
        else :
            # 异常处理！
            return 'Sorry, save error, Please upload a figure, try again.'
    # 从服务器到前端
    elif request.method == 'GET':
        image = open('image/upload/content/'+filename,'rb')
        response = make_response(image.read())
        response.headers['Content-Type'] = 'image/png'
        image.close()
        return response

@app.route('/image/upload/style/<string:filename>', methods = ['GET', 'POST'])
def upload_style(filename):
    # 给服务器保存
    if request.method == 'POST':
        f = request.files.get('image')
        if allowed_file(f.filename):
            try:
                f.save('image/upload/style/'+filename)
            except BaseException:
                return 'Sorry, save error, Please upload a figure, try again.'
            return render_template('index.html',rand=filename[5:-4])
        else:
            return 'Sorry, save error, Please upload a figure, try again.'
    # 从服务器获取
    elif request.method == 'GET':
        image = open('image/upload/style/'+filename,'rb')
        response = make_response(image.read())
        response.headers['Content-Type'] = 'image/png'
        image.close()
        return response

@app.route('/submit/<string:rand>',methods = ['GET','POST'])
def submit(rand):
    # 执行风格化函数，生成图像。
    contentpath = 'image/upload/content/content'+rand+'.png'
    stylepath = 'image/upload/style/style'+rand+'.png'
    convertpath = 'image/upload/convert/convert'+rand+'.png'
    transfer(contentpath, stylepath, convertpath)
    return render_template('submissions.html',rand=rand)

@app.route('/image/upload/convert/<string:filename>', methods = ['GET', 'POST'])
def convertoutput(filename):
    if request.method == 'GET':
        image = open('image/upload/convert/'+filename,'rb')
        response = make_response(image.read())
        response.headers['Content-Type'] = 'image/png'
        image.close()
        return response

@app.route('/makePrivate/<string:rand>', methods = ['GET', 'POST'])
def makePrivate(rand):
    remove('image/upload/content/content'+rand+'.png')
    remove('image/upload/style/style'+rand+'.png')
    remove('image/upload/convert/convert'+rand+'.png')
    return redirect('/index')

@app.route('/show',methods = ['GET','POST'])
def show():
    files = []
    for f in listdir('image/upload/convert/'):
        if f != 'README.md':
            files.append(f[7:-4])
    return render_template('show.html',result=files)


if __name__ == '__main__':
    app.run(debug=True)

