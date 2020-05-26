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
        response = make_response(open('favicon.ico','rb').read())
        response.headers['Content-Type'] = 'image/png'
        return response

@app.route('/entry')
def entry():
    return redirect('/index')

@app.route('/index', methods = ['GET', 'POST'])
def entry_page() -> 'html':
    return render_template('index.html')

@app.route('/image/upload/content', methods = ['GET', 'POST'])
def upload_content():
    if request.method == 'POST':
        f = request.files.get('image')
        if allowed_file(f.filename):
            try:
                f.save('./image/upload/content/content.png')
            except BaseException:
                print('save error!')
            return render_template('index.html')
        else :
            # 异常处理！
            return render_template('index.html')

@app.route('/image/upload/style', methods = ['GET', 'POST'])
def upload_style():
    if request.method == 'POST':
        f = request.files.get('image')
        if allowed_file(f.filename):
            f.save('./image/upload/style/style.png')
            return render_template('index.html')
        else:
            return render_template('index.html')

@app.route('/submit',methods = ['GET','POST'])
def submit():
    # 执行风格化函数，生成图像。
    contentpath = abspath('image/upload/content/content.png')
    stylepath = abspath('image/upload/style/style.png')
    convertpath = abspath('image/upload/convert/convert.png')
    transfer(contentpath, stylepath, convertpath)
    return render_template('submissions.html')

@app.route('/image/upload/content/content.png', methods = ['GET', 'POST'])
def content():
    if request.method == 'GET':
        response = make_response(open('image/upload/content/content.png','rb').read())
        response.headers['Content-Type'] = 'image/png'
        return response

@app.route('/image/upload/style/style.png', methods = ['GET', 'POST'])
def style():
    if request.method == 'GET':
        response = make_response(open('image/upload/style/style.png','rb').read())
        response.headers['Content-Type'] = 'image/png'
        return response

@app.route('/image/upload/convert/convert.png', methods = ['GET', 'POST'])
def convertoutput():
    if request.method == 'GET':
        response = make_response(open('image/upload/convert/convert.png','rb').read())
        response.headers['Content-Type'] = 'image/png'
        return response

@app.route('/makePrivate', methods = ['GET', 'POST'])
def makePrivate():
    return render_template('submissions.html')

if __name__ == '__main__':
    app.run(debug=True)

