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

from os.path import join, dirname, realpath

# 保证安全，只接受图片
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def base():
    return redirect('/entry')

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
            f.save('./app/image/upload/content/content.png')
            return render_template('index.html')
        else :
            # 异常处理！
            return render_template('index.html')

@app.route('/image/upload/style', methods = ['GET', 'POST'])
def upload_style():
    if request.method == 'POST':
        f = request.files.get('image')
        f.save('./app/image/upload/style/style.png')
        return render_template('index.html')

@app.route('/submit',methods = ['GET','POST'])
def submit():
    # 执行风格化函数，生成图像。
    return render_template('submissions.html')

@app.route('/image/upload/content/content.png', methods = ['GET', 'POST'])
def content():
    if request.method == 'GET':
        response = make_response(open('./app/image/upload/content/content.png','rb').read())
        response.headers['Content-Type'] = 'image/png'
        return response

@app.route('/image/upload/style/style.png', methods = ['GET', 'POST'])
def style():
    if request.method == 'GET':
        response = make_response(open('./app/image/upload/style/style.png','rb').read())
        response.headers['Content-Type'] = 'image/png'
        # ty(content, style, pixel)#开始转换
        return response

@app.route('/makePrivate', methods = ['GET', 'POST'])
def makePrivate():
    return render_template('submissions.html')

# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_files():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save('figure.png')
#         print('file uploaded successfully')
#         return render_template('ok.html')
#     #     response = make_response(f)
#     #     #   f.save(secure_filename(f.filename))
#     #     response.headers['Content-Type'] = 'image/png'
#     #     return response
#     # else :
#     #     pass

# @app.route('/show', methods=['GET'])
# def show_photo():
#     if request.method == 'GET':
#         response = make_response(open('figure.png','rb').read())
#         response.headers['Content-Type'] = 'image/png'
#         return response
#     else :
#         pass
    # file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    # if request.method == 'GET':
    #     if filename is None:
    #         pass
    #     else:
    #         image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
    #         response = make_response(image_data)
    #         response.headers['Content-Type'] = 'image/png'
    #         return response
    # else:
    #     pass


if __name__ == '__main__':
    app.run()

