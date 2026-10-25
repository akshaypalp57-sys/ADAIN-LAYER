import os
import torch
from flask import Flask, render_template, request, send_from_directory, jsonify, url_for
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Import your existing AdaIN Code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Bootstrap(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(
        '/Users/akshaypal/FIRST_NST.PJ/experiment/big_dataset/decoder_10.pth',
        map_location=device
    )
)

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    # Standard VGG Normalization & Preprocessing (No noise)
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_features = encoder(content_tensor)
        style_features = encoder(style_tensor)

        c_feat = content_features[-1]
        s_feat = style_features[-1]

        # 1. Standard AdaIN target features
        target = adaptive_instance_normalization(c_feat, s_feat)

        # 2. Clean Alpha Blending
        if alpha == 0.0:
            interpolated_feats = c_feat
        elif alpha == 1.0:
            interpolated_feats = target
        else:
            interpolated_feats = (1 - alpha) * c_feat + alpha * target

        # 3. Decoder generation with strict clamping to avoid artifacts
        generated = decoder(interpolated_feats)
        generated = torch.clamp(generated, 0, 1)

    return generated
def save_image(image, path, original_size=None):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)

    if original_size:
        image = image.resize(
            original_size,
            Image.Resampling.LANCZOS
        )

    image.save(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    alpha = 0.5
    if request.method == 'POST':
        try:
            alpha = float(request.form.get('alpha', 0.5))
        except ValueError:
            alpha = 0.5

    print(f"========== ALPHA DEBUG ==========")
    print("Alpha received from UI:", alpha)

    if request.method == 'POST':
        content_file = request.files.get('content')
        style_file = request.files.get('style')

        content_filename = request.form.get('content_path')
        style_filename = request.form.get('style_path')

        if content_file and content_file.filename and allowed_file(content_file.filename):
            content_filename = secure_filename(content_file.filename)
            content_file.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))

        if style_file and style_file.filename and allowed_file(style_file.filename):
            style_filename = secure_filename(style_file.filename)
            style_file.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            if os.path.exists(content_path) and os.path.exists(style_path):
                try:
                    content_image = Image.open(content_path).convert('RGB')
                    style_image = Image.open(style_path).convert('RGB')

                    print("Running style transfer...")
                    stylized_image = style_transfer(
                        content_image,
                        style_image,
                        encoder,
                        decoder,
                        alpha,
                        device
                    )

                    result_filename = 'stylized_' + content_filename
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

                    save_image(
                        stylized_image,
                        result_path,
                        original_size=content_image.size
                    )

                    result_image = result_filename
                    print("Style transfer completed successfully!")

                    # Return JSON for AJAX requests (No page reload!)
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({
                            'success': True,
                            'result_image_url': url_for('send_image', filename=result_image),
                            'content_path': content_filename,
                            'style_path': style_filename
                        })

                except Exception as e:
                    error = str(e)
                    print("Error during transfer:", error)
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'success': False, 'error': error})
            else:
                error = "Uploaded image files could not be found on server."
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': error})
        else:
            error = 'Please upload both content and style images.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': error})

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': False, 'error': error or 'Invalid request'})

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
        alpha=alpha
    )

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )

@app.route('/examples/<path:filename>')
def send_examples(filename):
    return send_from_directory(
        'examples',
        filename
    )

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple(
        'localhost',
        5001,
        app,
        use_reloader=True,
        use_debugger=True
    )