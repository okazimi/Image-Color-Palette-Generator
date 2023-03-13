# IMPORTS
import base64
import io
import os
# IMPORTS FOR SERVER/WEB DEV
from flask import Flask, render_template, request, url_for, redirect, flash
from flask_bootstrap import Bootstrap
# IMPORTS FOR IMAGE PROCESSING
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
# IMPORT FOR MATLAB PLOT
import matplotlib.pyplot as plt
# IMPORTS FOR FILE PROCESSING
from werkzeug.utils import secure_filename


# INITIALIZE APPLICATION
app = Flask(__name__)
# INITIALIZE SECRET KEY
app.secret_key = os.environ.get("application_secret_key")
# INITIALIZE UPLOAD FOLDER
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static\images')
# INITIALIZE BOOTSTRAP
Bootstrap(app)


# # SHOW IMAGE
# def show_img_compare(img_1, img_2):
#     f, ax = plt.subplots(1, 2, figsize=(10, 10))
#     ax[0].imshow(img_1)
#     ax[1].imshow(img_2)
#     ax[0].axis('off')
#     ax[1].axis('off')
#     f.tight_layout()
#     plt.show()


# RBG TO HEX CONVERTER
def rgb_to_hex(r, g, b):
    # RETURN CORRESPONDING HEX VALUE
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


# PALETTE GENERATOR
def palette_generator(clusters):
    # INITIALIZE WIDTH VARIABLE
    width = 300
    # CREATE 50 ARRAYS (300ROWS*3COLUMNS) OF ZEROS
    # SET DATA TYPE TO "NP.UNIT8" (ONLY POSITIVE NUMBERS) AS RBG VALUES RANGE FROM 0-255
    colors = np.zeros((50, width, 3), np.uint8)
    # DIVIDE NUMBER OF ROWS IN ZERO ARRAY BY THE NUMBER OF ROWS IN THE CLUSTERS
    # CLUSTERS = TRAINED ARRAY FROM KMEANS
    steps = width / clusters.cluster_centers_.shape[0]
    # FOR EACH CLUSTER
    for idx, centers in enumerate(clusters.cluster_centers_):
        # UPDATE COLORS
        colors[:, int(idx * steps):(int((idx + 1) * steps)), :] = centers
    # RETURN COLORS
    return colors


# CONVERT IMAGE TO BYTES
def convert_image_to_bytes(image):
    # INITIALIZE BYTES OBJECT
    data = io.BytesIO()
    # SAVE IMAGE AS BYTES
    image.save(data, "JPEG")
    # ENCODE IMAGE DATA
    encoded_img_data = base64.b64encode(data.getvalue())
    # RETURN ENCODED IMAGE DATA
    return encoded_img_data


# CLEAR STATIC FOLDER
def clear_static_folder():
    # LOOP THROUGH FILES IN STATIC FOLDER
    for file_name in os.listdir(app.config["UPLOAD_FOLDER"]):
        # REMOVE EACH FILE
        os.remove(app.config["UPLOAD_FOLDER"]+f"/{file_name}")


# HEX CODE GENERATOR
def hex_code_generator(color_palette):
    # INITIALIZE HEX CODE LIST (CONTAINS HEX CODES THAT WERE CONVERTED FROM RGB VALUES)
    hex_codes = []
    # INITIALIZE ARRAY INDEX (USED TO OBTAIN EACH COLOR, EACH COLOR SEPARATED BY APPROX. 60 ROWS)
    array_index = 0
    # LOOP THROUGH COLORS IN COLOR PALETTE
    for color in color_palette:
        # AT THE END OF THE COLOR PALETTE ARRAY
        if array_index >= 300:
            # END LOOP/BREAK
            break
        # NOT AT THE END OF THE COLOR PALETTE ARRAY
        else:
            # OBTAIN ARRAY CONTAINING COLOR/RGB VALUES ([R,B,G])
            array = color[array_index]
            # ASSIGN VALUE AT INDEX 0 OF ARRAY TO "R"
            r = array[0]
            # ASSIGN VALUE AT INDEX 1 OF ARRAY TO "G"
            g = array[1]
            # ASSIGN VALUE AT INDEX 2 OF ARRAY TO "B"
            b = array[2]
            # CONVERT OBTAINED RBG VALUES TO HEX CODE AND ADD TO THE HEX CODES LIST
            hex_codes.append(rgb_to_hex(r, g, b))
            # INCREMENT ARRAY INDEX TO OBTAIN FOLLOWING COLOR
            array_index += 60
    # RETURN HEX CODES
    return hex_codes


# HOME PAGE (REQUESTS: GET, POST)
@app.route("/", methods=["GET", "POST"])
def home():
    # START WITH CLEAN STATIC FOLDER
    clear_static_folder()
    # USER CLICKS UPLOAD BUTTON
    if request.method == "POST":
        # IF USER UPLOADED FILE (NOT EMPTY)
        if request.files["file"].filename != "":
            # OBTAIN FILE FROM USER
            file = request.files['file']
            # # SECURE FILE
            # filename = secure_filename(file.filename)
            # SAVE FILE AS "IMAGE.JPG" IN APPLICATION'S STATIC/IMAGES FOLDER
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg"))
            # REDIRECT USER TO PALETTE SCREEN
            return redirect(url_for('palette'))
        # USER DID NOT UPLOAD FILE (EMPTY)
        else:
            # FLASH MESSAGE AND PROMPT USER TO UPLOAD FILE
            flash("Please select a file")
            # REDIRECT USER TO HOME PAGE
            return redirect(url_for('home'))
    # RETURN HOME PAGE
    return render_template("index.html")


# PALETTE PAGE (REQUESTS: GET, POST)
@app.route("/palette", methods=["GET", "POST"])
def palette():
    # GENERATE COLOR PALETTE
    try:
        # OPEN THE USER'S UPLOADED IMAGE FROM STATIC/IMAGES FOLDER
        original_image = Image.open("static/images/image.jpg")
        # RESIZE IMAGE FOR FASTER PROCESSING
        original_image = original_image.resize((200, 200))
        # CONVERT IMAGE TO AN ARRAY
        array = np.asarray(original_image)
        # INITIALIZE K-MEANS
        # K-MEANS GROUPS A COLLECTION OF DATA POINTS TOGETHER DUE TO SIMILARITIES (A.K.A CLUSTERS)
        # THE NUMBER OF GROUPS IS BASED ON THE SUPPLIED NUMBER OF CENTROIDS (N_CLUSTERS), WHICH IN OUR CASE IS 5
        # https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
        clt = KMeans(n_init='auto', n_clusters=5)
        # RESHAPE ARRAY TO 1D ARRAY WITH 3 COLUMNS
        pixel_values = array.reshape(-1, 3)
        # PASS PIXEL VALUES TO K-MEANS AND TRAIN MODEL USING .FIT FUNCTION
        trained_model = clt.fit(pixel_values)
        # PASS TRAINED MODEL TO PALETTE GENERATOR
        color_palette = palette_generator(trained_model)
        # PASS GENERATED COLOR PALETTE TO HEX CODE GENERATOR
        hex_codes = hex_code_generator(color_palette)
        # CONVERT 3D PALETTE ARRAY TO IMAGE
        palette_image = Image.fromarray(color_palette)
        # SAVE PALETTE AS IMAGE
        palette_image.save(os.path.join(app.config['UPLOAD_FOLDER'], "palette.jpg"))
        # CONVERT ORIGINAL IMAGE TO BYTES
        encoded_original_image = convert_image_to_bytes(original_image)
        # CONVERT IMAGE PALETTE TO BYTES
        encoded_palette_image = convert_image_to_bytes(palette_image)
        # # DISPLAY IMAGE AND COLOR PALETTE USING MATLAB PLOT
        # show_img_compare(img, color_palette)
        # RETURN PALETTE PAGE (PASS IMAGE, COLOR PALETTE AND HEX CODES)
        return render_template("palette.html", image=encoded_original_image.decode("utf-8"), palette=encoded_palette_image.decode("utf-8"), hex_codes=hex_codes)
    # INVALID FILE UPLOADED
    except OSError:
        # FLASH MESSAGE AND PROMPT USER TO UPLOAD FILE
        flash("File selected is invalid. Please try again.")
        # REDIRECT USER TO HOME PAGE
        return redirect(url_for('home'))


# RUN APPLICATION IN DEBUG MODE IF NAME OF FILE IS MAIN
if __name__ == "__main__":
    app.run(debug=True)
