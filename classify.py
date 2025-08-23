import gc

import torch
from PIL import Image
from PIL.ImageTransform import PerspectiveTransform
from transformers import AutoProcessor, AutoModelForCausalLM
from paddleocr import TextDetection, TextRecognition, TextImageUnwarping
import os, io, sys
from numpy import asarray
from pymongo import MongoClient
import numpy as np

from surya.common.surya.schema import TaskNames
from surya.debug.text import draw_text_on_image
from surya.logging import configure_logging, get_logger
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

od_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", torch_dtype=torch_dtype, trust_remote_code=True).to("cpu")
od_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

td_model = TextDetection(model_name="PP-OCRv5_server_det", limit_side_len=100000)

ocr_model = TextRecognition(model_name="latin_PP-OCRv5_mobile_rec") # not particularly good but avoids mess with non-latin characters
#ocr_model = TextRecognition(model_name="PP-OCRv5_server_rec") # looses spaces
#ocr_model = TextRecognition(model_name="en_PP-OCRv4_mobile_rec") # not great


def mongo_connect():
    mongoHosts = os.environ.get("MONGO_HOSTS")
    mongoUsername = os.environ.get("MONGO_USERNAME")
    mongoPassword = os.environ.get("MONGO_PASSWORD")
    mongoAuthDb = os.environ.get("MONGO_AUTH_DB")
    mongoTls = True
    if mongoHosts == None:
        mongoHosts = "localhost:27017"
        mongoTls = False
    uri = "mongodb://"
    uriCensored = uri
    if mongoUsername != None and mongoPassword != None:
        uri += mongoUsername + ":" + mongoPassword + "@"
        uriCensored += mongoUsername + ":***@"
    uri += mongoHosts + "/"
    uriCensored += mongoHosts + "/"
    if mongoAuthDb != None:
        uri += "?authSource=" + mongoAuthDb
        uriCensored += "?authSource=" + mongoAuthDb
    if mongoTls:
        tlsParam = "?tls=true" if uri.endswith("/") else "&tls=true"
        uri += tlsParam
        uriCensored += tlsParam
    print("connecting to " + uriCensored)
    return MongoClient(uri)


def train_bounding_box(image):
    inputs = od_processor(text="<OD>", images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = od_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False
    )
    generated_text = od_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = od_processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
    try:
        train_index = parsed_answer["<OD>"]["labels"].index("train")
        return parsed_answer["<OD>"]["bboxes"][train_index]
    except Exception:
        return None


def extract_texts_florence2(image):
    inputs = od_processor(text="<OCR>", images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = od_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False
    )
    generated_text = od_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = od_processor.post_process_generation(generated_text, task="<OCR>",
                                                         image_size=(image.width, image.height))
    print(parsed_answer)


def extract_texts_paddle(image):
    output = ocr_model.predict(asarray(image), batch_size=1)
    print(output[0]["rec_text"] + ": " + "%.2f" % output[0]["rec_score"])


def coords_to_bounding_box(coords):
    min_x = 1000000
    min_y = 1000000
    max_x = 0
    max_y = 0
    for coord in coords:
        min_x = min(min_x, coord[0])
        min_y = min(min_y, coord[1])
        max_x = max(max_x, coord[0])
        max_y = max(max_y, coord[1])
    return [min_x, min_y, max_x, max_y]



def image_crop_and_transform(image, coords):
    bounding_box = coords_to_bounding_box(coords)

    print("top left bounding box:     %d %d" % (bounding_box[0], bounding_box[1]))
    print("top right bounding box:    %d %d" % (bounding_box[2], bounding_box[1]))
    print("bottom right bounding box: %d %d" % (bounding_box[2], bounding_box[3]))
    print("bottom left bounding box:  %d %d" % (bounding_box[0], bounding_box[3]))


    np.set_printoptions(threshold=sys.maxsize)
    print(coords)
    print(bounding_box)


try:
    client = mongo_connect()

    db_bahnbilder = client["bahnbilder"]
    db_bahnbilder_files_original = client["bahnbilder-files-original"]

    coll_photos = db_bahnbilder["photos"]
    coll_files = db_bahnbilder_files_original["files"]

    photo_without_texts = coll_photos.find_one({"texts": None})

    if photo_without_texts != None:
        print(photo_without_texts["numId"])
        #jpeg = coll_files.find_one({"photoId": photo_without_texts["numId"]})
        jpeg = coll_files.find_one({"photoId": 60360})
        image = Image.open(io.BytesIO(jpeg["data"]))
        bounding_box = train_bounding_box(image)
        if bounding_box != None:
            print(bounding_box)
            image = image.crop(bounding_box)
            image.save("cropped.jpg", "JPEG")

            # Text detection quality improves a lot with upscaling...
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
            output = td_model.predict(asarray(image), batch_size=1)

            text_images = []
            for dt_poly in output[0]["dt_polys"]:
                bb = coords_to_bounding_box(dt_poly)
                text_images.append(image.crop(coords_to_bounding_box(dt_poly)))
                i = len(text_images) - 1
                text_images[i].save(f"cropped_text_{i}.jpg")

                #extract_texts_florence2(text_images[i])
                #extract_texts_paddle(text_images[i])

                foundation_predictor = FoundationPredictor()
                rec_predictor = RecognitionPredictor(foundation_predictor)
                predictions_by_image = rec_predictor(
                    [image],
                    task_names=["ocr_without_boxes"],
                    # det_predictor=det_predictor,
                    bboxes=[[bb]],
                    #highres_images=loader.highres_images,
                    math_mode=True,
                )
                print("%s %.2f" % (predictions_by_image[0].text_lines[0].text, predictions_by_image[0].text_lines[0].confidence))
                #for key, value in predictions_by_image[0].items():
                #    print(key, value)


        else:
            print("train not found!")

    client.close()

except Exception as e:
    raise Exception("The following error occurred: ", e)


