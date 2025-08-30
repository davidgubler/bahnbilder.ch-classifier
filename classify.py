import torch, json, time
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from paddleocr import TextDetection
import os, io, sys
from numpy import asarray
from pymongo import MongoClient

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

od_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", torch_dtype=torch_dtype, trust_remote_code=True).to("cpu")
od_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

td_model = TextDetection(model_name="PP-OCRv5_server_det", limit_side_len=100000)

surya_rec_predictor = RecognitionPredictor(FoundationPredictor())


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



try:
    client = mongo_connect()

    db_bahnbilder = client["bahnbilder"]
    db_bahnbilder_files_original = client["bahnbilder-files-original"]

    coll_photos = db_bahnbilder["photos"]
    coll_files = db_bahnbilder_files_original["files"]
    blacklist_numIds = []

    while True:
        photo_without_texts = coll_photos.find_one({"texts": None, "numId": { "$nin": blacklist_numIds } })
        if photo_without_texts == None:
            break
        blacklist_numIds.append(photo_without_texts["numId"])
        texts = []
        print(photo_without_texts["numId"])
        jpeg = coll_files.find_one({"photoId": photo_without_texts["numId"]})
        if jpeg == None:
            print("jpeg %d is missing!" % photo_without_texts["numId"])
            continue

        image = Image.open(io.BytesIO(jpeg["data"]))
        bounding_box = train_bounding_box(image)
        if bounding_box != None:
            image = image.crop(bounding_box)
            image.save("cropped.jpg", "JPEG")

            # Text detection quality improves a lot with upscaling...
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
            output = td_model.predict(asarray(image), batch_size=1)

            text_images = []

            polygons = []

            for dt_poly in output[0]["dt_polys"]:
                bb = coords_to_bounding_box(dt_poly)
                text_images.append(image.crop(coords_to_bounding_box(dt_poly)))
                i = len(text_images) - 1
                text_images[i].save(f"cropped_text_{i}.jpg")
                polygons.append([dt_poly])

            images = [image] * len(polygons)
            task_names = ["ocr_without_boxes"] * len(polygons)

            predictions_by_image = surya_rec_predictor(
                images=images,
                task_names=task_names,
                polygons=polygons,
                math_mode=False,
            )
            for prediction in predictions_by_image:
                if prediction.text_lines[0].confidence > 0.6:
                    print("%s %.2f" % (prediction.text_lines[0].text, prediction.text_lines[0].confidence))
                    texts.append(prediction.text_lines[0].text)
        else:
            print("train not found!")

        print("setting texts for photo %d to %s" % (photo_without_texts["numId"], json.dumps(texts)))
        filter = { "_id": photo_without_texts["_id"] }
        update_op = { "$set": {"texts": texts } }
        coll_photos.update_one(filter, update_op)
    client.close()

except Exception as e:
    raise Exception("The following error occurred: ", e)


