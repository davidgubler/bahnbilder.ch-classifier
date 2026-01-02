import torch, json, time
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration, LlavaForConditionalGeneration
from paddleocr import TextDetection
import os, io, sys
from numpy import asarray
from pymongo import MongoClient

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor

od_model = Florence2ForConditionalGeneration.from_pretrained("florence-community/Florence-2-base", dtype=torch.bfloat16, device_map="cpu")
od_processor = AutoProcessor.from_pretrained("florence-community/Florence-2-base")

td_model = TextDetection(model_name="PP-OCRv5_server_det", limit_side_len=100000)

surya_rec_predictor = RecognitionPredictor(FoundationPredictor())

caption_model_name = "fancyfeast/llama-joycaption-beta-one-hf-llava"
caption_processor = AutoProcessor.from_pretrained(caption_model_name)
caption_model = LlavaForConditionalGeneration.from_pretrained(caption_model_name, dtype="bfloat16", device_map="cpu")
caption_model_prompt = "Write a list of Booru-like tags for this image within 40 words."
caption_model_convo = [
    {"role": "system", "content": "You are a helpful image captioner."},
    {"role": "user", "content": caption_model_prompt},
]
caption_model_convo_string = caption_processor.apply_chat_template(caption_model_convo, tokenize=False, add_generation_prompt=True)

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
    try:
        inputs = od_processor(text="<OD>", images=image, return_tensors="pt").to("cpu", torch.bfloat16)
        generated_ids = od_model.generate(
            **inputs,
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )
        generated_text = od_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = od_processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
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

def generate_labels(photo, image, coll_photos):
    photo_id = photo["numId"]
    print(f"{photo_id}: generating labels")
    inputs = caption_processor(text=[caption_model_convo_string], images=[image], return_tensors="pt").to("cpu")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    generate_ids = caption_model.generate(**inputs, max_new_tokens=300, do_sample=True, suppress_tokens=None, use_cache=True, temperature=0.6, top_k=None, top_p=0.9)[0]
    caption = caption_processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    labels = caption.splitlines()[-1].split(", ")
    print("setting labels for photo %d to %s" % (photo["numId"], json.dumps(labels)))
    filter = { "_id": photo["_id"] }
    update_op = { "$set": {"labels": labels } }
    coll_photos.update_one(filter, update_op)

def extract_texts(photo, image, coll_photos):
    photo_id = photo["numId"]
    print(f"{photo_id}: extracting texts")
    bounding_box = train_bounding_box(image)
    texts = []
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
        texts = [t for t in texts if t.strip()] # remove empty strings
    else:
        print("train not found!")
    print("setting texts for photo %d to %s" % (photo["numId"], json.dumps(texts)))
    filter = { "_id": photo["_id"] }
    update_op = { "$set": {"texts": texts } }
    coll_photos.update_one(filter, update_op)
 


try:
    client = mongo_connect()

    db_bahnbilder = client["bahnbilder"]
    db_bahnbilder_files_original = client["bahnbilder-files-original"]

    coll_photos = db_bahnbilder["photos"]
    coll_files = db_bahnbilder_files_original["files"]
    blacklist_numIds = []

    while True:
        coll_photos_watch = coll_photos.watch() # start before query to avoid race conditions
        photo = coll_photos.find_one({"texts": None, "numId": { "$nin": blacklist_numIds } })
        if photo == None:
            photo = coll_photos.find_one({"labels": None, "numId": { "$nin": blacklist_numIds } })
        if photo == None:
            print("waiting for changes")
            next(coll_photos_watch)
            continue

        jpeg = coll_files.find_one({"photoId": photo["numId"]})
        if jpeg == None:
            print("jpeg %d is missing!" % photo["numId"])
            time.sleep(0.1)
            continue

        try:
            image = Image.open(io.BytesIO(jpeg["data"]))
            if photo.get("texts", None) == None:
                extract_texts(photo, image, coll_photos)
            if photo.get("labels", None) == None:
                generate_labels(photo, image, coll_photos)

        except Exception as e:
            print(f"Error processing photo: {e}")
            blacklist_numIds.append(photo["numId"])

    client.close()

except Exception as e:
    raise Exception("The following error occurred: ", e)


