from pathlib import Path
import urllib.request


model_path_to_url = {
    Path("models/vgg/ndcc_stanford_dogs_30.pt"):
    "https://tufts.box.com/shared/static/qyx3hrczo937hpjigolrrbjs0zo7pk0f.pt",

    Path("models/vgg/ndcc_stanford_dogs_times_1e-1_30.pt"):
    "https://tufts.box.com/shared/static/sfc0ser0ys3phutmr9png0gvreq436kp.pt",

    Path("models/vgg/ndcc_plus_30.pt"):
    "https://tufts.box.com/shared/static/zijwq9qjgizpw5fdd3ceo4fmokj3o5xu.pt",

    Path("models/vgg/vgg_classifier_1000.pt"):
    "https://tufts.box.com/shared/static/lzbx7ph26fl70otyjyab7z9ka7bpickq.pt",

    Path("models/vgg/vgg_classifier_1000_2.pt"):
    "https://tufts.box.com/shared/static/8fn8vpbr30as6holo20gdutyhsjvhapp.pt",

    Path("models/vgg/vgg_classifier_1000_3.pt"):
    "https://tufts.box.com/shared/static/52s070o8hve4wxv9penk58yjcbe5pjfo.pt",

    Path("models/vgg/vgg_classifier_1000_4.pt"):
    "https://tufts.box.com/shared/static/o5x4iqeivdv1j4dnr9muxrlbuhj4ia9c.pt",

    Path("models/vgg/vgg_classifier_1000_5.pt"):
    "https://tufts.box.com/shared/static/vbbvdgknw2ozebali0giiylss1ax9urd.pt",

    Path("models/polycraft/noisy/scale_1/patch_based/8000.pt"):
    "https://tufts.box.com/shared/static/xjjl6ogwdxeyp6972x3x94c2haoh7twj.pt",

    Path("models/polycraft/noisy/scale_1/fullimage_based/8000.pt"):
    "https://tufts.box.com/shared/static/m1q942jp2r4r29xgx5rn19caha980ubo.pt",
    
    Path("models/polycraft/noisy/scale_1/patch_based/8000_plus.pt"):
    "https://tufts.box.com/shared/static/3nl567njynbbafq3mqp8l1dj80mv32xy.pt",
}

if __name__ == "__main__":
    for model_path, url in model_path_to_url.items():
        if not model_path.exists():
            print("Downloading " + str(model_path))
            urllib.request.urlretrieve(url, model_path)
        else:
            print("Found " + str(model_path))
