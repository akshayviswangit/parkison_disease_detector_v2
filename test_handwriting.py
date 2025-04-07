from transformers import pipeline

pipe = pipeline("image-classification", "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")

print(pipe(r"data\image_data\pd-hw-parkinson\pd-hw-parkinson\V03PE04.png"))