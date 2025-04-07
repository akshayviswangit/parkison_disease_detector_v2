from transformers import pipeline

pipe = pipeline("image-classification", "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")

print(pipe("data\pd-hw-healthy\pd-hw-healthy\V01PE01.png"))