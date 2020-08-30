from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()
prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath( execution_path + r"\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()

def predict(img):
    try:
        predictions, percentage_probabilities = prediction.predictImage(execution_path+r"\{}".format(img), result_count=5)
    except :
        predictions, percentage_probabilities = prediction.predictImage("{}".format(img), result_count=5)
    for index in range(len(predictions)):
        print(predictions[index] , " : " , percentage_probabilities[index])
# predict("sample.jfif")
