imgSize=224

data_folder = "models/"

sub_dir="colab/"
data_type="aug/"
model1=data_folder+sub_dir+data_type+"MobileNetV2_aug.tflite"
model2=data_folder+sub_dir+data_type+"MobileNetV2_aug_edgetpu.tflite"
# data_type="noaug/"
# model1=data_folder+sub_dir+data_type+"MobileNetV2.tflite"
# model2=data_folder+sub_dir+data_type+"MobileNetV2_edgetpu.tflite"
# 
# sub_dir="tm/"
# data_type="aug/"
# model1=data_folder+sub_dir+data_type+"model.tflite"
# model2=data_folder+sub_dir+data_type+"model_edgetpu.tflite"
# data_type="noaug/"
# model1=data_folder+sub_dir+data_type+"model.tflite"
# model2=data_folder+sub_dir+data_type+"model_edgetpu.tflite"

list_model=[model1, model2]
critical_sound = 'templates/critical.mp3'
good_sound = 'templates/good.mp3'
list_sound = [critical_sound, good_sound]

