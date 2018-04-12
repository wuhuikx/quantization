import caffe
import numpy as np

int8_model = "./models/resnet50_int8.prototxt"
fp32_model = "./models/resnet50_fp32.prototxt"
weights = "./models/resnet50.caffemodel"

caffe.set_mode_cpu()

int8_net = caffe.Net(int8_model, weights, caffe.TEST)
fp32_net = caffe.Net(fp32_model, weights, caffe.TEST)
#print('blobs {}\n params {}'.format(int8_net.blobs.keys(), int8_net.params.keys()))
#print('blobs {}\n params {}'.format(fp32_net.blobs.keys(), fp32_net.params.keys()))

#squeeze: remove dims = 1
#convert [h, w, c] to [n, c, h, w]
image = np.array(caffe.io.load_image("./images/cat224.jpg")).squeeze()
image = image.transpose(2, 0, 1)
image = image[np.newaxis, :, :, :]
print(image.shape)

int8_net.blobs['data'].data[...] = image
int8_net.forward()
int8_fc1000 = int8_net.blobs["fc1000"]
int8_prob = int8_net.blobs["prob"]
np.save("./results/resnet50_int8_fc1000.npy", int8_fc1000.data)
np.save("./results/resnet50_int8_prob.npy", int8_prob.data)


fp32_net.blobs['data'].data[...] = image
fp32_net.forward()
fp32_fc1000 = fp32_net.blobs['fc1000']
fp32_prob = fp32_net.blobs["prob"]
np.save("./results/resnet50_fp32_fc1000.npy", fp32_fc1000.data)
np.save("./results/resnet50_fp32_prob.npy", fp32_prob.data)
