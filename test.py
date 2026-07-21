import joblib
from PIL import Image
import matplotlib.pyplot as plt
from preprocessing.main import pre_process_func, FFT

def pre_process(path):
    test_img = Image.open(path).convert('L')
    test_img = pre_process_func(test_img)

    test_img = test_img.mean(dim=0, keepdim=True)
    test_img = FFT(test_img).reshape(1, -1)

    return test_img

model = joblib.load('ai_detector.joblib')
test_img = pre_process('test images/test.webp')
real_test_img = pre_process('test images/real_test.webp')

ai_prediction = model.predict(test_img)
real_prediction = model.predict(real_test_img)
mappings = {1: 'AI', 0:'real'}

fig, ax = plt.subplots(1, 2)
ax[0].imshow(Image.open('test images/test.webp'))
ax[0].text(0.05, 0.95, f'The model predicted that this image is {mappings[ai_prediction[0]]}',
           transform=ax[0].transAxes, fontsize=8, verticalalignment='top', c='white')

ax[1].imshow(Image.open('test images/real_test.webp'))
ax[1].text(0.05, 0.95, f'The model predicted that this image is {mappings[real_prediction[0]]}',
           transform=ax[1].transAxes, fontsize=8, verticalalignment='top', c='white')

plt.show()