import cv2
import matplotlib.pyplot as plt
from rockedgesdetectors import ModelPiDiNet
from pathlib import Path

checkpoint_path_7 = Path("models/pidinetmodels/table7_pidinet.pth")
checkpoint_path_5 = Path("models/pidinetmodels/table5_pidinet.pth")
image_path = Path("data/test_image/test.png")

model = ModelPiDiNet(checkpoint_path_7)
image = cv2.imread(str(image_path))
result_1 = model(image)


model = ModelPiDiNet(checkpoint_path_5)
image = cv2.imread(str(image_path))
result_2 = model(image)


fig = plt.figure(figsize=(7, 9))
axs = [fig.add_subplot(2, 2, 1),
       fig.add_subplot(2, 2, 3),
       fig.add_subplot(2, 2, 4)]
axs[0].imshow(image)
axs[1].imshow(result_1)
axs[2].imshow(result_2)
plt.show()
