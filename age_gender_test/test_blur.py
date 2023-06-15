import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

img0=Image.open('test/113005.jpg.jpg')
img1=transforms.GaussianBlur((5, 5), sigma=(1.0, 1.0))(img0)
img2=transforms.RandomAffine(0, (0.1, 0.1))(img1)
img3 = transforms.RandomRotation(degrees=(15, 15))(img2)
axs = plt.figure().subplots(1, 4)
axs[0].imshow(img0);axs[0].set_title('src');axs[0].axis('off')
axs[1].imshow(img1);axs[1].set_title('GaussianBlur');axs[1].axis('off')
axs[2].imshow(img2);axs[2].set_title('RandomAffine');axs[2].axis('off')
axs[3].imshow(img3);axs[3].set_title('RandomRotation');axs[3].axis('off')
# plt.show()
plt.savefig("res_blur.jpg")

