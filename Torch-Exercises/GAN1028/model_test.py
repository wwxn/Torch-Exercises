import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import definition as df

IMG_SIZE = 28 * 28
VECTOR_SIZE = 3
BATCH_SIZE = 1

test_model = df.Generator(input_size=VECTOR_SIZE, output_size=IMG_SIZE)
test_model.load_state_dict(torch.load("model_G.pt"))

test_model.eval()

while True:
    G_input = torch.randn((1, VECTOR_SIZE))
    print(G_input)
    test_out = test_model.forward(G_input)

    unloader = transforms.ToPILImage()
    test_out = test_out.data.reshape(1, 28, 28)
    # test_out=test_out.ge(-0.5).float()
    # print(test_out)
    image = unloader(test_out)
    plt.imshow(image,'gray')
    plt.show()
