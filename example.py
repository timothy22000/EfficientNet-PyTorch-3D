from efficientnet_pytorch_3d import EfficientNet3D, MultiModalEfficientNet3D
import torch
from torchsummary import summary


print("Test")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

model = MultiModalEfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)

# summary(model, input_size=(1, 224, 224, 224))
summary(model, [(1, 224, 224, 224),(1, 224, 224, 224),(1, 224, 224, 224),(1, 224, 224, 224)])

model = model.to(device)
inputs1 = torch.randn((1, 1, 224, 224, 224)).to(device)
inputs2 = torch.randn((1, 1, 224, 224, 224)).to(device)
inputs3 = torch.randn((1, 1, 224, 224, 224)).to(device)
inputs4 = torch.randn((1, 1, 224, 224, 224)).to(device)

labels = torch.tensor([0]).to(device)
# test forward
num_classes = 2

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for epoch in range(2):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model.forward(inputs1, inputs2, inputs3, inputs4)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

print('Finished Training')
