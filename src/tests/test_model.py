import torch
import timm


def test_model():

    train_data = torch.randn(1, 3, 100, 100)
    model = timm.create_model("resnet18", pretrained=True, num_classes=101)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    output = model(train_data)

    assert output.shape == (
        1,
        101,
    ), "The output of the model does not have the right output dimension"
