from model.model_fed import CNN, LeNet
from model.model_res import ResNet18, ResNet34, ResNet50
# from model.model_res_official import ResNet50
import torchvision.models as models
from model.efficientnet import EfficientNet
import torch.nn as nn


def build_model(args):
    # choose different Neural network model for different args or input
    if args.model == 'cnn':
        netglob = CNN(args=args).to(args.device)
    if args.model == 'lenet':
        netglob = LeNet().to(args.device)
    elif args.model == 'resnet18':
        netglob = ResNet18(args.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'resnet34':
        netglob = ResNet34(args.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'resnet50':
        netglob = ResNet50(args.num_classes)
        netglob = netglob.to(args.device)
        # netglob = ResNet50(pretrained=False)
        # if args.pretrained:
        #     model = models.resnet50(pretrained=True)
        #     netglob.load_state_dict(model.state_dict())
        # netglob.fc = nn.Linear(2048, args.num_classes)
        # netglob = netglob.to(args.device)
    elif args.model == 'resnext':
        netglob = models.resnext50_32x4d()
        netglob.fc = nn.Linear(2048, args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'wideresnet':
        netglob = models.wide_resnet50_2()
        netglob.fc = nn.Linear(2048, args.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'vgg11':
        netglob = models.vgg11()
        netglob.fc = nn.Linear(4096, args.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'vgg16':
        netglob = models.vgg16()
        netglob.fc = nn.Linear(4096, args.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'densenet':
        netglob = models.densenet121()
        netglob.fc = nn.Linear(64, args.num_classes)
        netglob = netglob.to(args.device)
    elif args.model == 'efficientnet':
        netglob = EfficientNet.from_name('efficientnet-b0')
        netglob.fc = nn.Linear(1280, args.num_classes)
        netglob = netglob.to(args.device)

    # elif args.model == 'alexnet':
    #     netglob = models.alexnet()
    #     netglob.fc = nn.Linear(4096, args.num_classes)
    #     netglob = netglob.to(args.device)
    # elif args.model == 'inception':
    #     netglob = models.inception_v3()
    #     netglob.fc = nn.Linear(2048, args.num_classes)
    #     netglob = netglob.to(args.device)
    # elif args.model == 'googlenet':
    #     netglob = models.googlenet()
    #     netglob.fc = nn.Linear(1024, args.num_classes)
    #     netglob = netglob.to(args.device)
    else:
        exit('Error: unrecognized model')

    return netglob
