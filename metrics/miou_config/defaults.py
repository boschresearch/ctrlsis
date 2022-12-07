from yacs.config import CfgNode as CN

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"

_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.num_class = 150

_C.DATASET.imgSizes = (300, 375, 450, 525, 600)

_C.DATASET.imgMaxSize = 1000

_C.DATASET.padding_constant = 8

_C.DATASET.segm_downsampling_rate = 8

_C.DATASET.random_flip = True

_C.MODEL = CN()

_C.MODEL.arch_encoder = "resnet50dilated"

_C.MODEL.arch_decoder = "ppm_deepsup"

_C.MODEL.weights_encoder = ""

_C.MODEL.weights_decoder = ""

_C.MODEL.fc_dim = 2048

_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2

_C.TRAIN.num_epoch = 20

_C.TRAIN.start_epoch = 0

_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02

_C.TRAIN.lr_pow = 0.9

_C.TRAIN.beta1 = 0.9

_C.TRAIN.weight_decay = 1e-4

_C.TRAIN.deep_sup_scale = 0.4

_C.TRAIN.fix_bn = False

_C.TRAIN.workers = 16

_C.TRAIN.disp_iter = 20

_C.TRAIN.seed = 304

_C.VAL = CN()

_C.VAL.batch_size = 1

_C.VAL.visualize = False

_C.VAL.checkpoint = "epoch_20.pth"

_C.TEST = CN()

_C.TEST.batch_size = 1

_C.TEST.checkpoint = "epoch_20.pth"

_C.TEST.result = "./"
