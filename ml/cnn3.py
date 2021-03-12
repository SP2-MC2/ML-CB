"""
__version__ = '1.0'
__author__ = 'Nathan Reitinger'

- used to test options for learning canvas fingerprinting images
- see Section 3.5.1 for information on why it may not be the best idea to rely
  on an image-based model alone when classifying canvas fingerprinting images
"""

#  ██████ ███    ██ ███    ██
# ██      ████   ██ ████   ██
# ██      ██ ██  ██ ██ ██  ██
# ██      ██  ██ ██ ██  ██ ██
#  ██████ ██   ████ ██   ████

# from fastai.vision import *
from fastai.vision import ImageList, imagenet_stats, cnn_learner, models
from fastai.vision import get_transforms, ResizeMethod, accuracy, torch
from fastai.vision import ClassificationInterpretation, doc
from fastai.metrics import error_rate
from fastai.imports import Path
import fastai

# from fastai import *
import warnings
import os

# change if GPU available (tested only on CPU)
fastai.torch_core.defaults.device = torch.device("cpu")
# defaults.device = torch.device('cpu')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# https://github.com/dmlc/xgboost/issues/1715
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# training is very, very slow
os.environ["OMP_NUM_THREADS"] = "1"

## helpful way to initially get folders
# import split_folders
# split_folders.ratio('<path>', output='<path>/split', seed=1337, ratio=(.8, .2)) # uses default values
# sys.exit()

path = Path("data/CNN/-released/split")

################################################################################
# fastai uses databunches
################################################################################
data = (
    ImageList.from_folder(path / "train")
    .split_by_rand_pct(0.1, seed=33)
    .label_from_folder()
    # .add_test_folder('..'/path/'test')
    .transform(
        get_transforms(do_flip=True, flip_vert=True),
        size=150,
        resize_method=ResizeMethod.SQUISH,
        padding_mode="zeros",
    )
    .databunch(bs=64)
    .normalize(imagenet_stats)
)

# ## turn this on for regular testing
# option_name = 'CNN__original'
# data_test = (ImageList.from_folder(path)
#                 .split_by_folder(train='train', valid='test')
#                 .label_from_folder()
#                 .transform(get_transforms(do_flip=True,flip_vert=True),size=150,resize_method=ResizeMethod.SQUISH,padding_mode='zeros')
#                 .databunch(bs=64)
#                 .normalize(imagenet_stats))

## turn this on for test_suite
option_name = "CNN__testSuite"
data_test = (
    ImageList.from_folder(path)
    .split_by_folder(train="train", valid="test_suite")
    .label_from_folder()
    .transform(
        get_transforms(do_flip=True, flip_vert=True),
        size=150,
        resize_method=ResizeMethod.SQUISH,
        padding_mode="zeros",
    )
    .databunch(bs=64)
    .normalize(imagenet_stats)
)

## stats on data
# print(data.classes)
# print(data)
# print(data_test)
# print(data_test.c)
# print(len(data.train_ds.y.items))
# print(len(data.valid_dl.y.items))
# print(len(data_test.train_ds.y.items))
# print(len(data_test.valid_dl.y.items))

################################################################################
# model parameters
################################################################################
learn = cnn_learner(
    data,  # training on low res first
    models.resnet50,  # resnet50 arch with pretrained weights
    metrics=[accuracy, error_rate],
    model_dir="/tmp/model/",
)  # specifying a different directory
# as /input is a read-only directory
# and will throw an error while
# using lr_find()

## variables to hold metrics
accuracy_all = []
list = []
run_f1 = []
run_accuracy = []
run_precision = []
run_recall = []

# how many times to run the CNN (helps with generalization for the paper)
number_of_runs = 10

for i in range(number_of_runs):

    ############################################################################
    # fit the model
    ############################################################################

    print("\n\n================== training run", i, "==================\n\n")

    ## helps with learning, unfreezing weights may be unnecessary
    learn.unfreeze()  # unfreezing the inital layers
    learn.lr_find()  # finding best learning rate
    # learn.recorder.plot()
    learn.fit_one_cycle(3, slice(1e-4, 1e-3))

    ## stat on loss function used
    # print(learn.loss_func)

    ############################################################################
    # results
    ############################################################################
    learn.data = data_test  # loading the test data
    interp = ClassificationInterpretation.from_learner(learn)
    losses, idxs = interp.top_losses()
    interp.plot_top_losses(15, figsize=(30, 30))
    # plt.show()
    doc(interp.plot_top_losses)
    # plt.savefig(str(i) + 'FINAL_top-losses.pdf', dpi=1200)
    interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
    # plt.savefig(str(i) + 'FINAL_confusion.pdf', dpi=1200)
    interp.most_confused(min_val=2)
    # plt.show()

    temp = learn.validate(data_test.valid_dl)
    print("\nrun results:")
    print(temp)
    print("\naccuracy:", temp[1])
    accuracy_all.append(temp[1])
    confusion = interp.confusion_matrix()

    # === revised metrics === #
    true_positive = confusion[1][1]
    true_negative = confusion[0][0]
    false_positive = confusion[0][1]
    false_negative = confusion[1][0]

    if true_positive == 0 and (true_positive + false_positive) == 0:
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (
        true_positive + false_positive + false_negative + true_negative
    )
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    print(
        "\nmore detail:",
        "accuracy",
        accuracy,
        "F1",
        f1,
        "precision",
        precision,
        "recall",
        recall,
    )

    all = {}
    all["f1_" + option_name] = f1
    all["accuracy_" + option_name] = accuracy
    all["precision_" + option_name] = precision
    all["recall_" + option_name] = recall
    list.append(all)

print("\noverall:")
print(accuracy_all)

################################################################################
# save results locally
################################################################################
print("\n==== saving file ====\n")

df = pd.DataFrame(list)
print(df.head())

name = "final_outputs/" + option_name + ".csv"
if not os.path.isfile(name):
    df.to_csv(name, header=True, index=False)
else:
    df.to_csv(name, mode="a", header=False, index=False)
df = pd.read_csv(name)
print("\n\nsaved==>", name)
print(df.head())
