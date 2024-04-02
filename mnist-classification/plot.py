import json
import os
import matplotlib.pyplot as plt


def get_json(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data


def get_data(model):
    data = get_json('output/' + model + '.json')
    return data


def plot_inter_model_comparison(option):
    if option == 'vanilla':
        data_softm = get_data('softmax_dropout_0_bnorm_False')
        data_mlp = get_data('mlp_h1_512_h2_256_dropout_0_bnorm_False')
        data_cnn = get_data('cnn_h1_512_h2_256_dropout_0_bnorm_False_oc1_32_oc2_64')
    elif option == 'dropout':
        data_softm = get_data('softmax_dropout_0.2_bnorm_False')
        data_mlp = get_data('mlp_h1_512_h2_256_dropout_0.2_bnorm_False')
        data_cnn = get_data('cnn_h1_512_h2_256_dropout_0.2_bnorm_False_oc1_32_oc2_64')
    else:
        data_softm = get_data('softmax_dropout_0_bnorm_True')
        data_mlp = get_data('mlp_h1_512_h2_256_dropout_0_bnorm_True')
        data_cnn = get_data('cnn_h1_512_h2_256_dropout_0_bnorm_True_oc1_32_oc2_64')

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(data_softm['loss_train'], color='b', label='Softmax - Train')
    ax1.plot(data_softm['loss_test'], color='c', linestyle='dotted', label='Softmax - Test')
    ax1.plot(data_mlp['loss_train'], color='r', label='MLP - Train')
    ax1.plot(data_mlp['loss_test'], color='m', linestyle='dotted', label='MLP - Test')
    ax1.plot(data_cnn['loss_train'], color='g', label='CNN - Train')
    ax1.plot(data_cnn['loss_test'], color='y', linestyle='dotted', label='CNN - Test')
    ax1.set_title('a) Loss for {} settings'.format(option))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(data_softm['accuracy_train'], color='b', label='Softmax - Train')
    ax2.plot(data_softm['accuracy_test'], color='c', linestyle='dotted', label='Softmax - Test')
    ax2.plot(data_mlp['accuracy_train'], color='r', label='MLP - Train')
    ax2.plot(data_mlp['accuracy_test'], color='m', linestyle='dotted', label='MLP - Test')
    ax2.plot(data_cnn['accuracy_train'], color='g', label='CNN - Train')
    ax2.plot(data_cnn['accuracy_test'], color='y', linestyle='dotted', label='CNN - Test')
    ax2.set_title('b) Accuracy for {} settings'.format(option))
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.savefig('./plots/{}.png'.format(option))
    fig.show()


def plot_intra_model_comparison(option):
    if option == 'softmax':
        data_van = get_data('softmax_dropout_0_bnorm_False')
        data_drop = get_data('softmax_dropout_0.2_bnorm_False')
        data_bn = get_data('softmax_dropout_0_bnorm_True')
    elif option == 'MLP':
        data_van = get_data('mlp_h1_512_h2_256_dropout_0_bnorm_False')
        data_drop = get_data('mlp_h1_512_h2_256_dropout_0.2_bnorm_False')
        data_bn = get_data('mlp_h1_512_h2_256_dropout_0_bnorm_True')
    else:
        data_van = get_data('cnn_h1_512_h2_256_dropout_0_bnorm_False_oc1_32_oc2_64')
        data_drop = get_data('cnn_h1_512_h2_256_dropout_0.2_bnorm_False_oc1_32_oc2_64')
        data_bn = get_data('cnn_h1_512_h2_256_dropout_0_bnorm_True_oc1_32_oc2_64')

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(data_van['loss_train'], color='b', label='Vanilla - Train')
    ax1.plot(data_van['loss_test'], color='c', linestyle='dotted', label='Vanilla - Test')
    ax1.plot(data_drop['loss_train'], color='r', label='Dropout - Train')
    ax1.plot(data_drop['loss_test'], color='m', linestyle='dotted', label='Dropout - Test')
    ax1.plot(data_bn['loss_train'], color='g', label='BatchNorm - Train')
    ax1.plot(data_bn['loss_test'], color='y', linestyle='dotted', label='BatchNorm - Test')
    ax1.set_title('a) Loss for {} model'.format(option))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(data_van['accuracy_train'], color='b', label='Vanilla - Train')
    ax2.plot(data_van['accuracy_test'], color='c', linestyle='dotted', label='Vanilla - Test')
    ax2.plot(data_drop['accuracy_train'], color='r', label='Dropout - Train')
    ax2.plot(data_drop['accuracy_test'], color='m', linestyle='dotted', label='Dropout - Test')
    ax2.plot(data_bn['accuracy_train'], color='g', label='BatchNorm - Train')
    ax2.plot(data_bn['accuracy_test'], color='y', linestyle='dotted', label='BatchNorm - Test')
    ax2.set_title('b) Accuracy for {} model'.format(option))
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.savefig('./plots/{}.png'.format(option))
    fig.show()


def plot_mlp_comparison():
    data_64 = get_data('mlp_h1_64_h2_32_dropout_0_bnorm_False')
    data_512 = get_data('mlp_h1_512_h2_256_dropout_0_bnorm_False')
    data_1024 = get_data('mlp_h1_1024_h2_512_dropout_0_bnorm_False')

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(data_64['loss_train'], color='b', label='Train - h1 = 64, h2 = 32')
    ax1.plot(data_64['loss_test'], color='c', linestyle='dotted', label='Test - h1 = 64, h2 = 32')
    ax1.plot(data_512['loss_train'], color='r', label='Train - h1 = 512, h2 = 256')
    ax1.plot(data_512['loss_test'], color='m', linestyle='dotted', label='Test - h1 = 512, h2 = 256')
    ax1.plot(data_1024['loss_train'], color='g', label='Train - h1 = 1024, h2 = 512')
    ax1.plot(data_1024['loss_test'], color='y', linestyle='dotted', label='Test - h1 = 1024, h2 = 512')
    ax1.set_title('a) Loss for MLP model')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(data_64['accuracy_train'], color='b', label='Train - h1 = 64, h2 = 32')
    ax2.plot(data_64['accuracy_test'], color='c', linestyle='dotted', label='Test - h1 = 64, h2 = 32')
    ax2.plot(data_512['accuracy_train'], color='r', label='Train - h1 = 512, h2 = 256')
    ax2.plot(data_512['accuracy_test'], color='m', linestyle='dotted', label='Test - h1 = 512, h2 = 256')
    ax2.plot(data_1024['accuracy_train'], color='g', label='Train - h1 = 1024, h2 = 512')
    ax2.plot(data_1024['accuracy_test'], color='y', linestyle='dotted', label='Test - h1 = 1024, h2 = 512')
    ax2.set_title('b) Accuracy for MLP model')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    plt.legend()

    plt.savefig('./plots/mlp_comparison.png')
    fig.show()


def plot_cnn_comparison():
    data_64_32 = get_data('cnn_h1_64_h2_32_dropout_0_bnorm_False_oc1_32_oc2_64')
    data_512_4 = get_data('cnn_h1_512_h2_256_dropout_0_bnorm_False_oc1_4_oc2_8')
    data_512_32 = get_data('cnn_h1_512_h2_256_dropout_0_bnorm_False_oc1_32_oc2_64')

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(data_64_32['loss_train'], color='b', label='Train - h1 = 64, h2 = 32, $C^1_{out}=32$, $C^2_{out}=64$')
    ax1.plot(data_64_32['loss_test'], color='c', linestyle='dotted',
             label='Test - h1 = 64, h2 = 32, $C^1_{out}=32$, $C^2_{out}=64$')
    ax1.plot(data_512_4['loss_train'], color='r', label='Train - h1 = 512, h2 = 256, $C^1_{out}=4$, $C^2_{out}=8$')
    ax1.plot(data_512_4['loss_test'], color='m', linestyle='dotted',
             label='Test - h1 = 512, h2 = 256, $C^1_{out}=4$, $C^2_{out}=8$')
    ax1.plot(data_512_32['loss_train'], color='g', label='Train - h1 = 512, h2 = 256, $C^1_{out}=32$, $C^2_{out}=64$')
    ax1.plot(data_512_32['loss_test'], color='y', linestyle='dotted',
             label='Test - h1 = 512, h2 = 256, $C^1_{out}=32$, $C^2_{out}=64$')
    ax1.set_title('a) Loss for CNN model')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(data_64_32['accuracy_train'], color='b', label='Train - h1 = 64, h2 = 32, $C^1_{out}=32$, $C^2_{out}=64$')
    ax2.plot(data_64_32['accuracy_test'], color='c', linestyle='dotted',
             label='Test - h1 = 64, h2 = 32, $C^1_{out}=32$, $C^2_{out}=64$')
    ax2.plot(data_512_4['accuracy_train'], color='r', label='Train - h1 = 512, h2 = 256, $C^1_{out}=4$, $C^2_{out}=8$')
    ax2.plot(data_512_4['accuracy_test'], color='m', linestyle='dotted',
             label='Test - h1 = 512, h2 = 256, $C^1_{out}=4$, $C^2_{out}=8$')
    ax2.plot(data_512_32['accuracy_train'], color='g',
             label='Train - h1 = 512, h2 = 256, $C^1_{out}=32$, $C^2_{out}=64$')
    ax2.plot(data_512_32['accuracy_test'], color='y', linestyle='dotted',
             label='Test - h1 = 512, h2 = 256, $C^1_{out}=32$, $C^2_{out}=64$')
    ax2.set_title('b) Accuracy for MLP model')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.savefig('./plots/cnn_comparison.png')
    fig.show()


if __name__ == '__main__':
    os.makedirs('./plots', exist_ok=True)

    plot_inter_model_comparison('vanilla')
    plot_inter_model_comparison('dropout')
    plot_inter_model_comparison('batch normalization')

    plot_intra_model_comparison('softmax')
    plot_intra_model_comparison('MLP')
    plot_intra_model_comparison('CNN')

    plot_mlp_comparison()
    plot_cnn_comparison()
