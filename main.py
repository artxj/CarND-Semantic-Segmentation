import argparse
from distutils.version import LooseVersion
import os.path
import tensorflow as tf
import warnings

import helper
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    layer_conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='layer_conv1x1')

    layer_up1 = tf.layers.conv2d_transpose(layer_conv1x1, num_classes, 4, strides=(2, 2), padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='layer_up1')
    layer_up1 = tf.add(layer_up1, vgg_layer4_out)

    layer_up2 = tf.layers.conv2d_transpose(layer_up1, num_classes, 4, strides=(2, 2), padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='layer_up2')
    layer_up2 = tf.add(layer_up2, vgg_layer3_out)

    layer_up3 = tf.layers.conv2d_transpose(layer_up2, num_classes, 16, strides=(8, 8), padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='layer_up3')
    return layer_up3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def save_model(sess, saver, save_path):
    """
    Save trained model
    :param sess: TF Session
    :param saver: TF Saver
    :param save_path: Path to save the model
    """
    saver.save(sess, save_path)

def restore_model(sess, saver, save_path):
    """
    Save trained model
    :param sess: TF Session
    :param saver: TF Saver
    :param save_path: Path to save the model
    """
    saver.restore(sess, save_path)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver, save_path):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param saver: TF Saver
    :param save_path: Path to save trained model
    """
    # TODO: Implement function
    for epoch_num in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={keep_prob: 0.8, learning_rate: 1e-3,
                input_image: image, correct_label: label})
            print('Epoch {} loss = {}'.format(epoch_num, loss))
    save_model(sess, saver, save_path)

tests.test_train_nn(train_nn)

def parse_args(save_path, epochs, batch_size):
    """
    Parses console arguments.
    :param save_path: Path a model to be saved (or loaded from)
    :param epochs: Number of epochs
    :param batch_size: Batch size
    """
    parser = argparse.ArgumentParser(description='Semantic segmentation fully convolutional neural network.')
    parser.add_argument(
        '--model_path',
        type=str,
        default=save_path,
        help='Path to a model to be loaded (optionally) and to be saved.'
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load the pre-trained model.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=epochs,
        help='Number of epochs to be run.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=batch_size,
        help='Batch size.'
    )
    return parser.parse_args()

def run():
    num_classes = 2
    image_shape = (160, 576)
    epochs = 10
    batch_size = 32
    save_path = './model/model.ckpt'
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    saver = tf.train.Saver()

    args = parse_args(save_path, epochs, batch_size)
    epochs = args.epochs
    batch_size = args.batch_size
    save_path = args.save_path
    load_model = args.load

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # tensorflow placeholders
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_label = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 2), name='correct_label')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        if load_model:
            restore_model(sess, saver, save_path)
        else:
            train_nn(sess, epochs, batch_size, get_batches_fn,
                train_op, cross_entropy_loss, image_input,
                correct_label, keep_prob, learning_rate,
                saver, save_path)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
