import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    Automatically finds the last convolutional layer if name is not provided.
    """
    # 1. Automatic Layer Detection
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    
    # 2. Find the layer object
    last_conv_layer = model.get_layer(last_conv_layer_name)
    if last_conv_layer is None:
        return None

    # 3. Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # 4. Gradient Tape to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # We want the gradient of the winning class with respect to the feature map
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 5. Compute Gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 6. Global Average Pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 7. Multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image.
    """
    heatmap = np.uint8(255 * heatmap)

    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.array(img)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    superimposed_img = jet * alpha + img
    return np.clip(superimposed_img, 0, 255).astype('uint8')