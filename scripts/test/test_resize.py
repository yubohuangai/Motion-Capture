import os
import cv2
import torch
import numpy as np

# -----------------------------
# Your functions
# -----------------------------
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.uint8)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h,
           (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns: preprocessed tensor, original image, original dimensions
    """
    if isinstance(img, str):
        orig_im = cv2.imread(img)
        if orig_im is None:
            raise FileNotFoundError(f"Image not found: {img}")
    else:
        orig_im = img

    dim = orig_im.shape[1], orig_im.shape[0]  # width, height
    img_resized = letterbox_image(orig_im, (inp_dim, inp_dim))
    img_ = img_resized[:, :, ::-1].transpose((2, 0, 1)).copy()  # BGR -> RGB, HWC -> CHW
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

# -----------------------------
# Test script
# -----------------------------
if __name__ == "__main__":
    original_image = "/Users/yubo/data/s2/seq1/images/01/000000.jpg"
    output_dir = "/output"
    os.makedirs(output_dir, exist_ok=True)

    # Set target input dimension
    inp_dim = 416  # You can try 640 for higher resolution

    # Preprocess
    img_tensor, orig_img, orig_dim = prep_image(original_image, inp_dim)

    print(f"Original image size: {orig_dim}")
    print(f"Preprocessed tensor shape: {img_tensor.shape}")

    # Convert tensor back to image for visualization
    img_out = img_tensor.squeeze(0).numpy().transpose(1, 2, 0) * 255
    img_out = img_out[:, :, ::-1].astype(np.uint8)  # RGB -> BGR

    output_path = os.path.join(output_dir, "preprocessed_000000.jpg")
    cv2.imwrite(output_path, img_out)
    print(f"Preprocessed image saved to: {output_path}")