# Harris Corner Detector

This repository contains an implementation of the Harris Corner Detection
algorithm — a classic computer vision technique to identify corner features,
a common type of interest point in images. The system uses a **RealSense
camera** as the input sensor.

The processing pipeline includes:
 - Applying a **Sobel filter** to compute image gradients
 - Calculating the **Harris corner response (R)** for each pixel
 - Performing **non-maximum suppression** to localize distinct corner points


https://github.com/user-attachments/assets/e44093a4-f6bd-41a9-b783-df40fbe02da3
