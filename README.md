# Image Synthesis using Generative Adverserial Networks

GANs have revolutionized the field of computer vision by enabling the generation of realistic and high-quality images. In this project, we explore the fascinating world of GANs and delve into their application for image synthesis. By training a generator network to create images that are indistinguishable from real ones, we aim to unlock the potential for various creative and practical purposes, such as generating artwork, enhancing image processing techniques, and even simulating realistic scenarios.
<p  align="center">
<img  width="350"  src="https://github.com/m22cs058/image-synthesis/blob/main/plots/faces/generated_images_epoch_49.png?raw=true"  alt="Material Bread logo">
<img  width="350"  src="https://github.com/m22cs058/image-synthesis/blob/main/plots/mnist/generated_images_epoch_33_step_63150.png?raw=true"  alt="Material Bread logo">

</p>

## Dataset
We have employed three diverse datasets: MNIST, Fashion MNIST, and a small subset of CelebA. The MNIST dataset consists of handwritten digits, Fashion MNIST comprises various fashion items, and CelebA contains a collection of celebrity faces. By incorporating these datasets, we aim to showcase the versatility and applicability of our GAN model across different domains, ranging from digit recognition and fashion generation to realistic face synthesis. This multi-dataset approach allows us to explore the capabilities of our model in generating diverse and visually appealing images, demonstrating the broad potential of GANs in various image synthesis tasks.
MNIST and Fashion-MNIST can be found on torchvision datasets. To download CelebA small dataset:

    !wget https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip
    !unzip -q processed-celeba-small.zip

## Setup
Clone the repository

    git clone git@github.com:m22cs058/image-synthesis.git

Create a conda environment

    conda create -n image-synthesis python=3.9.13
    conda activate image-synthesis
 
Install Necessary Libraries

    pip install -r requirements.txt
