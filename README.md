# Weakly Supervised GAN Model

This repository contains an implementation of a Weakly Supervised Generative Adversarial Network (GAN) model for image generation. This model utilizes weak supervision techniques noise regularization to train a GAN with minimal labeled data, making it suitable for scenarios where obtaining large amounts of labeled data is challenging or expensive.

## Overview

Computer vision tasks such as object recognition and image classification require annotated data which is costly and difficult to obtain. Further, in applications like medical imaging, it is costly to get a large image dataset. Generative Adversarial Networks (GANs) have shown remarkable success in generating realistic images. With Weakly supervised learning, we aim to alleviate this dependency on fully labeled datasets by developing an architecture which generates images and utilizes noisy labels to regularize the classfication model. 

This project implements a weakly supervised GAN classification model that can label fashion item with high accuracy using with minimal labeled data. .

## Key Features

- **Weakly Supervised Training:** Utilizes weak supervision signals for training, reducing the reliance on fully labeled datasets.
  ![Weakly Supervised GAN architecture](https://github.com/rajaspandey/weakly-supervised-gan/blob/main/docs/DLProjarchitecture.drawio.png)
- **Accurate classification with limited labeled data:** Achieves accuracy of 89% with only 50% clean data.
  ![Prediction on FashionMNIST dataset](https://github.com/rajaspandey/weakly-supervised-gan/blob/main/docs/NR_WSGAN_Top3.png)
- **High-Quality Image Generation:** Produces realistic and high-quality images that satisfy the desired constraints.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/rajaspandey/weakly-supervised-gan.git
    ```

2. (Optional) Set up your environment (e.g., virtual environment) as needed.


## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.


## Acknowledgments

- Jordan Axelrod [jordanIAxelrod](https://github.com/jordanIAxelrod)
- Rathi Kashi [rathikashi](https://github.com/rathikashi)
## Contact

For any inquiries or questions, please contact [Rajas Pandey](mailto:rajaspandey9@gmail.com).

---

Feel free to customize this README according to your project's specific details and requirements. Happy coding!
