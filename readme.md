# Lien Kaggle notebook: https://www.kaggle.com/code/becharrihajar/notebookd173e0311c

# Laboratoire de Deep Learning - Rapport


# Objective:
The main goal of this lab was to gain familiarity with the PyTorch library and build various neural architectures for computer vision tasks, including CNNs, RCNNs, FCNNs, and Vision Transformers (ViT).

# Part 1: CNN Classifier
1. Established a CNN Architecture: We defined a Convolutional Neural Network architecture using the PyTorch library to classify the MNIST dataset. This involved specifying layers such as convolutional layers, pooling layers, and fully connected layers, as well as hyperparameters such as kernel size, padding, stride, and optimizers.
2. Implemented Faster R-CNN: We also implemented the Faster R-CNN architecture for object detection on the MNIST dataset.
3. Comparison of Models: We compared the performance of the CNN and Faster R-CNN models using various metrics such as accuracy, F1 score, loss, and training time.
4. Fine-Tuning with VGG16 and AlexNet: We fine-tuned pre-trained VGG16 and AlexNet models on the MNIST dataset and compared their performance with the CNN and Faster R-CNN models.

# Part 2: Vision Transformer (ViT)
1. Established ViT Model Architecture: Following a tutorial, we built a Vision Transformer (ViT) model from scratch and performed a classification task on the MNIST dataset using PyTorch.
2. Interpretation and Comparison: We interpreted the results obtained from the ViT model and compared them with the results obtained in Part 1.

Apologies for the oversight. Let's include the comparison of the models based on the results you provided:

# Comparison of Models:
1. CNN vs. Faster R-CNN:
   - **Accuracy:** CNN achieved a significantly higher accuracy (98%) compared to Faster R-CNN (9%) on the MNIST dataset.
   - **F1 Score:** CNN had a perfect F1 score (1.0), while Faster R-CNN had a much lower F1 score (0.04).
   - **Loss:** CNN had a lower loss value compared to Faster R-CNN.
   - **Training Time:** The training time for CNN was likely shorter than Faster R-CNN due to its simpler architecture.

2. CNN vs. ViT:
   - **Accuracy:** CNN achieved a higher accuracy (98%) compared to ViT (65%) on the MNIST dataset.
   - **F1 Score:** CNN had a perfect F1 score (1.0), while ViT had a lower F1 score.
   - **Loss:** CNN had a lower loss value compared to ViT.
   - **Training Time:** The training time for CNN might have been shorter than ViT due to its simpler architecture.

3. Faster R-CNN vs. ViT:
   - **Accuracy:** Faster R-CNN achieved a higher accuracy (9%) compared to ViT (65%) on the MNIST dataset.
   - **F1 Score:** Faster R-CNN had a slightly higher F1 score compared to ViT.
   - **Loss:** ViT might have had a lower loss value compared to Faster R-CNN.
   - **Training Time:** The training time for ViT was likely longer than Faster R-CNN due to its more complex architecture.

Based on these comparisons, it appears that CNNs performed better than Faster R-CNN and ViT on the MNIST dataset. However, the comparison also depends on the specific metrics and tasks used for evaluation. Further experimentation and analysis may be necessary to draw more conclusive findings.



# Summary:
Throughout the lab, we learned how to build various neural architectures for computer vision tasks using PyTorch, including CNNs, RCNNs, FCNNs, and Vision Transformers. We also gained experience in fine-tuning pre-trained models and comparing their performance. Additionally, we explored the capabilities of Vision Transformers in image classification tasks and compared them with traditional CNN-based approaches.

Overall, this lab provided valuable hands-on experience with deep learning techniques for computer vision tasks and helped us gain a deeper understanding of the strengths and limitations of different neural architectures.