Steps to run the project:
1. Upload all the files and folders in this repository to google colab '/content/' folder or in current working directory (both are same).
2. Run 'colab_file.ipynb' in Google Colab.
3. For downloading dataset: Download kaggle.json file by following instruction from https://www.kaggle.com/general/74235 and make sure its present in '/content/drive/MyDrive/Kaggle' before downloading the dataset.


Notes:
1. For changing the number of files in training per class, modify parameter 'train_per_cls'.
2. For updating train-val split, modify parameter 'valSplit'.
3. 'results' folder contains the confusion matrix on Test results.
4. 'logs' folder will keep the logs for every run.
5. 'checkpoint' folder will save the checkpoint after each 'save_ckpt_after' (parameter) during training.
6. All parameters are present in main.py