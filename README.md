Steps to run the project:
1. Run 'colab_file.ipynb' in Google Colab.

Notes:
1. For changing the number of files in training per class, modify parameter 'train_per_cls'.
2. For updating train-val split, modify parameter 'valSplit'.
3. 'results' folder contains the confusion matrix on Test results.
4. 'logs' folder will keep the logs for every run.
5. 'checkpoint' folder will save the checkpoint after each 'save_ckpt_after' (parameter) during training.
6. All parameters are present in main.py