# =====================================================================================================================
# Aufgabe S&P 500 Classifier with PyTorch Lightning
# 20.11.2021, Thomas Iten
# =====================================================================================================================

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

from datamodule import SP500DataModule
from model import SP500ClassifierModel

# ---------------------------------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------------------------------

epochs = 30
batch_size = 48
learning_rate = 0.0001

train_val_test_split=[80,10,10]
gpus = 1 if torch.cuda.is_available() else 0
progress_bar_refresh_rate=16

print("# ------------------------------------------------------------------------------------------------------------")
print("# Train and test")
print("# ------------------------------------------------------------------------------------------------------------")

datamodule = SP500DataModule(batch_size=batch_size, train_val_test_split=train_val_test_split)
model = SP500ClassifierModel(learning_rate=learning_rate)

trainer = pl.Trainer(gpus=gpus, max_epochs=epochs, progress_bar_refresh_rate=progress_bar_refresh_rate)

trainer.fit(model, datamodule)
trainer.test(datamodule=datamodule)

print("# ------------------------------------------------------------------------------------------------------------")
print("# Run learning finder variation 1 : auto_lr_find=True")
print("# ------------------------------------------------------------------------------------------------------------")

datamodule = SP500DataModule(batch_size=batch_size, train_val_test_split=train_val_test_split)
model = SP500ClassifierModel(learning_rate=learning_rate)

trainer = pl.Trainer(gpus=gpus, max_epochs=epochs, progress_bar_refresh_rate=progress_bar_refresh_rate, auto_lr_find=True)

trainer.tune(model, datamodule)

print("# ------------------------------------------------------------------------------------------------------------")
print("# Run learning finder variation 2 : trainer.tuner.lr_find")
print("# ------------------------------------------------------------------------------------------------------------")

datamodule = SP500DataModule(batch_size=batch_size, train_val_test_split=train_val_test_split)
model = SP500ClassifierModel(learning_rate=learning_rate)

trainer = pl.Trainer(min_epochs=5)
lr_finder = trainer.tuner.lr_find(model, datamodule)

# Results can be found in
print(lr_finder.results)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
print(new_lr)

# update hparams of the model
model.hparams.lr = new_lr

print("# ------------------------------------------------------------------------------------------------------------")
print("# Train with early stop")
print("# ------------------------------------------------------------------------------------------------------------")

early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt

if not os.path.exists('models'):
    os.makedirs('models')

checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    dirpath="models",
    filename="sp500-{epoch:02d}-{val_accuracy:.2f}",
    save_top_k = 5,
    mode="max"
)

trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], progress_bar_refresh_rate=progress_bar_refresh_rate, max_epochs=epochs)

trainer.fit(model, datamodule)
trainer.test(datamodule=datamodule)

# =====================================================================================================================
# The end