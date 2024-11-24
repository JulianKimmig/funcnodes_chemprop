import unittest
from funcnodes_chemprop import functions as fn
import pandas as pd
import os
from lightning import pytorch as pl
from chemprop import models


class TestFunctions(unittest.TestCase):
    def test_train(self):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "mol.csv"))
        smiles_column = "smiles"
        target_columns = "lipo"
        train_loader, val_loader, test_loader, scaler = fn.make_data(
            df, smiles_column, target_columns
        )
        model = fn.make_model(scaler)

        self.assertIsInstance(
            model,
            models.MPNN,
            f"Model is not a MPNN but {type(model)}",
        )

        self.assertIsInstance(
            model,
            pl.LightningModule,
            f"Model is not a LightningModule but {type(model)}",
        )
        _ = fn.predict(model, df[smiles_column])
