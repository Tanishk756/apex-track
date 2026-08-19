"""
Unit Tests for Dataset Management, Synthetic Dataset Synthesis, and Fine-Tuning Engine
=======================================================================================
"""

import os
from pathlib import Path
import pytest
from tools.dataset_manager import DatasetManager, PRESET_DATASETS
from tools.train_detector import TacticalModelTrainer
from apex.engine.training.training_pipeline import AutonomousFineTuner


class TestDatasetManager:
    """Test suite for DatasetManager dataset listing and synthetic rendering."""

    def test_list_datasets(self):
        mgr = DatasetManager.instance()
        catalog = mgr.list_datasets()
        assert len(catalog) >= 4
        names = [d["name"] for d in catalog]
        assert "drone_uav" in names
        assert "battlefield_vehicles" in names
        assert "thermal_ir" in names

    def test_synthesize_dataset(self, tmp_path):
        mgr = DatasetManager(datasets_dir=str(tmp_path))
        res = mgr.synthesize_dataset(dataset_name="drone_uav", num_samples=10)
        assert res["status"] == "SUCCESS"
        assert res["samples_generated"] == 10
        assert Path(res["data_yaml"]).exists()

        train_imgs = list((tmp_path / "drone_uav" / "images" / "train").glob("*.jpg"))
        assert len(train_imgs) == 8
        val_imgs = list((tmp_path / "drone_uav" / "images" / "val").glob("*.jpg"))
        assert len(val_imgs) == 2

        lbl_file = tmp_path / "drone_uav" / "labels" / "train" / (train_imgs[0].stem + ".txt")
        assert lbl_file.exists()
        content = lbl_file.read_text().strip()
        assert len(content) > 0

    def test_aod4_synthesis(self, tmp_path):
        mgr = DatasetManager(datasets_dir=str(tmp_path))
        res = mgr.synthesize_dataset(dataset_name="aod_4", num_samples=12)
        assert res["status"] == "SUCCESS"
        assert res["dataset_name"] == "aod_4"
        assert "drone" in res["classes"]
        assert Path(res["data_yaml"]).exists()


class TestModelTrainer:
    """Test suite for TacticalModelTrainer."""

    def test_trainer_execution(self, tmp_path):
        ds_mgr = DatasetManager(datasets_dir=str(tmp_path / "ds"))
        ds_res = ds_mgr.synthesize_dataset(dataset_name="thermal_ir", num_samples=8)

        trainer = TacticalModelTrainer(output_dir=str(tmp_path / "models"))
        res = trainer.train_model(
            dataset_yaml=ds_res["data_yaml"],
            backbone="yolov8n.pt",
            epochs=1,
            batch_size=2,
            output_name="test_apex.pt",
        )

        assert res["status"] == "SUCCESS"
        assert res["epochs_completed"] == 1
        assert Path(res["weights_saved"]).exists()


class TestAutonomousFineTuner:
    """Test suite for continuous online active learning harvester."""

    def test_fine_tuner_trigger(self):
        tuner = AutonomousFineTuner.instance()
        res = tuner.trigger_rigorous_fine_tune()
        assert res["status"] == "COMPLETED"
        assert tuner.get_status()["auto_training_status"] == "ACTIVE_MONITORING"
