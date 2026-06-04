import json

import numpy as np
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from clifford.database import Registry
from clifford.search import Search
from clifford.store import Store


class ModelBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = Registry()
        self.search_engine = Search()
        self.store = Store()
        self.selected_model_name = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Model Registry")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Saved networks, architecture metadata, and quick inference checks.")
        subtitle.setObjectName("MutedLabel")
        
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models...")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_models)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_models)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.refresh_button)
        
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(5)
        self.model_table.setHorizontalHeaderLabels(["Name", "Registry", "Artifact", "Updated", "Path"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.model_table.clicked.connect(self.on_model_selected)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(180)

        infer_group = QGroupBox("Quick Inference")
        infer_layout = QVBoxLayout(infer_group)
        infer_input_layout = QHBoxLayout()
        self.inference_input = QLineEdit()
        self.inference_input.setPlaceholderText("Comma-separated input, e.g. 0,1")
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False)
        infer_input_layout.addWidget(self.inference_input)
        infer_input_layout.addWidget(self.predict_button)
        self.prediction_output = QTextEdit()
        self.prediction_output.setReadOnly(True)
        self.prediction_output.setMaximumHeight(90)
        infer_layout.addLayout(infer_input_layout)
        infer_layout.addWidget(self.prediction_output)
        
        self.delete_button = QPushButton("Delete Model")
        self.delete_button.clicked.connect(self.delete)
        self.delete_button.setEnabled(False)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(search_layout)
        layout.addWidget(self.model_table)
        details_label = QLabel("Model Details")
        details_label.setObjectName("SectionTitle")
        layout.addWidget(details_label)
        layout.addWidget(self.details_text)
        layout.addWidget(infer_group)
        layout.addWidget(self.delete_button)
        
        self.setLayout(layout)
        self.refresh_models()

    def refresh_models(self):
        self.populate_table(self.available_models())

    def search_models(self):
        query = self.search_input.text().strip().lower()
        models = self.available_models()
        if query:
            models = [model for model in models if query in model["name"].lower()]
        self.populate_table(models)

    def available_models(self):
        registered = {model.name: model for model in self.registry.list_models()}
        artifacts = self.store.list_paths()
        names = sorted(set(registered) | set(artifacts))

        models = []
        for name in names:
            registry_model = registered.get(name)
            artifact_path = artifacts.get(name)
            models.append(
                {
                    "name": name,
                    "registry": "Registered" if registry_model else "File only",
                    "artifact": "Available" if artifact_path else "Missing file",
                    "updated": registry_model.updated_at[:19] if registry_model else "",
                    "path": str(artifact_path) if artifact_path else "",
                    "metadata": registry_model,
                }
            )
        return models

    def populate_table(self, models):
        self.model_table.setRowCount(len(models))
        for row, model in enumerate(models):
            self.model_table.setItem(row, 0, QTableWidgetItem(model["name"]))
            self.model_table.setItem(row, 1, QTableWidgetItem(model["registry"]))
            self.model_table.setItem(row, 2, QTableWidgetItem(model["artifact"]))
            self.model_table.setItem(row, 3, QTableWidgetItem(model["updated"]))
            self.model_table.setItem(row, 4, QTableWidgetItem(model["path"]))

    def on_model_selected(self, index):
        self.selected_model_name = self.model_table.item(index.row(), 0).text()
        model = self.registry.get_model(self.selected_model_name)
        artifact = None
        if self.store.exists(self.selected_model_name):
            try:
                artifact = self.store.get_metadata(self.selected_model_name)
            except Exception as e:
                artifact = {"error": str(e)}

        details = [f"Name: {self.selected_model_name}"]
        if model:
            details.extend(
                [
                    "Registry: registered",
                    f"Architecture: {model.architecture}",
                    f"Hyperparameters: {model.hyperparameters}",
                    f"Created: {model.created_at}",
                    f"Updated: {model.updated_at}",
                ]
            )
        else:
            details.append("Registry: file artifact only")

        if artifact:
            details.append(f"Artifact: {artifact.get('path', 'unknown')}")
            if artifact.get("saved_at"):
                details.append(f"Saved: {artifact['saved_at']}")
            if artifact.get("architecture"):
                details.append("Artifact architecture:")
                details.append(json.dumps(artifact["architecture"], indent=2))
            if artifact.get("error"):
                details.append(f"Artifact error: {artifact['error']}")

        self.details_text.setText("\n".join(details))
        self.delete_button.setEnabled(True)
        self.predict_button.setEnabled(self.store.exists(self.selected_model_name))

    def run_prediction(self):
        if not self.selected_model_name:
            return

        try:
            values = [
                float(value.strip())
                for value in self.inference_input.text().split(",")
                if value.strip()
            ]
            if not values:
                self.prediction_output.setText("Enter at least one input value.")
                return

            network = self.store.load(self.selected_model_name)
            prediction = network.predict(np.array([values], dtype=np.float64))
            self.prediction_output.setText(np.array2string(prediction, precision=6))
        except Exception as e:
            self.prediction_output.setText(f"Prediction failed: {e}")

    def delete(self):
        if self.selected_model_name:
            self.registry.delete_model(self.selected_model_name)
            self.store.delete(self.selected_model_name)
            self.refresh_models()
            self.details_text.clear()
            self.prediction_output.clear()
            self.delete_button.setEnabled(False)
            self.predict_button.setEnabled(False)
            self.selected_model_name = None
