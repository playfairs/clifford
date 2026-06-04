from datetime import datetime

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from clifford.core.exceptions import DatabaseError
from clifford.data import generate_circle_dataset, generate_spiral_dataset, generate_xor_dataset
from clifford.model import (
    Adam,
    BinaryCrossEntropy,
    Dense,
    Linear,
    MeanSquaredError,
    Network,
    ReLU,
    SGD,
    Sigmoid,
    Tanh,
)
from clifford.train import Trainer
from clifford.database import Registry
from clifford.store import Store


class TrainingThread(QThread):
    progress = Signal(int, int, float)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, network, trainer, X, y):
        super().__init__()
        self.network = network
        self.trainer = trainer
        self.X = X
        self.y = y

    def run(self):
        try:
            loss = 0.0
            for epoch in range(self.trainer.config.epochs):
                loss = self.trainer._train_epoch(self.X, self.y)
                self.trainer.history["loss"].append(loss)
                self.progress.emit(epoch + 1, self.trainer.config.epochs, loss)

            self.finished.emit({"final_loss": loss, "history": self.trainer.history})
        except Exception as e:
            self.error.emit(str(e))


class TrainingDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = Registry()
        self.store = Store()
        self.training_thread = None
        self.active_run_id = None
        self.active_model_name = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Trainer")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Create, continue, train, and save NumPy neural networks.")
        subtitle.setObjectName("MutedLabel")

        model_group = QGroupBox("Model")
        model_grid = QGridLayout(model_group)

        self.model_source_combo = QComboBox()
        self.model_source_combo.addItems(["Create new model", "Continue saved model"])
        self.model_source_combo.currentTextChanged.connect(self.on_model_source_changed)

        self.saved_model_combo = QComboBox()
        self.saved_model_combo.setEnabled(False)

        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("model_name")

        self.architecture_input = QLineEdit()
        self.architecture_input.setPlaceholderText("64:relu,32:relu,1:sigmoid")
        self.architecture_input.setText("8:relu,4:relu,1:sigmoid")

        model_grid.addWidget(QLabel("Source"), 0, 0)
        model_grid.addWidget(self.model_source_combo, 0, 1)
        model_grid.addWidget(QLabel("Saved Model"), 0, 2)
        model_grid.addWidget(self.saved_model_combo, 0, 3)
        model_grid.addWidget(QLabel("Name"), 1, 0)
        model_grid.addWidget(self.model_name_input, 1, 1)
        model_grid.addWidget(QLabel("Architecture"), 1, 2)
        model_grid.addWidget(self.architecture_input, 1, 3)

        trainer_group = QGroupBox("Training Setup")
        trainer_grid = QGridLayout(trainer_group)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["XOR", "Circle", "Spiral"])

        self.samples_input = QSpinBox()
        self.samples_input.setRange(64, 100000)
        self.samples_input.setValue(1000)

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(100)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 10000)
        self.batch_size_input.setValue(32)

        self.lr_input = QLineEdit()
        self.lr_input.setText("0.01")

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam"])

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["Mean Squared Error", "Binary Cross Entropy"])

        trainer_grid.addWidget(QLabel("Dataset"), 0, 0)
        trainer_grid.addWidget(self.dataset_combo, 0, 1)
        trainer_grid.addWidget(QLabel("Samples"), 0, 2)
        trainer_grid.addWidget(self.samples_input, 0, 3)
        trainer_grid.addWidget(QLabel("Epochs"), 1, 0)
        trainer_grid.addWidget(self.epochs_input, 1, 1)
        trainer_grid.addWidget(QLabel("Batch Size"), 1, 2)
        trainer_grid.addWidget(self.batch_size_input, 1, 3)
        trainer_grid.addWidget(QLabel("Learning Rate"), 2, 0)
        trainer_grid.addWidget(self.lr_input, 2, 1)
        trainer_grid.addWidget(QLabel("Optimizer"), 2, 2)
        trainer_grid.addWidget(self.optimizer_combo, 2, 3)
        trainer_grid.addWidget(QLabel("Loss"), 3, 0)
        trainer_grid.addWidget(self.loss_combo, 3, 1)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        self.refresh_button = QPushButton("Refresh Models")
        self.refresh_button.clicked.connect(self.refresh_saved_models)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.refresh_button)

        status_group = QGroupBox("Run Monitor")
        status_layout = QVBoxLayout(status_group)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("StatusLabel")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        status_layout.addWidget(self.progress_label)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.log_text)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(model_group)
        layout.addWidget(trainer_group)
        layout.addLayout(button_layout)
        layout.addWidget(status_group, 1)

        self.setLayout(layout)
        self.refresh_saved_models()

    def refresh_saved_models(self):
        current = self.saved_model_combo.currentText()
        self.saved_model_combo.clear()
        self.saved_model_combo.addItems(self.store.list())
        if current:
            index = self.saved_model_combo.findText(current)
            if index >= 0:
                self.saved_model_combo.setCurrentIndex(index)

    def on_model_source_changed(self, value):
        use_saved = value == "Continue saved model"
        self.saved_model_combo.setEnabled(use_saved)
        self.architecture_input.setEnabled(not use_saved)
        self.model_name_input.setPlaceholderText("Optional new save name" if use_saved else "model_name")

    def start_training(self):
        try:
            learning_rate = float(self.lr_input.text())
        except ValueError:
            self.log_text.append("Error: Learning rate must be numeric")
            return

        try:
            dataset = self.create_dataset()
            network, model_name = self.prepare_network(learning_rate)
            trainer = Trainer(network)
            trainer.config.epochs = self.epochs_input.value()
            trainer.config.batch_size = self.batch_size_input.value()
            trainer.config.verbose = False

            model_id = self.ensure_registered_model(network, model_name)
            run_config = {
                "dataset": dataset.name,
                "samples": dataset.samples,
                "epochs": trainer.config.epochs,
                "batch_size": trainer.config.batch_size,
                "learning_rate": learning_rate,
                "optimizer": self.optimizer_combo.currentText(),
                "loss": self.loss_combo.currentText(),
            }
            self.active_run_id = self.registry.register_training_run(model_id, run_config)
            self.active_model_name = model_name

            self.training_thread = TrainingThread(network, trainer, dataset.X, dataset.y)
            self.training_thread.progress.connect(self.update_progress)
            self.training_thread.finished.connect(self.training_finished)
            self.training_thread.error.connect(self.training_error)

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setMaximum(trainer.config.epochs)
            self.progress_bar.setValue(0)
            self.training_thread.start()

            self.log_text.append(f"Started run {self.active_run_id[:8]} for {model_name}")
            self.log_text.append(f"Dataset: {dataset.name} ({dataset.samples} samples)")
            self.log_text.append(network.summary())
        except Exception as e:
            self.log_text.append(f"Error: {e}")

    def prepare_network(self, learning_rate):
        if self.model_source_combo.currentText() == "Continue saved model":
            saved_name = self.saved_model_combo.currentText()
            if not saved_name:
                raise ValueError("Choose a saved model to continue")
            network = self.store.load(saved_name)
            model_name = self.model_name_input.text().strip() or saved_name
        else:
            model_name = self.model_name_input.text().strip()
            if not model_name:
                raise ValueError("Model name is required")
            network = self.create_network(model_name)

        network.compile(loss=self.create_loss(), optimizer=self.create_optimizer(learning_rate))
        network.name = model_name
        return network, model_name

    def create_network(self, model_name):
        architecture = self.architecture_input.text().strip()
        if not architecture:
            raise ValueError("Architecture is required")

        network = Network(name=model_name)
        for layer_spec in architecture.split(","):
            parts = [part.strip() for part in layer_spec.split(":")]
            units = int(parts[0])
            activation = parts[1].lower() if len(parts) > 1 else "relu"
            network.add(Dense(units=units, activation=self.create_activation(activation)))
        return network

    def create_activation(self, name):
        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh,
            "linear": Linear,
        }
        return activation_map.get(name, ReLU)()

    def create_optimizer(self, learning_rate):
        if self.optimizer_combo.currentText() == "Adam":
            return Adam(learning_rate=learning_rate)
        return SGD(learning_rate=learning_rate)

    def create_loss(self):
        if self.loss_combo.currentText() == "Binary Cross Entropy":
            return BinaryCrossEntropy()
        return MeanSquaredError()

    def create_dataset(self):
        samples = self.samples_input.value()
        dataset_name = self.dataset_combo.currentText()
        if dataset_name == "Circle":
            dataset = generate_circle_dataset(samples=samples)
        elif dataset_name == "Spiral":
            dataset = generate_spiral_dataset(samples=samples)
        else:
            dataset = generate_xor_dataset(samples=samples)
        dataset.normalize()
        return dataset

    def ensure_registered_model(self, network, model_name):
        architecture = network.get_architecture()
        hyperparameters = {
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_size_input.value(),
            "learning_rate": self.lr_input.text(),
            "optimizer": self.optimizer_combo.currentText(),
            "loss": self.loss_combo.currentText(),
        }

        model_id = self.registry.get_model_id(model_name)
        if model_id:
            self.registry.update_model(model_name, architecture, hyperparameters)
            return model_id

        try:
            return self.registry.register_model(model_name, architecture, hyperparameters)
        except DatabaseError:
            self.registry.update_model(model_name, architecture, hyperparameters)
            return self.registry.get_model_id(model_name) or model_name

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            if self.active_run_id:
                self.registry.update_training_run(
                    self.active_run_id,
                    end_time=datetime.now().isoformat(),
                    status="stopped",
                )
            self.log_text.append("Training stopped")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def update_progress(self, epoch, total_epochs, loss):
        self.progress_bar.setMaximum(total_epochs)
        self.progress_bar.setValue(epoch)
        self.progress_label.setText(f"Epoch {epoch}/{total_epochs} | loss {loss:.6f}")
        self.log_text.append(f"Epoch {epoch}/{total_epochs} | loss {loss:.6f}")
        if self.active_run_id:
            self.registry.log_metric(self.active_run_id, epoch, "loss", loss)

    def training_finished(self, result):
        final_loss = result["final_loss"]
        self.log_text.append(f"Training completed. Final loss: {final_loss:.6f}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        model_name = self.active_model_name or self.model_name_input.text().strip()
        self.store.save(
            self.training_thread.network,
            model_name,
            metadata={
                "run_id": self.active_run_id,
                "final_loss": final_loss,
                "history": result["history"],
            },
        )
        self.ensure_registered_model(self.training_thread.network, model_name)

        if self.active_run_id:
            self.registry.update_training_run(
                self.active_run_id,
                end_time=datetime.now().isoformat(),
                status="completed",
                final_loss=final_loss,
                final_metrics={"loss": final_loss},
            )

        self.refresh_saved_models()
        self.log_text.append(f"Model saved: {model_name}")
        self.active_run_id = None
        self.active_model_name = None

    def training_error(self, error_msg):
        self.log_text.append(f"Error: {error_msg}")
        if self.active_run_id:
            self.registry.update_training_run(
                self.active_run_id,
                end_time=datetime.now().isoformat(),
                status="failed",
            )
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
