from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox, QComboBox, QPushButton, QTextEdit, QProgressBar
from PySide6.QtCore import Qt, QThread, Signal

from clifford.model import Network, Dense, ReLU, Sigmoid, MeanSquaredError, SGD
from clifford.train import Trainer
from clifford.data import Dataset
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
            for epoch in range(self.trainer.config.epochs):
                loss = self.trainer._train_epoch(self.X, self.y)
                self.progress.emit(epoch + 1, self.trainer.config.epochs, loss)
            
            result = {"final_loss": loss}
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TrainingDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = Registry()
        self.store = Store()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        config_layout = QHBoxLayout()
        
        model_layout = QVBoxLayout()
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Model Name")
        model_layout.addWidget(QLabel("Model Name:"))
        model_layout.addWidget(self.model_name_input)
        
        arch_layout = QVBoxLayout()
        self.architecture_input = QLineEdit()
        self.architecture_input.setPlaceholderText("Architecture (e.g., 64:relu,32:sigmoid,1:sigmoid)")
        arch_layout.addWidget(QLabel("Architecture:"))
        arch_layout.addWidget(self.architecture_input)
        
        epochs_layout = QVBoxLayout()
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(100)
        epochs_layout.addWidget(QLabel("Epochs:"))
        epochs_layout.addWidget(self.epochs_input)
        
        batch_size_layout = QVBoxLayout()
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 1000)
        self.batch_size_input.setValue(32)
        batch_size_layout.addWidget(QLabel("Batch Size:"))
        batch_size_layout.addWidget(self.batch_size_input)
        
        lr_layout = QVBoxLayout()
        self.lr_input = QLineEdit()
        self.lr_input.setText("0.01")
        lr_layout.addWidget(QLabel("Learning Rate:"))
        lr_layout.addWidget(self.lr_input)
        
        config_layout.addLayout(model_layout)
        config_layout.addLayout(arch_layout)
        config_layout.addLayout(epochs_layout)
        config_layout.addLayout(batch_size_layout)
        config_layout.addLayout(lr_layout)
        
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready to train")
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        layout.addLayout(config_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
        self.training_thread = None

    def start_training(self):
        model_name = self.model_name_input.text()
        architecture = self.architecture_input.text()
        epochs = self.epochs_input.value()
        batch_size = self.batch_size_input.value()
        learning_rate = float(self.lr_input.text())
        
        if not model_name or not architecture:
            self.log_text.append("Error: Model name and architecture are required")
            return
        
        try:
            network = Network(name=model_name)
            
            for layer_spec in architecture.split(","):
                parts = layer_spec.split(":")
                units = int(parts[0])
                activation = parts[1] if len(parts) > 1 else "relu"
                
                if activation == "relu":
                    act = ReLU()
                elif activation == "sigmoid":
                    act = Sigmoid()
                else:
                    act = ReLU()
                
                network.add(Dense(units=units, activation=act))
            
            network.compile(loss=MeanSquaredError(), optimizer=SGD(learning_rate=learning_rate))
            
            from clifford import generate_xor_dataset
            dataset = generate_xor_dataset(samples=1000)
            dataset.normalize()
            
            trainer = Trainer(network)
            trainer.config.epochs = epochs
            trainer.config.batch_size = batch_size
            trainer.config.verbose = False
            
            self.training_thread = TrainingThread(network, trainer, dataset.X, dataset.y)
            self.training_thread.progress.connect(self.update_progress)
            self.training_thread.finished.connect(self.training_finished)
            self.training_thread.error.connect(self.training_error)
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.training_thread.start()
            
            self.log_text.append(f"Started training: {model_name}")
            self.log_text.append(f"Architecture: {architecture}")
            self.log_text.append(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
            
        except Exception as e:
            self.log_text.append(f"Error: {str(e)}")

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.log_text.append("Training stopped")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def update_progress(self, epoch, total_epochs, loss):
        self.progress_bar.setMaximum(total_epochs)
        self.progress_bar.setValue(epoch)
        self.progress_label.setText(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}")
        self.log_text.append(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.6f}")

    def training_finished(self, result):
        self.log_text.append(f"Training completed. Final loss: {result['final_loss']:.6f}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        model_name = self.model_name_input.text()
        self.store.save(self.training_thread.network, model_name)
        self.registry.register_model(
            name=model_name,
            architecture=self.training_thread.network.get_architecture(),
            hyperparameters={"epochs": self.epochs_input.value(), "batch_size": self.batch_size_input.value()}
        )
        self.log_text.append(f"Model saved: {model_name}")

    def training_error(self, error_msg):
        self.log_text.append(f"Error: {error_msg}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
