from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QTextEdit, QTabWidget, QSplitter, QSpinBox, QDoubleSpinBox
from PySide6.QtCore import Qt
from datetime import datetime
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLM"))
from avon import Avon, AvonConfig, AvonGenerator, AvonTrainer, Tokenizer


class AvonView(QWidget):
    def __init__(self):
        super().__init__()
        
        try:
            self.avon_config = AvonConfig(vocab_size=10000, d_model=256, n_heads=4, n_layers=4, d_ff=1024)
            self.avon = Avon(self.avon_config)
            self.tokenizer = Tokenizer(vocab_size=10000)
            self.avon_generator = AvonGenerator(self.avon, self.tokenizer)
            self.avon_trainer = AvonTrainer(self.avon, self.avon_config)
            self.avon_available = True
        except Exception as e:
            self.avon_available = False
            print(f"Avon initialization failed: {e}")
        
        self.chat_history: list = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Avon LLM")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Transformer-based language model for text generation.")
        subtitle.setObjectName("MutedLabel")

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        
        tabs.addTab(self._create_chat_tab(), "Chat")
        tabs.addTab(self._create_config_tab(), "Config")
        tabs.addTab(self._create_training_tab(), "Training")
        tabs.addTab(self._create_info_tab(), "Info")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(tabs)
        self.setLayout(layout)

    def _create_chat_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        splitter = QSplitter(Qt.Vertical)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #c0c0c0;
                border: 1px solid #222222;
                border-radius: 2px;
                padding: 12px;
            }
        """)
        splitter.addWidget(self.chat_display)
        
        input_widget = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background-color: #0a0a0a;
                color: #c0c0c0;
                border: 1px solid #333333;
                border-radius: 2px;
                padding: 8px 12px;
            }
            QLineEdit:focus {
                border: 1px solid #555555;
            }
        """)
        self.chat_input.returnPressed.connect(self._send_chat_message)
        input_layout.addWidget(self.chat_input)
        
        send_button = QPushButton("Send")
        send_button.clicked.connect(self._send_chat_message)
        input_layout.addWidget(send_button)
        
        input_widget.setLayout(input_layout)
        splitter.addWidget(input_widget)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        layout.addWidget(splitter)
        widget.setLayout(layout)
        return widget

    def _create_config_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        config_layout = QVBoxLayout()
        
        vocab_size_layout = QHBoxLayout()
        vocab_size_layout.addWidget(QLabel("Vocab Size:"))
        self.vocab_size_spin = QSpinBox()
        self.vocab_size_spin.setRange(1000, 100000)
        self.vocab_size_spin.setValue(10000)
        vocab_size_layout.addWidget(self.vocab_size_spin)
        config_layout.addLayout(vocab_size_layout)
        
        d_model_layout = QHBoxLayout()
        d_model_layout.addWidget(QLabel("Model Dimension:"))
        self.d_model_spin = QSpinBox()
        self.d_model_spin.setRange(64, 2048)
        self.d_model_spin.setValue(256)
        d_model_layout.addWidget(self.d_model_spin)
        config_layout.addLayout(d_model_layout)
        
        n_heads_layout = QHBoxLayout()
        n_heads_layout.addWidget(QLabel("Number of Heads:"))
        self.n_heads_spin = QSpinBox()
        self.n_heads_spin.setRange(1, 32)
        self.n_heads_spin.setValue(4)
        n_heads_layout.addWidget(self.n_heads_spin)
        config_layout.addLayout(n_heads_layout)
        
        n_layers_layout = QHBoxLayout()
        n_layers_layout.addWidget(QLabel("Number of Layers:"))
        self.n_layers_spin = QSpinBox()
        self.n_layers_spin.setRange(1, 24)
        self.n_layers_spin.setValue(4)
        n_layers_layout.addWidget(self.n_layers_spin)
        config_layout.addLayout(n_layers_layout)
        
        d_ff_layout = QHBoxLayout()
        d_ff_layout.addWidget(QLabel("Feed-Forward Dimension:"))
        self.d_ff_spin = QSpinBox()
        self.d_ff_spin.setRange(128, 8192)
        self.d_ff_spin.setValue(1024)
        d_ff_layout.addWidget(self.d_ff_spin)
        config_layout.addLayout(d_ff_layout)
        
        dropout_layout = QHBoxLayout()
        dropout_layout.addWidget(QLabel("Dropout:"))
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setSingleStep(0.1)
        self.dropout_spin.setValue(0.1)
        dropout_layout.addWidget(self.dropout_spin)
        config_layout.addLayout(dropout_layout)
        
        apply_button = QPushButton("Apply Configuration")
        apply_button.clicked.connect(self._apply_config)
        config_layout.addWidget(apply_button)
        
        layout.addLayout(config_layout)
        widget.setLayout(layout)
        return widget

    def _create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel("Training controls for Avon model")
        info_label.setObjectName("MutedLabel")
        layout.addWidget(info_label)
        
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Training Data (text):"))
        self.training_data_input = QTextEdit()
        self.training_data_input.setPlaceholderText("Enter training text here (one sentence per line)...")
        self.training_data_input.setMaximumHeight(100)
        data_layout.addWidget(self.training_data_input)
        layout.addLayout(data_layout)
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        batch_size_layout = QHBoxLayout()
        batch_size_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        batch_size_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_size_layout)
        
        train_button = QPushButton("Start Training")
        train_button.clicked.connect(self._start_training)
        layout.addWidget(train_button)
        
        save_button = QPushButton("Save Checkpoint")
        save_button.clicked.connect(self._save_checkpoint)
        layout.addWidget(save_button)
        
        load_button = QPushButton("Load Checkpoint")
        load_button.clicked.connect(self._load_checkpoint)
        layout.addWidget(load_button)
        
        self.training_progress = QLabel("Ready to train")
        self.training_progress.setObjectName("MutedLabel")
        layout.addWidget(self.training_progress)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_info_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #c0c0c0;
                border: 1px solid #222222;
                border-radius: 2px;
                padding: 12px;
            }
        """)
        
        info = """
Avon LLM - Transformer Architecture

Model Configuration:
- Architecture: Decoder-only Transformer
- Attention: Multi-head self-attention
- Activation: GELU
- Positional Encoding: Sinusoidal

Components:
- Multi-head attention mechanism
- Feed-forward networks
- Layer normalization
- Positional encoding
- Token embeddings

Features:
- Text generation with sampling
- Configurable model size
- Training pipeline with checkpoints
- Temperature, top-k, top-p sampling

Status: """ + ("Available" if self.avon_available else "Not Available")
        
        info_text.setText(info)
        layout.addWidget(info_text)
        widget.setLayout(layout)
        return widget

    def _send_chat_message(self):
        message = self.chat_input.text()
        if not message:
            return
        
        self.chat_input.clear()
        
        self._add_chat_message("user", message)
        
        try:
            if self.avon_available and self.tokenizer.vocab_built:
                response = self.avon_generator.generate_text(
                    message,
                    max_length=50,
                    temperature=0.8
                )
            else:
                response = "Avon model is not yet trained. Please train the model first."
            
            self._add_chat_message("ai", response)
            
        except Exception as e:
            self._add_chat_message("ai", f"Error: {str(e)}")

    def _add_chat_message(self, sender: str, message: str):
        self.chat_history.append({
            "sender": sender,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        if sender == "user":
            formatted_message = f"<b>You:</b> {message}"
        else:
            formatted_message = f"<b>Avon:</b> {message}"
        
        self.chat_display.append(formatted_message)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def _apply_config(self):
        try:
            vocab_size = self.vocab_size_spin.value()
            d_model = self.d_model_spin.value()
            n_heads = self.n_heads_spin.value()
            n_layers = self.n_layers_spin.value()
            d_ff = self.d_ff_spin.value()
            dropout = self.dropout_spin.value()
            
            self.avon_config = AvonConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout
            )
            
            self.avon = Avon(self.avon_config)
            self.avon_generator = AvonGenerator(self.avon, self.tokenizer)
            self.avon_trainer = AvonTrainer(self.avon, self.avon_config)
            
            self._add_chat_message("ai", "Configuration applied successfully.")
            
        except Exception as e:
            self._add_chat_message("ai", f"Error applying configuration: {str(e)}")

    def _start_training(self):
        if not self.avon_available:
            self._add_chat_message("ai", "Avon model is not available.")
            return
        
        training_text = self.training_data_input.toPlainText()
        if not training_text:
            self._add_chat_message("ai", "Please enter training data first.")
            return
        
        try:
            sentences = [line.strip() for line in training_text.split('\n') if line.strip()]
            
            if not sentences:
                self._add_chat_message("ai", "No valid training data found.")
                return
            
            if len(sentences) < 10:
                self._add_chat_message("ai", "Warning: Training with very few samples. For meaningful learning, use at least 10-50 diverse sentences.")
            
            self.tokenizer.build_vocab(sentences)
            
            input_ids = []
            target_ids = []
            
            for sentence in sentences:
                ids = self.tokenizer.encode(sentence)
                if len(ids) > 1:
                    input_ids.append(ids[:-1])
                    target_ids.append(ids[1:])
            
            if not input_ids:
                self._add_chat_message("ai", "Training data too short. Please use longer sentences.")
                return
            
            epochs = self.epochs_spin.value()
            batch_size = self.batch_size_spin.value()
            
            self._add_chat_message("ai", f"Starting training on {len(input_ids)} samples for {epochs} epochs...")
            self.training_progress.setText(f"Training on {len(input_ids)} samples...")
            
            losses = self.avon_trainer.train(input_ids, target_ids, epochs, batch_size)
            
            avg_loss = np.mean(losses) if losses else 0.0
            self._add_chat_message("ai", f"Training completed. Final loss: {avg_loss:.4f}")
            self.training_progress.setText(f"Training completed. Loss: {avg_loss:.4f}")
            
            if avg_loss > 0.1:
                self._add_chat_message("ai", "Note: Loss is still high. This is expected with manual gradient computation. For production use, consider using automatic differentiation frameworks like PyTorch or TensorFlow.")
            
        except Exception as e:
            self._add_chat_message("ai", f"Training error: {str(e)}")
            self.training_progress.setText("Training failed")

    def _save_checkpoint(self):
        try:
            self.avon_trainer.save_checkpoint()
            self._add_chat_message("ai", "Checkpoint saved successfully.")
        except Exception as e:
            self._add_chat_message("ai", f"Error saving checkpoint: {str(e)}")

    def _load_checkpoint(self):
        self._add_chat_message("ai", "Checkpoint loading - specify checkpoint path to load.")
