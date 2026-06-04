from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QComboBox
from PySide6.QtCore import Qt

from clifford.database import Registry
from clifford.search import Search


class CheckpointExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = Registry()
        self.search_engine = Search()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        filter_layout = QHBoxLayout()
        self.run_id_input = QLineEdit()
        self.run_id_input.setPlaceholderText("Run ID")
        self.load_button = QPushButton("Load Checkpoints")
        self.load_button.clicked.connect(self.load_checkpoints)
        
        filter_layout.addWidget(QLabel("Run ID:"))
        filter_layout.addWidget(self.run_id_input)
        filter_layout.addWidget(self.load_button)
        
        self.checkpoint_table = QTableWidget()
        self.checkpoint_table.setColumnCount(5)
        self.checkpoint_table.setHorizontalHeaderLabels(["Checkpoint ID", "Epoch", "Loss", "Metrics", "Timestamp"])
        self.checkpoint_table.horizontalHeader().setStretchLastSection(True)
        self.checkpoint_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addLayout(filter_layout)
        layout.addWidget(self.checkpoint_table)
        
        self.setLayout(layout)

    def load_checkpoints(self):
        run_id = self.run_id_input.text()
        if not run_id:
            return
        
        try:
            checkpoints = self.registry.list_checkpoints(run_id)
            self.populate_table(checkpoints)
        except Exception as e:
            self.checkpoint_table.setRowCount(0)

    def populate_table(self, checkpoints):
        self.checkpoint_table.setRowCount(len(checkpoints))
        for row, checkpoint in enumerate(checkpoints):
            self.checkpoint_table.setItem(row, 0, QTableWidgetItem(checkpoint.id[:8]))
            self.checkpoint_table.setItem(row, 1, QTableWidgetItem(str(checkpoint.epoch)))
            self.checkpoint_table.setItem(row, 2, QTableWidgetItem(f"{checkpoint.loss:.6f}"))
            metrics_str = ", ".join(f"{k}: {v}" for k, v in checkpoint.metrics.items())
            self.checkpoint_table.setItem(row, 3, QTableWidgetItem(metrics_str))
            self.checkpoint_table.setItem(row, 4, QTableWidgetItem(checkpoint.timestamp[:19]))
