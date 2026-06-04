from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QTextEdit
from PySide6.QtCore import Qt

from clifford.database import Registry
from clifford.search import Search


class ExperimentViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = Registry()
        self.search_engine = Search()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search runs...")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_runs)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_runs)
        
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.refresh_button)
        
        self.run_table = QTableWidget()
        self.run_table.setColumnCount(5)
        self.run_table.setHorizontalHeaderLabels(["Run ID", "Model ID", "Status", "Final Loss", "Start Time"])
        self.run_table.horizontalHeader().setStretchLastSection(True)
        self.run_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.run_table.clicked.connect(self.on_run_selected)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(150)
        
        layout.addLayout(search_layout)
        layout.addWidget(self.run_table)
        layout.addWidget(QLabel("Run Details:"))
        layout.addWidget(self.details_text)
        
        self.setLayout(layout)
        self.refresh_runs()

    def refresh_runs(self):
        runs = self.registry.list_training_runs()
        self.populate_table(runs)

    def search_runs(self):
        query = self.search_input.text()
        if query:
            runs = self.search_engine.search_training_runs(status=query)
        else:
            runs = self.registry.list_training_runs()
        self.populate_table(runs)

    def populate_table(self, runs):
        self.run_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            self.run_table.setItem(row, 0, QTableWidgetItem(run.id[:8]))
            self.run_table.setItem(row, 1, QTableWidgetItem(run.model_id[:8]))
            self.run_table.setItem(row, 2, QTableWidgetItem(run.status))
            self.run_table.setItem(row, 3, QTableWidgetItem(f"{run.final_loss:.6f}" if run.final_loss else "N/A"))
            self.run_table.setItem(row, 4, QTableWidgetItem(run.start_time[:19]))

    def on_run_selected(self, index):
        run_id = self.run_table.item(index.row(), 0).text()
        runs = self.registry.list_training_runs()
        for run in runs:
            if run.id.startswith(run_id):
                details = f"Run ID: {run.id}\n"
                details += f"Model ID: {run.model_id}\n"
                details += f"Status: {run.status}\n"
                details += f"Start Time: {run.start_time}\n"
                details += f"End Time: {run.end_time}\n"
                details += f"Final Loss: {run.final_loss}\n"
                details += f"Config: {run.config}"
                self.details_text.setText(details)
                break
