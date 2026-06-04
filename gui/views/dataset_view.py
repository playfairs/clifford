from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QComboBox
from PySide6.QtCore import Qt

from clifford.database import Registry
from clifford.search import Search


class DatasetBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = Registry()
        self.search_engine = Search()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search datasets...")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_datasets)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_datasets)
        
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.refresh_button)
        
        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(4)
        self.dataset_table.setHorizontalHeaderLabels(["Name", "Samples", "Features", "Classes"])
        self.dataset_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addLayout(search_layout)
        layout.addWidget(self.dataset_table)
        
        self.setLayout(layout)
        self.refresh_datasets()

    def refresh_datasets(self):
        datasets = self.registry.list_datasets()
        self.populate_table(datasets)

    def search_datasets(self):
        query = self.search_input.text()
        if query:
            datasets = self.search_engine.search_datasets(name=query)
        else:
            datasets = self.registry.list_datasets()
        self.populate_table(datasets)

    def populate_table(self, datasets):
        self.dataset_table.setRowCount(len(datasets))
        for row, dataset in enumerate(datasets):
            self.dataset_table.setItem(row, 0, QTableWidgetItem(dataset.name))
            self.dataset_table.setItem(row, 1, QTableWidgetItem(str(dataset.samples)))
            self.dataset_table.setItem(row, 2, QTableWidgetItem(str(dataset.features)))
            self.dataset_table.setItem(row, 3, QTableWidgetItem(str(dataset.classes) if dataset.classes else "N/A"))
