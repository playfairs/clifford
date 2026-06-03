from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QTextEdit
from PySide6.QtCore import Qt

from clifford import ModelRegistry, SearchEngine, ModelPersistence


class ModelBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = ModelRegistry()
        self.search_engine = SearchEngine()
        self.persistence = ModelPersistence()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models...")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_models)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_models)
        
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.refresh_button)
        
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(3)
        self.model_table.setHorizontalHeaderLabels(["Name", "Created", "Updated"])
        self.model_table.horizontalHeader().setStretchLastSection(True)
        self.model_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.model_table.clicked.connect(self.on_model_selected)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(150)
        
        self.delete_button = QPushButton("Delete Model")
        self.delete_button.clicked.connect(self.delete_model)
        self.delete_button.setEnabled(False)
        
        layout.addLayout(search_layout)
        layout.addWidget(self.model_table)
        layout.addWidget(QLabel("Model Details:"))
        layout.addWidget(self.details_text)
        layout.addWidget(self.delete_button)
        
        self.setLayout(layout)
        self.refresh_models()

    def refresh_models(self):
        models = self.registry.list_models()
        self.populate_table(models)

    def search_models(self):
        query = self.search_input.text()
        if query:
            models = self.search_engine.search_models(name=query)
        else:
            models = self.registry.list_models()
        self.populate_table(models)

    def populate_table(self, models):
        self.model_table.setRowCount(len(models))
        for row, model in enumerate(models):
            self.model_table.setItem(row, 0, QTableWidgetItem(model.name))
            self.model_table.setItem(row, 1, QTableWidgetItem(model.created_at[:19]))
            self.model_table.setItem(row, 2, QTableWidgetItem(model.updated_at[:19]))

    def on_model_selected(self, index):
        self.selected_model_name = self.model_table.item(index.row(), 0).text()
        model = self.registry.get_model(self.selected_model_name)
        if model:
            details = f"Name: {model.name}\n"
            details += f"Architecture: {model.architecture}\n"
            details += f"Hyperparameters: {model.hyperparameters}\n"
            details += f"Created: {model.created_at}\n"
            details += f"Updated: {model.updated_at}"
            self.details_text.setText(details)
            self.delete_button.setEnabled(True)

    def delete_model(self):
        if hasattr(self, 'selected_model_name'):
            self.registry.delete_model(self.selected_model_name)
            self.persistence.delete_model(self.selected_model_name)
            self.refresh_models()
            self.details_text.clear()
            self.delete_button.setEnabled(False)
