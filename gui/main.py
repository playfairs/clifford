import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QListWidget, QListWidgetItem, QSplitter, QDialog, QFormLayout, QLineEdit, QSpinBox, QComboBox, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from clifford.utils import get_asset_path
from clifford.config import ConfigManager


class SettingsDialog(QDialog):
    def __init__(self, config, config_manager, parent=None):
        super().__init__(parent)
        self.config = config
        self.config_manager = config_manager
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Settings")
        self.setFixedWidth(400)
        
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.config.gui.theme)
        
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(800, 1920)
        self.window_width_spin.setValue(self.config.gui.window_width)
        
        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(600, 1080)
        self.window_height_spin.setValue(self.config.gui.window_height)
        
        form_layout.addRow("Theme:", self.theme_combo)
        form_layout.addRow("Window Width:", self.window_width_spin)
        form_layout.addRow("Window Height:", self.window_height_spin)
        
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def save_settings(self):
        self.config.gui.theme = self.theme_combo.currentText()
        self.config.gui.window_width = self.window_width_spin.value()
        self.config.gui.window_height = self.window_height_spin.value()
        self.config_manager.save()
        self.accept()


class CliffordGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Clifford")
        self.setGeometry(100, 100, self.config.gui.window_width, self.config.gui.window_height)
        
        icon_path = get_asset_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Horizontal)
        
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.update_sidebar_style()
        
        from gui.views.dataset_view import DatasetBrowser
        from gui.views.model_view import ModelBrowser
        from gui.views.training_view import TrainingDashboard
        from gui.views.experiment_view import ExperimentViewer
        from gui.views.checkpoint_view import CheckpointExplorer
        from gui.views.memory_view import MemoryView
        from gui.views.reasoning_view import ReasoningView
        from gui.views.graph_view import GraphView
        from gui.views.experiments_view import ExperimentsView
        
        self.dataset_browser = DatasetBrowser()
        self.model_browser = ModelBrowser()
        self.training_dashboard = TrainingDashboard()
        self.experiment_viewer = ExperimentViewer()
        self.checkpoint_explorer = CheckpointExplorer()
        self.memory_view = MemoryView()
        self.reasoning_view = ReasoningView()
        self.graph_view = GraphView()
        self.experiments_view = ExperimentsView()
        
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.dataset_browser)
        self.stacked_widget.addWidget(self.model_browser)
        self.stacked_widget.addWidget(self.training_dashboard)
        self.stacked_widget.addWidget(self.experiment_viewer)
        self.stacked_widget.addWidget(self.checkpoint_explorer)
        self.stacked_widget.addWidget(self.memory_view)
        self.stacked_widget.addWidget(self.reasoning_view)
        self.stacked_widget.addWidget(self.graph_view)
        self.stacked_widget.addWidget(self.experiments_view)
        
        self.sidebar.addItem("Datasets")
        self.sidebar.addItem("Models")
        self.sidebar.addItem("Training")
        self.sidebar.addItem("Experiments")
        self.sidebar.addItem("Checkpoints")
        self.sidebar.addItem("Memory")
        self.sidebar.addItem("Reasoning")
        self.sidebar.addItem("Knowledge Graph")
        self.sidebar.addItem("Experiment Tracking")
        self.sidebar.addItem("Settings")
        
        self.sidebar.currentRowChanged.connect(self.change_view)
        self.sidebar.setCurrentRow(0)
        
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.stacked_widget)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        self.apply_theme()

    def update_sidebar_style(self):
        if self.config.gui.theme == "dark":
            self.sidebar.setStyleSheet("""
                QListWidget {
                    background-color: #252526;
                    border: none;
                    padding: 8px;
                }
                QListWidget::item {
                    background-color: transparent;
                    color: #cccccc;
                    padding: 12px 16px;
                    border-radius: 4px;
                    margin: 2px 0;
                }
                QListWidget::item:hover {
                    background-color: #3c3c3c;
                }
                QListWidget::item:selected {
                    background-color: #0078d4;
                    color: #ffffff;
                }
            """)
        else:
            self.sidebar.setStyleSheet("""
                QListWidget {
                    background-color: #f5f5f5;
                    border: none;
                    padding: 8px;
                }
                QListWidget::item {
                    background-color: transparent;
                    color: #000000;
                    padding: 12px 16px;
                    border-radius: 4px;
                    margin: 2px 0;
                }
                QListWidget::item:hover {
                    background-color: #e0e0e0;
                }
                QListWidget::item:selected {
                    background-color: #0078d4;
                    color: #ffffff;
                }
            """)

    def change_view(self, index):
        if index == 8:
            dialog = SettingsDialog(self.config, self.config_manager, self)
            if dialog.exec() == QDialog.Accepted:
                self.apply_theme()
            self.sidebar.setCurrentRow(0)
        else:
            self.stacked_widget.setCurrentIndex(index)

    def apply_theme(self):
        self.update_sidebar_style()
        if self.config.gui.theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                }
                QStackedWidget {
                    background-color: #1e1e1e;
                }
                QPushButton {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #4a4a4a;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:pressed {
                    background-color: #2a2a2a;
                }
                QLabel {
                    color: #ffffff;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox {
                    background-color: #2a2a2a;
                    color: #ffffff;
                    border: 1px solid #4a4a4a;
                    padding: 6px;
                    border-radius: 4px;
                }
                QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 1px solid #0078d4;
                }
                QTableWidget {
                    background-color: #2a2a2a;
                    color: #ffffff;
                    gridline-color: #3c3c3c;
                    border: 1px solid #3c3c3c;
                    border-radius: 4px;
                }
                QTableWidget::item:selected {
                    background-color: #0078d4;
                }
                QTableWidget::header {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: none;
                    border-bottom: 1px solid #4a4a4a;
                    padding: 8px;
                }
                QProgressBar {
                    background-color: #2a2a2a;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #ffffff;
                }
                QStackedWidget {
                    background-color: #ffffff;
                }
                QPushButton {
                    background-color: #f0f0f0;
                    color: #000000;
                    border: 1px solid #d0d0d0;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
                QLabel {
                    color: #000000;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #d0d0d0;
                    padding: 6px;
                    border-radius: 4px;
                }
                QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 1px solid #0078d4;
                }
                QTableWidget {
                    background-color: #ffffff;
                    color: #000000;
                    gridline-color: #e0e0e0;
                    border: 1px solid #e0e0e0;
                    border-radius: 4px;
                }
                QTableWidget::item:selected {
                    background-color: #0078d4;
                    color: #ffffff;
                }
                QTableWidget::header {
                    background-color: #f0f0f0;
                    color: #000000;
                    border: none;
                    border-bottom: 1px solid #d0d0d0;
                    padding: 8px;
                }
                QProgressBar {
                    background-color: #f0f0f0;
                    border: 1px solid #d0d0d0;
                    border-radius: 4px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
            """)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Clifford")
    app.setOrganizationName("Clifford")
    
    icon_path = get_asset_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    window = CliffordGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
