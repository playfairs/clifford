import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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
        self.setWindowTitle("Clifford Neural Studio")
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
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(240)
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
        
        for label in [
            "Datasets",
            "Model Registry",
            "Trainer",
            "Training Runs",
            "Checkpoints",
            "Memory",
            "Reasoning",
            "Knowledge Graph",
            "Experiments",
            "Settings",
        ]:
            self.sidebar.addItem(label)
        self.settings_index = self.sidebar.count() - 1
        
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
                    background-color: #111827;
                    border: none;
                    padding: 16px 10px;
                    outline: 0;
                }
                QListWidget::item {
                    background-color: transparent;
                    color: #cbd5e1;
                    padding: 13px 16px;
                    border-radius: 6px;
                    margin: 3px 0;
                }
                QListWidget::item:hover {
                    background-color: #1f2937;
                    color: #ffffff;
                }
                QListWidget::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
            """)
        else:
            self.sidebar.setStyleSheet("""
                QListWidget {
                    background-color: #f8fafc;
                    border: none;
                    padding: 16px 10px;
                    outline: 0;
                }
                QListWidget::item {
                    background-color: transparent;
                    color: #334155;
                    padding: 13px 16px;
                    border-radius: 6px;
                    margin: 3px 0;
                }
                QListWidget::item:hover {
                    background-color: #e2e8f0;
                }
                QListWidget::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
            """)

    def change_view(self, index):
        if index == self.settings_index:
            dialog = SettingsDialog(self.config, self.config_manager, self)
            if dialog.exec() == QDialog.Accepted:
                self.apply_theme()
            self.sidebar.setCurrentRow(0)
        elif 0 <= index < self.stacked_widget.count():
            self.stacked_widget.setCurrentIndex(index)
        else:
            self.sidebar.setCurrentRow(0)

    def apply_theme(self):
        self.update_sidebar_style()
        if self.config.gui.theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #0f172a;
                }
                QStackedWidget, QWidget {
                    background-color: #0f172a;
                }
                QPushButton {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #3b82f6;
                    padding: 9px 14px;
                    border-radius: 6px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #1d4ed8;
                }
                QPushButton:pressed {
                    background-color: #1e40af;
                }
                QPushButton:disabled {
                    background-color: #1e293b;
                    border-color: #334155;
                    color: #64748b;
                }
                QLabel {
                    color: #e5e7eb;
                }
                QLabel#PageTitle {
                    color: #f8fafc;
                    font-size: 26px;
                    font-weight: 700;
                }
                QLabel#SectionTitle {
                    color: #f8fafc;
                    font-size: 15px;
                    font-weight: 700;
                }
                QLabel#MutedLabel {
                    color: #94a3b8;
                }
                QLabel#StatusLabel {
                    color: #bfdbfe;
                    font-weight: 700;
                }
                QGroupBox {
                    background-color: #111827;
                    border: 1px solid #243244;
                    border-radius: 8px;
                    color: #e5e7eb;
                    font-weight: 700;
                    margin-top: 12px;
                    padding: 14px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox {
                    background-color: #020617;
                    color: #e5e7eb;
                    border: 1px solid #334155;
                    padding: 8px;
                    border-radius: 6px;
                    selection-background-color: #2563eb;
                }
                QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 1px solid #38bdf8;
                }
                QTableWidget {
                    background-color: #111827;
                    alternate-background-color: #0f172a;
                    color: #e5e7eb;
                    gridline-color: #243244;
                    border: 1px solid #243244;
                    border-radius: 8px;
                }
                QTableWidget::item:selected {
                    background-color: #2563eb;
                }
                QHeaderView::section {
                    background-color: #1f2937;
                    color: #ffffff;
                    border: none;
                    border-bottom: 1px solid #334155;
                    padding: 8px;
                }
                QProgressBar {
                    background-color: #020617;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    text-align: center;
                    color: #e5e7eb;
                    padding: 1px;
                }
                QProgressBar::chunk {
                    background-color: #22c55e;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f8fafc;
                }
                QStackedWidget, QWidget {
                    background-color: #f8fafc;
                }
                QPushButton {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #2563eb;
                    padding: 9px 14px;
                    border-radius: 6px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #1d4ed8;
                }
                QPushButton:pressed {
                    background-color: #1e40af;
                }
                QPushButton:disabled {
                    background-color: #e2e8f0;
                    border-color: #cbd5e1;
                    color: #94a3b8;
                }
                QLabel {
                    color: #0f172a;
                }
                QLabel#PageTitle {
                    color: #0f172a;
                    font-size: 26px;
                    font-weight: 700;
                }
                QLabel#SectionTitle {
                    color: #0f172a;
                    font-size: 15px;
                    font-weight: 700;
                }
                QLabel#MutedLabel {
                    color: #64748b;
                }
                QLabel#StatusLabel {
                    color: #1d4ed8;
                    font-weight: 700;
                }
                QGroupBox {
                    background-color: #ffffff;
                    border: 1px solid #dbe3ef;
                    border-radius: 8px;
                    color: #0f172a;
                    font-weight: 700;
                    margin-top: 12px;
                    padding: 14px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox {
                    background-color: #ffffff;
                    color: #0f172a;
                    border: 1px solid #cbd5e1;
                    padding: 8px;
                    border-radius: 6px;
                }
                QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 1px solid #2563eb;
                }
                QTableWidget {
                    background-color: #ffffff;
                    alternate-background-color: #f8fafc;
                    color: #0f172a;
                    gridline-color: #e2e8f0;
                    border: 1px solid #dbe3ef;
                    border-radius: 8px;
                }
                QTableWidget::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
                QHeaderView::section {
                    background-color: #e2e8f0;
                    color: #0f172a;
                    border: none;
                    border-bottom: 1px solid #cbd5e1;
                    padding: 8px;
                }
                QProgressBar {
                    background-color: #e2e8f0;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    text-align: center;
                    color: #0f172a;
                    padding: 1px;
                }
                QProgressBar::chunk {
                    background-color: #22c55e;
                    border-radius: 5px;
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
