import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListWidget,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QWidget,
)

from clifford.utils import get_asset_path
from clifford.config import ConfigManager


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
        from gui.views.settings_view import SettingsView
        
        self.dataset_browser = DatasetBrowser()
        self.model_browser = ModelBrowser()
        self.training_dashboard = TrainingDashboard()
        self.experiment_viewer = ExperimentViewer()
        self.checkpoint_explorer = CheckpointExplorer()
        self.memory_view = MemoryView()
        self.reasoning_view = ReasoningView()
        self.graph_view = GraphView()
        self.experiments_view = ExperimentsView()
        self.settings_view = SettingsView(self.config, self.config_manager, self.apply_settings)
        
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
        self.stacked_widget.addWidget(self.settings_view)
        
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
        if 0 <= index < self.stacked_widget.count():
            self.stacked_widget.setCurrentIndex(index)
        else:
            self.sidebar.setCurrentRow(0)

    def apply_settings(self):
        self.apply_theme()
        self.resize(self.config.gui.window_width, self.config.gui.window_height)

    def apply_theme(self):
        self.update_sidebar_style()
        if self.config.gui.theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #0f172a;
                }
                QSplitter::handle {
                    background-color: #1e293b;
                }
                QStackedWidget, QWidget {
                    background-color: #0f172a;
                }
                QPushButton {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #3b82f6;
                    padding: 8px 14px;
                    border-radius: 6px;
                    font-weight: 600;
                    min-height: 20px;
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
                QPushButton#SecondaryButton {
                    background-color: #1e293b;
                    border-color: #334155;
                    color: #e2e8f0;
                }
                QPushButton#SecondaryButton:hover {
                    background-color: #273449;
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
                    padding: 7px 10px;
                    border-radius: 6px;
                    selection-background-color: #2563eb;
                    min-height: 22px;
                }
                QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 1px solid #38bdf8;
                }
                QComboBox {
                    padding-right: 28px;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 28px;
                }
                QComboBox::down-arrow {
                    width: 0;
                    height: 0;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 6px solid #94a3b8;
                    margin-right: 8px;
                }
                QComboBox QAbstractItemView {
                    background-color: #111827;
                    color: #e5e7eb;
                    border: 1px solid #334155;
                    selection-background-color: #2563eb;
                    selection-color: #ffffff;
                    outline: 0;
                    padding: 4px;
                }
                QCheckBox {
                    color: #e5e7eb;
                    spacing: 8px;
                    min-height: 24px;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border-radius: 4px;
                    border: 1px solid #475569;
                    background-color: #020617;
                }
                QCheckBox::indicator:checked {
                    background-color: #2563eb;
                    border-color: #60a5fa;
                }
                QTabWidget::pane {
                    background-color: #111827;
                    border: 1px solid #243244;
                    border-radius: 8px;
                    top: -1px;
                }
                QTabBar {
                    background-color: transparent;
                }
                QTabBar::tab {
                    background-color: #1e293b;
                    color: #cbd5e1;
                    border: 1px solid #334155;
                    border-bottom: none;
                    padding: 8px 16px;
                    min-width: 88px;
                    margin-right: 2px;
                }
                QTabBar::tab:first {
                    border-top-left-radius: 7px;
                }
                QTabBar::tab:last {
                    border-top-right-radius: 7px;
                }
                QTabBar::tab:hover {
                    background-color: #273449;
                    color: #ffffff;
                }
                QTabBar::tab:selected {
                    background-color: #2563eb;
                    border-color: #3b82f6;
                    color: #ffffff;
                    font-weight: 700;
                }
                QTableWidget {
                    background-color: #111827;
                    alternate-background-color: #0f172a;
                    color: #e5e7eb;
                    gridline-color: #243244;
                    border: 1px solid #243244;
                    border-radius: 8px;
                    selection-background-color: #2563eb;
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
                QScrollBar:vertical {
                    background-color: #0f172a;
                    width: 12px;
                    margin: 0;
                }
                QScrollBar::handle:vertical {
                    background-color: #334155;
                    border-radius: 6px;
                    min-height: 28px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f8fafc;
                }
                QSplitter::handle {
                    background-color: #e2e8f0;
                }
                QStackedWidget, QWidget {
                    background-color: #f8fafc;
                }
                QPushButton {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: 1px solid #2563eb;
                    padding: 8px 14px;
                    border-radius: 6px;
                    font-weight: 600;
                    min-height: 20px;
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
                QPushButton#SecondaryButton {
                    background-color: #ffffff;
                    border-color: #cbd5e1;
                    color: #334155;
                }
                QPushButton#SecondaryButton:hover {
                    background-color: #f1f5f9;
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
                    padding: 7px 10px;
                    border-radius: 6px;
                    min-height: 22px;
                }
                QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 1px solid #2563eb;
                }
                QComboBox {
                    padding-right: 28px;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 28px;
                }
                QComboBox::down-arrow {
                    width: 0;
                    height: 0;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 6px solid #64748b;
                    margin-right: 8px;
                }
                QComboBox QAbstractItemView {
                    background-color: #ffffff;
                    color: #0f172a;
                    border: 1px solid #cbd5e1;
                    selection-background-color: #2563eb;
                    selection-color: #ffffff;
                    outline: 0;
                    padding: 4px;
                }
                QCheckBox {
                    color: #0f172a;
                    spacing: 8px;
                    min-height: 24px;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border-radius: 4px;
                    border: 1px solid #94a3b8;
                    background-color: #ffffff;
                }
                QCheckBox::indicator:checked {
                    background-color: #2563eb;
                    border-color: #2563eb;
                }
                QTabWidget::pane {
                    background-color: #ffffff;
                    border: 1px solid #dbe3ef;
                    border-radius: 8px;
                    top: -1px;
                }
                QTabBar {
                    background-color: transparent;
                }
                QTabBar::tab {
                    background-color: #e2e8f0;
                    color: #334155;
                    border: 1px solid #cbd5e1;
                    border-bottom: none;
                    padding: 8px 16px;
                    min-width: 88px;
                    margin-right: 2px;
                }
                QTabBar::tab:first {
                    border-top-left-radius: 7px;
                }
                QTabBar::tab:last {
                    border-top-right-radius: 7px;
                }
                QTabBar::tab:hover {
                    background-color: #dbeafe;
                    color: #1e3a8a;
                }
                QTabBar::tab:selected {
                    background-color: #2563eb;
                    border-color: #2563eb;
                    color: #ffffff;
                    font-weight: 700;
                }
                QTableWidget {
                    background-color: #ffffff;
                    alternate-background-color: #f8fafc;
                    color: #0f172a;
                    gridline-color: #e2e8f0;
                    border: 1px solid #dbe3ef;
                    border-radius: 8px;
                    selection-background-color: #2563eb;
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
                QScrollBar:vertical {
                    background-color: #f8fafc;
                    width: 12px;
                    margin: 0;
                }
                QScrollBar::handle:vertical {
                    background-color: #cbd5e1;
                    border-radius: 6px;
                    min-height: 28px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0;
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
