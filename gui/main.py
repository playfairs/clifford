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
    QVBoxLayout,
    QLabel,
    QStatusBar,
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
        self.setWindowTitle("Clifford")
        self.setGeometry(100, 100, self.config.gui.window_width, self.config.gui.window_height)
        
        icon_path = get_asset_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        header = self._create_header()
        main_layout.addWidget(header)
        
        content_splitter = QSplitter(Qt.Horizontal)
        
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(260)
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
        from gui.views.avon_view import AvonView
        
        self.dataset_browser = DatasetBrowser()
        self.model_browser = ModelBrowser()
        self.training_dashboard = TrainingDashboard()
        self.experiment_viewer = ExperimentViewer()
        self.checkpoint_explorer = CheckpointExplorer()
        self.memory_view = MemoryView()
        self.reasoning_view = ReasoningView()
        self.graph_view = GraphView()
        self.experiments_view = ExperimentsView()
        self.avon_view = AvonView()
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
        self.stacked_widget.addWidget(self.avon_view)
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
            "Avon",
            "Settings",
        ]:
            self.sidebar.addItem(label)
        self.settings_index = self.sidebar.count() - 1
        
        self.sidebar.currentRowChanged.connect(self.change_view)
        self.sidebar.setCurrentRow(0)
        
        content_splitter.addWidget(self.sidebar)
        content_splitter.addWidget(self.stacked_widget)
        content_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(content_splitter)
        
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("StatusBar")
        self.update_status_bar_style()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        self.apply_theme()

    def _create_header(self):
        header = QWidget()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        header_layout.setSpacing(16)
        
        title_label = QLabel("Clifford")
        title_label.setObjectName("HeaderTitle")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        version_label = QLabel("Neural Studio")
        version_label.setObjectName("HeaderSubtitle")
        header_layout.addWidget(version_label)
        
        return header

    def update_sidebar_style(self):
        if self.config.gui.theme == "dark":
            self.sidebar.setStyleSheet("""
                QListWidget {
                    background-color: #0f0f0f;
                    border: none;
                    padding: 16px 12px;
                    outline: 0;
                }
                QListWidget::item {
                    background-color: transparent;
                    color: #888888;
                    padding: 14px 18px;
                    border-radius: 6px;
                    margin: 2px 0;
                    font-weight: 500;
                    font-size: 13px;
                }
                QListWidget::item:hover {
                    background-color: #1a1a1a;
                    color: #c0c0c0;
                }
                QListWidget::item:selected {
                    background-color: #1a1a1a;
                    color: #ffffff;
                    border-left: 3px solid #00ff00;
                }
            """)
        else:
            self.sidebar.setStyleSheet("""
                QListWidget {
                    background-color: #f8fafc;
                    border: none;
                    padding: 16px 12px;
                    outline: 0;
                }
                QListWidget::item {
                    background-color: transparent;
                    color: #334155;
                    padding: 14px 18px;
                    border-radius: 8px;
                    margin: 2px 0;
                    font-size: 13px;
                }
                QListWidget::item:hover {
                    background-color: #e2e8f0;
                }
                QListWidget::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
            """)

    def update_status_bar_style(self):
        if self.config.gui.theme == "dark":
            self.status_bar.setStyleSheet("""
                QStatusBar {
                    background-color: #0f0f0f;
                    color: #666666;
                    border-top: 1px solid #1a1a1a;
                    padding: 4px 12px;
                }
            """)
        else:
            self.status_bar.setStyleSheet("""
                QStatusBar {
                    background-color: #f8fafc;
                    color: #64748b;
                    border-top: 1px solid #e2e8f0;
                    padding: 4px 12px;
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
        self.update_status_bar_style()
        if self.config.gui.theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #0a0a0a;
                }
                QWidget#Header {
                    background-color: #0f0f0f;
                    border-bottom: 1px solid #1a1a1a;
                }
                QLabel#HeaderTitle {
                    color: #ffffff;
                    font-size: 20px;
                    font-weight: 700;
                    letter-spacing: 0.5px;
                }
                QLabel#HeaderSubtitle {
                    color: #666666;
                    font-size: 14px;
                    font-weight: 500;
                }
                QSplitter::handle {
                    background-color: #1a1a1a;
                }
                QStackedWidget, QWidget {
                    background-color: #0a0a0a;
                }
                QPushButton {
                    background-color: #1a1a1a;
                    color: #e0e0e0;
                    border: 1px solid #333333;
                    padding: 8px 14px;
                    border-radius: 2px;
                    font-weight: 500;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background-color: #2a2a2a;
                    border-color: #444444;
                }
                QPushButton:pressed {
                    background-color: #333333;
                }
                QPushButton:disabled {
                    background-color: #0a0a0a;
                    border-color: #1a1a1a;
                    color: #444444;
                }
                QPushButton#SecondaryButton {
                    background-color: #0a0a0a;
                    border-color: #333333;
                    color: #888888;
                }
                QPushButton#SecondaryButton:hover {
                    background-color: #1a1a1a;
                    color: #e0e0e0;
                }
                QLabel {
                    color: #c0c0c0;
                }
                QLabel#PageTitle {
                    color: #ffffff;
                    font-size: 24px;
                    font-weight: 600;
                    letter-spacing: 0.5px;
                }
                QLabel#SectionTitle {
                    color: #e0e0e0;
                    font-size: 14px;
                    font-weight: 600;
                    letter-spacing: 0.3px;
                }
                QLabel#MutedLabel {
                    color: #666666;
                }
                QLabel#StatusLabel {
                    color: #00ff00;
                    font-weight: 600;
                }
                QGroupBox {
                    background-color: #0f0f0f;
                    border: 1px solid #222222;
                    border-radius: 2px;
                    color: #c0c0c0;
                    font-weight: 500;
                    margin-top: 12px;
                    padding: 14px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox {
                    background-color: #0a0a0a;
                    color: #c0c0c0;
                    border: 1px solid #333333;
                    padding: 7px 10px;
                    border-radius: 2px;
                    selection-background-color: #1a1a1a;
                    selection-color: #ffffff;
                    min-height: 22px;
                }
                QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
                    border: 1px solid #555555;
                }
                QComboBox {
                    padding-right: 28px;
                    background-color: #0a0a0a;
                    border: 1px solid #333333;
                    border-radius: 2px;
                    min-height: 24px;
                }
                QComboBox:hover {
                    border-color: #444444;
                }
                QComboBox:focus {
                    border-color: #555555;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 24px;
                    background-color: transparent;
                }
                QComboBox::down-arrow {
                    width: 0;
                    height: 0;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-top: 5px solid #666666;
                    margin-right: 10px;
                }
                QComboBox QAbstractItemView {
                    background-color: #0f0f0f;
                    color: #c0c0c0;
                    border: 1px solid #333333;
                    selection-background-color: #1a1a1a;
                    selection-color: #ffffff;
                    outline: 0;
                    padding: 2px;
                    border-radius: 2px;
                }
                QComboBox QAbstractItemView::item {
                    padding: 6px 12px;
                    min-height: 20px;
                }
                QComboBox QAbstractItemView::item:hover {
                    background-color: #1a1a1a;
                }
                QComboBox QAbstractItemView::item:selected {
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                QCheckBox {
                    color: #c0c0c0;
                    spacing: 8px;
                    min-height: 24px;
                }
                QCheckBox::indicator {
                    width: 14px;
                    height: 14px;
                    border-radius: 1px;
                    border: 1px solid #444444;
                    background-color: #0a0a0a;
                }
                QCheckBox::indicator:checked {
                    background-color: #333333;
                    border-color: #555555;
                }
                QTabWidget::pane {
                    background-color: #0f0f0f;
                    border: 1px solid #222222;
                    border-radius: 2px;
                    top: -1px;
                }
                QTabBar {
                    background-color: transparent;
                }
                QTabBar::tab {
                    background-color: #1a1a1a;
                    color: #888888;
                    border: 1px solid #333333;
                    border-bottom: none;
                    padding: 8px 16px;
                    min-width: 88px;
                    margin-right: 2px;
                }
                QTabBar::tab:first {
                    border-top-left-radius: 2px;
                }
                QTabBar::tab:last {
                    border-top-right-radius: 2px;
                }
                QTabBar::tab:hover {
                    background-color: #2a2a2a;
                    color: #c0c0c0;
                }
                QTabBar::tab:selected {
                    background-color: #0f0f0f;
                    border-color: #444444;
                    color: #ffffff;
                    font-weight: 600;
                }
                QTableWidget {
                    background-color: #0f0f0f;
                    alternate-background-color: #0a0a0a;
                    color: #c0c0c0;
                    gridline-color: #1a1a1a;
                    border: 1px solid #222222;
                    border-radius: 2px;
                    selection-background-color: #1a1a1a;
                }
                QTableWidget::item:selected {
                    background-color: #1a1a1a;
                }
                QHeaderView::section {
                    background-color: #1a1a1a;
                    color: #888888;
                    border: none;
                    border-bottom: 1px solid #333333;
                    padding: 8px;
                }
                QProgressBar {
                    background-color: #0a0a0a;
                    border: 1px solid #333333;
                    border-radius: 2px;
                    text-align: center;
                    color: #c0c0c0;
                    padding: 1px;
                }
                QProgressBar::chunk {
                    background-color: #00ff00;
                    border-radius: 1px;
                }
                QScrollBar:vertical {
                    background-color: #0a0a0a;
                    width: 12px;
                    margin: 0;
                }
                QScrollBar::handle:vertical {
                    background-color: #333333;
                    border-radius: 2px;
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
                QWidget#Header {
                    background-color: #ffffff;
                    border-bottom: 1px solid #e2e8f0;
                }
                QLabel#HeaderTitle {
                    color: #0f172a;
                    font-size: 20px;
                    font-weight: 700;
                    letter-spacing: 0.5px;
                }
                QLabel#HeaderSubtitle {
                    color: #64748b;
                    font-size: 14px;
                    font-weight: 500;
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
