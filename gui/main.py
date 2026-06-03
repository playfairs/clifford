import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from clifford.utils import get_asset_path
from clifford.config import ConfigManager


class CliffordGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Clifford Neural Network Framework")
        self.setGeometry(100, 100, self.config.gui.window_width, self.config.gui.window_height)
        
        icon_path = get_asset_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        from gui.views.dataset_view import DatasetBrowser
        from gui.views.model_view import ModelBrowser
        from gui.views.training_view import TrainingDashboard
        from gui.views.experiment_view import ExperimentViewer
        from gui.views.checkpoint_view import CheckpointExplorer
        
        self.dataset_browser = DatasetBrowser()
        self.model_browser = ModelBrowser()
        self.training_dashboard = TrainingDashboard()
        self.experiment_viewer = ExperimentViewer()
        self.checkpoint_explorer = CheckpointExplorer()
        
        self.tab_widget.addTab(self.dataset_browser, "Datasets")
        self.tab_widget.addTab(self.model_browser, "Models")
        self.tab_widget.addTab(self.training_dashboard, "Training")
        self.tab_widget.addTab(self.experiment_viewer, "Experiments")
        self.tab_widget.addTab(self.checkpoint_explorer, "Checkpoints")
        
        self.apply_theme()

    def apply_theme(self):
        if self.config.gui.theme == "dark":
            style = """
                QMainWindow {
                    background-color: #2b2b2b;
                }
                QTabWidget::pane {
                    border: 1px solid #3c3c3c;
                    background-color: #2b2b2b;
                }
                QTabBar::tab {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    padding: 8px 16px;
                }
                QTabBar::tab:selected {
                    background-color: #4a4a4a;
                }
                QPushButton {
                    background-color: #4a4a4a;
                    color: #ffffff;
                    border: 1px solid #5c5c5c;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #5c5c5c;
                }
                QLabel {
                    color: #ffffff;
                }
                QTableWidget {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    gridline-color: #5c5c5c;
                }
                QTableWidget::item:selected {
                    background-color: #4a4a4a;
                }
                QLineEdit, QTextEdit, QComboBox {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #5c5c5c;
                    padding: 4px;
                }
            """
            self.setStyleSheet(style)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Clifford")
    app.setOrganizationName("Clifford")
    
    window = CliffordGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
