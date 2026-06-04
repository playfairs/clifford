import sqlite3

from PySide6.QtWidgets import QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from clifford.database import Registry


class ExperimentsView(QWidget):
    def __init__(self):
        super().__init__()
        self.registry = Registry()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Experiments")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Group runs and compare experiment outcomes.")
        subtitle.setObjectName("MutedLabel")
        
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        
        tabs.addTab(self._create_experiments_tab(), "Experiments")
        tabs.addTab(self._create_runs_tab(), "Runs")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(tabs)
        self.setLayout(layout)

    def _create_experiments_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Name:"))
        self.experiment_name = QLineEdit()
        input_layout.addWidget(self.experiment_name)
        input_layout.addWidget(QLabel("Description:"))
        self.experiment_desc = QLineEdit()
        input_layout.addWidget(self.experiment_desc)
        add_button = QPushButton("Create Experiment")
        add_button.clicked.connect(self._create_experiment)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.experiments_table = QTableWidget()
        self.experiments_table.setColumnCount(4)
        self.experiments_table.setHorizontalHeaderLabels(["ID", "Name", "Description", "Tags"])
        self._stretch_table(self.experiments_table)
        layout.addWidget(self.experiments_table)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_experiments)
        layout.addWidget(refresh_button)
        
        widget.setLayout(layout)
        return widget

    def _create_runs_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Experiment ID:"))
        self.run_experiment_id = QLineEdit()
        input_layout.addWidget(self.run_experiment_id)
        add_button = QPushButton("Create Run")
        add_button.clicked.connect(self._create_run)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.runs_table = QTableWidget()
        self.runs_table.setColumnCount(5)
        self.runs_table.setHorizontalHeaderLabels(["ID", "Experiment ID", "Model ID", "Status", "Start Time"])
        self._stretch_table(self.runs_table)
        layout.addWidget(self.runs_table)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_runs)
        layout.addWidget(refresh_button)
        
        widget.setLayout(layout)
        return widget

    def _stretch_table(self, table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _create_experiment(self):
        name = self.experiment_name.text()
        description = self.experiment_desc.text()
        if name:
            self.registry.create_experiment(name, description)
            self.experiment_name.clear()
            self.experiment_desc.clear()
            self._refresh_experiments()

    def _create_run(self):
        experiment_id = self.run_experiment_id.text()
        if experiment_id:
            self.registry.create_experiment_run(experiment_id)
            self.run_experiment_id.clear()
            self._refresh_runs()

    def _refresh_experiments(self):
        experiments = self.registry.list_experiments()
        self.experiments_table.setRowCount(len(experiments))
        for i, exp in enumerate(experiments):
            self.experiments_table.setItem(i, 0, QTableWidgetItem(exp["id"][:8]))
            self.experiments_table.setItem(i, 1, QTableWidgetItem(exp["name"]))
            self.experiments_table.setItem(i, 2, QTableWidgetItem(exp.get("description", "")))
            self.experiments_table.setItem(i, 3, QTableWidgetItem(", ".join(exp["tags"])))

    def _refresh_runs(self):
        with sqlite3.connect(self.registry.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, experiment_id, model_id, status, start_time FROM experiment_runs")
            rows = cursor.fetchall()
            
        self.runs_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.runs_table.setItem(i, 0, QTableWidgetItem(row[0][:8]))
            self.runs_table.setItem(i, 1, QTableWidgetItem(row[1][:8]))
            self.runs_table.setItem(i, 2, QTableWidgetItem(row[2][:8] if row[2] else ""))
            self.runs_table.setItem(i, 3, QTableWidgetItem(row[3]))
            self.runs_table.setItem(i, 4, QTableWidgetItem(row[4]))
