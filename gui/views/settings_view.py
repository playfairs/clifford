from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsView(QWidget):
    def __init__(self, config, config_manager, on_apply=None):
        super().__init__()
        self.config = config
        self.config_manager = config_manager
        self.on_apply = on_apply
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Application appearance, storage paths, and refresh behavior.")
        subtitle.setObjectName("MutedLabel")

        appearance_group = QGroupBox("Appearance")
        appearance_form = QFormLayout(appearance_group)
        appearance_form.setHorizontalSpacing(18)
        appearance_form.setVerticalSpacing(12)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.config.gui.theme)

        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(800, 3840)
        self.window_width_spin.setValue(self.config.gui.window_width)

        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(600, 2160)
        self.window_height_spin.setValue(self.config.gui.window_height)

        appearance_form.addRow("Theme", self.theme_combo)
        appearance_form.addRow("Window width", self.window_width_spin)
        appearance_form.addRow("Window height", self.window_height_spin)

        storage_group = QGroupBox("Storage")
        storage_form = QFormLayout(storage_group)
        storage_form.setHorizontalSpacing(18)
        storage_form.setVerticalSpacing(12)

        self.database_path_input = QLineEdit()
        self.database_path_input.setText(self.config.database.path)

        self.model_path_input = QLineEdit()
        self.model_path_input.setText(self.config.model.save_path)

        self.checkpoint_interval_spin = QSpinBox()
        self.checkpoint_interval_spin.setRange(1, 10000)
        self.checkpoint_interval_spin.setValue(self.config.model.checkpoint_interval)

        self.auto_save_check = QCheckBox("Save trained models automatically")
        self.auto_save_check.setChecked(self.config.model.auto_save)

        storage_form.addRow("Database path", self.database_path_input)
        storage_form.addRow("Model path", self.model_path_input)
        storage_form.addRow("Checkpoint interval", self.checkpoint_interval_spin)
        storage_form.addRow("", self.auto_save_check)

        behavior_group = QGroupBox("Behavior")
        behavior_form = QFormLayout(behavior_group)
        behavior_form.setHorizontalSpacing(18)
        behavior_form.setVerticalSpacing(12)

        self.auto_refresh_check = QCheckBox("Refresh views automatically")
        self.auto_refresh_check.setChecked(self.config.gui.auto_refresh)

        self.refresh_interval_spin = QSpinBox()
        self.refresh_interval_spin.setRange(1, 3600)
        self.refresh_interval_spin.setValue(self.config.gui.refresh_interval)

        behavior_form.addRow("", self.auto_refresh_check)
        behavior_form.addRow("Refresh interval", self.refresh_interval_spin)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        save_button = QPushButton("Save Settings")
        save_button.setObjectName("PrimaryButton")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.status_label)
        button_layout.addWidget(save_button)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(appearance_group)
        layout.addWidget(storage_group)
        layout.addWidget(behavior_group)
        layout.addLayout(button_layout)
        layout.addStretch(1)

        self.setLayout(layout)

    def save_settings(self):
        self.config.gui.theme = self.theme_combo.currentText()
        self.config.gui.window_width = self.window_width_spin.value()
        self.config.gui.window_height = self.window_height_spin.value()
        self.config.gui.auto_refresh = self.auto_refresh_check.isChecked()
        self.config.gui.refresh_interval = self.refresh_interval_spin.value()

        self.config.database.path = self.database_path_input.text().strip() or "database/clifford.db"
        self.config.model.save_path = self.model_path_input.text().strip() or "models"
        self.config.model.checkpoint_interval = self.checkpoint_interval_spin.value()
        self.config.model.auto_save = self.auto_save_check.isChecked()

        self.config_manager.save()
        self.status_label.setText("Saved")

        if self.on_apply:
            self.on_apply()
