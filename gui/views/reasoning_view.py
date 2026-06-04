from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QTabWidget
from PySide6.QtCore import Qt

from clifford.reason import ReasoningEngine, ReasoningStep, Decision, Plan


class ReasoningView(QWidget):
    def __init__(self):
        super().__init__()
        self.reasoning_engine = ReasoningEngine()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        tabs = QTabWidget()
        
        tabs.addTab(self._create_reasoning_tab(), "Reason")
        tabs.addTab(self._create_decisions_tab(), "Decisions")
        tabs.addTab(self._create_plans_tab(), "Plans")
        tabs.addTab(self._create_history_tab(), "History")
        
        layout.addWidget(tabs)
        self.setLayout(layout)

    def _create_reasoning_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Query:"))
        self.query_input = QLineEdit()
        input_layout.addWidget(self.query_input)
        reason_button = QPushButton("Reason")
        reason_button.clicked.connect(self._reason)
        input_layout.addWidget(reason_button)
        layout.addLayout(input_layout)
        
        self.reasoning_output = QTextEdit()
        self.reasoning_output.setReadOnly(True)
        layout.addWidget(QLabel("Reasoning Steps:"))
        layout.addWidget(self.reasoning_output)
        
        widget.setLayout(layout)
        return widget

    def _create_decisions_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Action:"))
        self.action_input = QLineEdit()
        input_layout.addWidget(self.action_input)
        decide_button = QPushButton("Make Decision")
        decide_button.clicked.connect(self._make_decision)
        input_layout.addWidget(decide_button)
        layout.addLayout(input_layout)
        
        self.decisions_table = QTableWidget()
        self.decisions_table.setColumnCount(4)
        self.decisions_table.setHorizontalHeaderLabels(["ID", "Action", "Confidence", "Timestamp"])
        layout.addWidget(self.decisions_table)
        
        widget.setLayout(layout)
        return widget

    def _create_plans_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Goal:"))
        self.goal_input = QLineEdit()
        input_layout.addWidget(self.goal_input)
        create_plan_button = QPushButton("Create Plan")
        create_plan_button.clicked.connect(self._create_plan)
        input_layout.addWidget(create_plan_button)
        layout.addLayout(input_layout)
        
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Steps (comma-separated):"))
        self.steps_input = QLineEdit()
        steps_layout.addWidget(self.steps_input)
        layout.addLayout(steps_layout)
        
        execute_button = QPushButton("Execute Next Step")
        execute_button.clicked.connect(self._execute_plan_step)
        layout.addWidget(execute_button)
        
        self.plan_output = QTextEdit()
        self.plan_output.setReadOnly(True)
        layout.addWidget(QLabel("Current Plan:"))
        layout.addWidget(self.plan_output)
        
        widget.setLayout(layout)
        return widget

    def _create_history_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["ID", "Action", "Confidence", "Timestamp"])
        layout.addWidget(self.history_table)
        
        refresh_button = QPushButton("Refresh History")
        refresh_button.clicked.connect(self._refresh_history)
        layout.addWidget(refresh_button)
        
        widget.setLayout(layout)
        return widget

    def _reason(self):
        query = self.query_input.text()
        if query:
            steps = self.reasoning_engine.reason(query)
            output = ""
            for i, step in enumerate(steps, 1):
                output += f"Step {i}: {step.thought}\n"
                output += f"  Confidence: {step.confidence:.2f}\n"
                output += f"  Evidence: {', '.join(step.evidence)}\n\n"
            self.reasoning_output.setText(output)

    def _make_decision(self):
        action = self.action_input.text()
        if action:
            reasoning_steps = self.reasoning_engine.reason(action)
            decision = self.reasoning_engine.make_decision(action, reasoning_steps)
            self._refresh_decisions()
            self.action_input.clear()

    def _create_plan(self):
        goal = self.goal_input.text()
        steps_str = self.steps_input.text()
        if goal and steps_str:
            steps = [s.strip() for s in steps_str.split(",")]
            plan = self.reasoning_engine.create_plan(goal, steps)
            self._update_plan_display()
            self.goal_input.clear()
            self.steps_input.clear()

    def _execute_plan_step(self):
        step = self.reasoning_engine.execute_plan_step()
        if step:
            self._update_plan_display()
        else:
            self.reasoning_engine.complete_plan()
            self.plan_output.setText("Plan completed!")

    def _update_plan_display(self):
        if self.reasoning_engine.current_plan:
            plan = self.reasoning_engine.current_plan
            output = f"Goal: {plan.goal}\n"
            output += f"Status: {plan.status}\n"
            output += f"Current Step: {plan.current_step}/{len(plan.steps)}\n\n"
            output += "Steps:\n"
            for i, step in enumerate(plan.steps, 1):
                marker = ">> " if i == plan.current_step else "   "
                output += f"{marker}{i}. {step}\n"
            self.plan_output.setText(output)

    def _refresh_decisions(self):
        decisions = self.reasoning_engine.reasoning_history
        self.decisions_table.setRowCount(len(decisions))
        for i, decision in enumerate(decisions):
            self.decisions_table.setItem(i, 0, QTableWidgetItem(decision.id[:8]))
            self.decisions_table.setItem(i, 1, QTableWidgetItem(decision.action))
            self.decisions_table.setItem(i, 2, QTableWidgetItem(str(decision.confidence)))
            self.decisions_table.setItem(i, 3, QTableWidgetItem(decision.timestamp))

    def _refresh_history(self):
        self._refresh_decisions()
