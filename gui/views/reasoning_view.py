from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QTextEdit, QTabWidget, QSplitter
from PySide6.QtCore import Qt
from datetime import datetime
import sys
from pathlib import Path

from clifford.reason import ReasoningEngine, ReasoningStep, Decision, Plan

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLM"))
from avon import Avon, AvonConfig, AvonGenerator, Tokenizer


class ReasoningView(QWidget):
    def __init__(self):
        super().__init__()
        self.reasoning_engine = ReasoningEngine()
        
        try:
            self.avon_config = AvonConfig(vocab_size=10000, d_model=256, n_heads=4, n_layers=4, d_ff=1024)
            self.avon = Avon(self.avon_config)
            self.tokenizer = Tokenizer(vocab_size=10000)
            self.avon_generator = AvonGenerator(self.avon, self.tokenizer)
            self.avon_available = True
        except Exception as e:
            self.avon_available = False
            print(f"Avon initialization failed: {e}")
        
        self.chat_history: List[Dict[str, str]] = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Reasoning")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Reasoning traces, decisions, plans, and history.")
        subtitle.setObjectName("MutedLabel")

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        
        tabs.addTab(self._create_chat_tab(), "Chat")
        tabs.addTab(self._create_reasoning_tab(), "Reason")
        tabs.addTab(self._create_decisions_tab(), "Decisions")
        tabs.addTab(self._create_plans_tab(), "Plans")
        tabs.addTab(self._create_history_tab(), "History")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(tabs)
        self.setLayout(layout)

    def _create_chat_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        splitter = QSplitter(Qt.Vertical)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #c0c0c0;
                border: 1px solid #222222;
                border-radius: 2px;
                padding: 12px;
            }
        """)
        splitter.addWidget(self.chat_display)
        
        input_widget = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background-color: #0a0a0a;
                color: #c0c0c0;
                border: 1px solid #333333;
                border-radius: 2px;
                padding: 8px 12px;
            }
            QLineEdit:focus {
                border: 1px solid #555555;
            }
        """)
        self.chat_input.returnPressed.connect(self._send_chat_message)
        input_layout.addWidget(self.chat_input)
        
        send_button = QPushButton("Send")
        send_button.clicked.connect(self._send_chat_message)
        input_layout.addWidget(send_button)
        
        input_widget.setLayout(input_layout)
        splitter.addWidget(input_widget)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        layout.addWidget(splitter)
        widget.setLayout(layout)
        return widget

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
        self._stretch_table(self.decisions_table)
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
        self._stretch_table(self.history_table)
        layout.addWidget(self.history_table)
        
        refresh_button = QPushButton("Refresh History")
        refresh_button.clicked.connect(self._refresh_history)
        layout.addWidget(refresh_button)
        
        widget.setLayout(layout)
        return widget

    def _stretch_table(self, table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _send_chat_message(self):
        message = self.chat_input.text()
        if not message:
            return
        
        self.chat_input.clear()
        
        self._add_chat_message("user", message)
        
        try:
            if self.avon_available and self.tokenizer.vocab_built:
                response = self.avon_generator.generate_text(
                    message,
                    max_length=50,
                    temperature=0.8
                )
            else:
                response = self._fallback_response(message)
            
            self._add_chat_message("ai", response)
            
        except Exception as e:
            self._add_chat_message("ai", f"I encountered an error: {str(e)}")

    def _fallback_response(self, message: str) -> str:
        reasoning_steps = self.reasoning_engine.reason(message)
        if reasoning_steps and len(reasoning_steps) > 0:
            main_thought = reasoning_steps[0].thought
            return f"{main_thought}"
        return "I'm processing your request, but my language model is not yet trained. Please train the Avon model first."

    def _add_chat_message(self, sender: str, message: str):
        self.chat_history.append({
            "sender": sender,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        if sender == "user":
            formatted_message = f"<b>You:</b> {message}"
        else:
            formatted_message = f"<b>AI:</b> {message}"
        
        self.chat_display.append(formatted_message)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def _reason(self):
        query = self.query_input.text()
        if query:
            try:
                steps = self.reasoning_engine.reason(query)
                if not steps:
                    self.reasoning_output.setText("No reasoning steps generated.")
                    return
                output = ""
                for i, step in enumerate(steps, 1):
                    output += f"Step {i}: {step.thought}\n"
                    output += f"  Confidence: {step.confidence:.2f}\n"
                    output += f"  Evidence: {', '.join(step.evidence)}\n\n"
                self.reasoning_output.setText(output)
            except Exception as e:
                self.reasoning_output.setText(f"Error: {str(e)}")

    def _make_decision(self):
        action = self.action_input.text()
        if action:
            try:
                reasoning_steps = self.reasoning_engine.reason(action)
                decision = self.reasoning_engine.make_decision(action, reasoning_steps)
                self._refresh_decisions()
                self.action_input.clear()
            except Exception as e:
                self.reasoning_output.setText(f"Error: {str(e)}")

    def _create_plan(self):
        goal = self.goal_input.text()
        steps_str = self.steps_input.text()
        if goal and steps_str:
            try:
                steps = [s.strip() for s in steps_str.split(",")]
                if not steps:
                    self.plan_output.setText("Error: No steps provided.")
                    return
                plan = self.reasoning_engine.create_plan(goal, steps)
                self._update_plan_display()
                self.goal_input.clear()
                self.steps_input.clear()
            except Exception as e:
                self.plan_output.setText(f"Error: {str(e)}")

    def _execute_plan_step(self):
        try:
            step = self.reasoning_engine.execute_plan_step()
            if step:
                self._update_plan_display()
            else:
                self.reasoning_engine.complete_plan()
                self.plan_output.setText("Plan completed!")
        except Exception as e:
            self.plan_output.setText(f"Error: {str(e)}")

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
