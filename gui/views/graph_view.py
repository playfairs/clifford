from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QTabWidget
from PySide6.QtCore import Qt

from clifford.graph import KnowledgeGraph, Entity, Relationship, Fact


class GraphView(QWidget):
    def __init__(self):
        super().__init__()
        self.knowledge_graph = KnowledgeGraph()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        tabs = QTabWidget()
        
        tabs.addTab(self._create_entities_tab(), "Entities")
        tabs.addTab(self._create_relationships_tab(), "Relationships")
        tabs.addTab(self._create_facts_tab(), "Facts")
        tabs.addTab(self._create_search_tab(), "Search")
        tabs.addTab(self._create_stats_tab(), "Stats")
        
        layout.addWidget(tabs)
        self.setLayout(layout)

    def _create_entities_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Name:"))
        self.entity_name = QLineEdit()
        input_layout.addWidget(self.entity_name)
        input_layout.addWidget(QLabel("Type:"))
        self.entity_type = QLineEdit()
        input_layout.addWidget(self.entity_type)
        add_button = QPushButton("Add Entity")
        add_button.clicked.connect(self._add_entity)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.entities_table = QTableWidget()
        self.entities_table.setColumnCount(4)
        self.entities_table.setHorizontalHeaderLabels(["ID", "Name", "Type", "Confidence"])
        layout.addWidget(self.entities_table)
        
        widget.setLayout(layout)
        return widget

    def _create_relationships_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Source:"))
        self.rel_source = QLineEdit()
        input_layout.addWidget(self.rel_source)
        input_layout.addWidget(QLabel("Target:"))
        self.rel_target = QLineEdit()
        input_layout.addWidget(self.rel_target)
        input_layout.addWidget(QLabel("Type:"))
        self.rel_type = QLineEdit()
        input_layout.addWidget(self.rel_type)
        add_button = QPushButton("Add Relationship")
        add_button.clicked.connect(self._add_relationship)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.relationships_table = QTableWidget()
        self.relationships_table.setColumnCount(4)
        self.relationships_table.setHorizontalHeaderLabels(["ID", "Source", "Target", "Type"])
        layout.addWidget(self.relationships_table)
        
        widget.setLayout(layout)
        return widget

    def _create_facts_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Statement:"))
        self.fact_statement = QLineEdit()
        input_layout.addWidget(self.fact_statement)
        add_button = QPushButton("Add Fact")
        add_button.clicked.connect(self._add_fact)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.facts_table = QTableWidget()
        self.facts_table.setColumnCount(4)
        self.facts_table.setHorizontalHeaderLabels(["ID", "Statement", "Confidence", "Timestamp"])
        layout.addWidget(self.facts_table)
        
        widget.setLayout(layout)
        return widget

    def _create_search_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        search_layout.addWidget(self.search_input)
        search_button = QPushButton("Search Entities")
        search_button.clicked.connect(self._search_entities)
        search_layout.addWidget(search_button)
        search_facts_button = QPushButton("Search Facts")
        search_facts_button.clicked.connect(self._search_facts)
        search_layout.addWidget(search_facts_button)
        layout.addLayout(search_layout)
        
        self.search_table = QTableWidget()
        self.search_table.setColumnCount(3)
        self.search_table.setHorizontalHeaderLabels(["ID", "Content", "Confidence"])
        layout.addWidget(self.search_table)
        
        widget.setLayout(layout)
        return widget

    def _create_stats_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.stats_output = QTextEdit()
        self.stats_output.setReadOnly(True)
        layout.addWidget(self.stats_output)
        
        refresh_button = QPushButton("Refresh Stats")
        refresh_button.clicked.connect(self._refresh_stats)
        layout.addWidget(refresh_button)
        
        widget.setLayout(layout)
        return widget

    def _add_entity(self):
        name = self.entity_name.text()
        entity_type = self.entity_type.text()
        if name and entity_type:
            self.knowledge_graph.add_entity(name, entity_type)
            self.entity_name.clear()
            self.entity_type.clear()
            self._refresh_entities()

    def _add_relationship(self):
        source = self.rel_source.text()
        target = self.rel_target.text()
        rel_type = self.rel_type.text()
        if source and target and rel_type:
            self.knowledge_graph.add_relationship(source, target, rel_type)
            self.rel_source.clear()
            self.rel_target.clear()
            self.rel_type.clear()
            self._refresh_relationships()

    def _add_fact(self):
        statement = self.fact_statement.text()
        if statement:
            self.knowledge_graph.add_fact(statement, [])
            self.fact_statement.clear()
            self._refresh_facts()

    def _search_entities(self):
        query = self.search_input.text()
        if query:
            results = self.knowledge_graph.search_entities(query)
            self._refresh_search_results(results, "entity")

    def _search_facts(self):
        query = self.search_input.text()
        if query:
            results = self.knowledge_graph.search_facts(query)
            self._refresh_search_results(results, "fact")

    def _refresh_entities(self):
        entities = list(self.knowledge_graph.entities.values())
        self.entities_table.setRowCount(len(entities))
        for i, entity in enumerate(entities):
            self.entities_table.setItem(i, 0, QTableWidgetItem(entity.id[:8]))
            self.entities_table.setItem(i, 1, QTableWidgetItem(entity.name))
            self.entities_table.setItem(i, 2, QTableWidgetItem(entity.entity_type))
            self.entities_table.setItem(i, 3, QTableWidgetItem(str(entity.confidence)))

    def _refresh_relationships(self):
        relationships = list(self.knowledge_graph.relationships.values())
        self.relationships_table.setRowCount(len(relationships))
        for i, rel in enumerate(relationships):
            self.relationships_table.setItem(i, 0, QTableWidgetItem(rel.id[:8]))
            self.relationships_table.setItem(i, 1, QTableWidgetItem(rel.source))
            self.relationships_table.setItem(i, 2, QTableWidgetItem(rel.target))
            self.relationships_table.setItem(i, 3, QTableWidgetItem(rel.relation_type))

    def _refresh_facts(self):
        facts = list(self.knowledge_graph.facts.values())
        self.facts_table.setRowCount(len(facts))
        for i, fact in enumerate(facts):
            self.facts_table.setItem(i, 0, QTableWidgetItem(fact.id[:8]))
            self.facts_table.setItem(i, 1, QTableWidgetItem(fact.statement))
            self.facts_table.setItem(i, 2, QTableWidgetItem(str(fact.confidence)))
            self.facts_table.setItem(i, 3, QTableWidgetItem(fact.timestamp))

    def _refresh_search_results(self, results, result_type):
        self.search_table.setRowCount(len(results))
        for i, item in enumerate(results):
            if result_type == "entity":
                self.search_table.setItem(i, 0, QTableWidgetItem(item.id[:8]))
                self.search_table.setItem(i, 1, QTableWidgetItem(item.name))
                self.search_table.setItem(i, 2, QTableWidgetItem(str(item.confidence)))
            else:
                self.search_table.setItem(i, 0, QTableWidgetItem(item.id[:8]))
                self.search_table.setItem(i, 1, QTableWidgetItem(item.statement))
                self.search_table.setItem(i, 2, QTableWidgetItem(str(item.confidence)))

    def _refresh_stats(self):
        stats = self.knowledge_graph.get_stats()
        output = f"Entity Count: {stats['entity_count']}\n"
        output += f"Relationship Count: {stats['relationship_count']}\n"
        output += f"Fact Count: {stats['fact_count']}\n"
        output += f"Entity Types: {stats['entity_types']}\n"
        output += f"Relationship Types: {stats['relationship_types']}\n"
        self.stats_output.setText(output)
