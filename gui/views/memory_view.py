from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QTabWidget

from clifford.memory import WorkingMemory, ShortTermMemory, LongTermMemory, EpisodicMemoryStore, SemanticMemoryStore, PersistentMemory, SearchableMemory, MemoryRanker, MemorySummarizer, MemoryClusterer


class MemoryView(QWidget):
    def __init__(self):
        super().__init__()
        self.working_memory = WorkingMemory()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.episodic_memory = EpisodicMemoryStore()
        self.semantic_memory = SemanticMemoryStore()
        self.persistent_memory = PersistentMemory()
        self.searchable_memory = SearchableMemory()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Memory")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Working, short-term, long-term, and searchable memory stores.")
        subtitle.setObjectName("MutedLabel")

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        
        tabs.addTab(self._create_working_memory_tab(), "Working")
        tabs.addTab(self._create_short_term_memory_tab(), "Short-term")
        tabs.addTab(self._create_long_term_memory_tab(), "Long-term")
        tabs.addTab(self._create_episodic_memory_tab(), "Episodic")
        tabs.addTab(self._create_semantic_memory_tab(), "Semantic")
        tabs.addTab(self._create_persistent_memory_tab(), "Persistent")
        tabs.addTab(self._create_searchable_memory_tab(), "Searchable")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(tabs)
        self.setLayout(layout)

    def _create_working_memory_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Content:"))
        self.working_input = QLineEdit()
        input_layout.addWidget(self.working_input)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_working_memory)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.working_table = QTableWidget()
        self.working_table.setColumnCount(3)
        self.working_table.setHorizontalHeaderLabels(["ID", "Content", "Importance"])
        self._stretch_table(self.working_table)
        layout.addWidget(self.working_table)
        
        widget.setLayout(layout)
        return widget

    def _create_short_term_memory_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Content:"))
        self.short_term_input = QLineEdit()
        input_layout.addWidget(self.short_term_input)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_short_term_memory)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.short_term_table = QTableWidget()
        self.short_term_table.setColumnCount(3)
        self.short_term_table.setHorizontalHeaderLabels(["ID", "Content", "Timestamp"])
        self._stretch_table(self.short_term_table)
        layout.addWidget(self.short_term_table)
        
        widget.setLayout(layout)
        return widget

    def _create_long_term_memory_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Content:"))
        self.long_term_input = QLineEdit()
        input_layout.addWidget(self.long_term_input)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_long_term_memory)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.long_term_table = QTableWidget()
        self.long_term_table.setColumnCount(4)
        self.long_term_table.setHorizontalHeaderLabels(["ID", "Content", "Importance", "Tags"])
        self._stretch_table(self.long_term_table)
        layout.addWidget(self.long_term_table)
        
        widget.setLayout(layout)
        return widget

    def _create_episodic_memory_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Episode:"))
        self.episodic_input = QLineEdit()
        input_layout.addWidget(self.episodic_input)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_episodic_memory)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.episodic_table = QTableWidget()
        self.episodic_table.setColumnCount(3)
        self.episodic_table.setHorizontalHeaderLabels(["ID", "Episode", "Importance"])
        self._stretch_table(self.episodic_table)
        layout.addWidget(self.episodic_table)
        
        widget.setLayout(layout)
        return widget

    def _create_semantic_memory_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Concept:"))
        self.semantic_input = QLineEdit()
        input_layout.addWidget(self.semantic_input)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_semantic_memory)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.semantic_table = QTableWidget()
        self.semantic_table.setColumnCount(3)
        self.semantic_table.setHorizontalHeaderLabels(["ID", "Concept", "Confidence"])
        self._stretch_table(self.semantic_table)
        layout.addWidget(self.semantic_table)
        
        widget.setLayout(layout)
        return widget

    def _create_persistent_memory_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Content:"))
        self.persistent_input = QLineEdit()
        input_layout.addWidget(self.persistent_input)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_persistent_memory)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)
        
        self.persistent_table = QTableWidget()
        self.persistent_table.setColumnCount(3)
        self.persistent_table.setHorizontalHeaderLabels(["ID", "Content", "Timestamp"])
        self._stretch_table(self.persistent_table)
        layout.addWidget(self.persistent_table)
        
        widget.setLayout(layout)
        return widget

    def _create_searchable_memory_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        search_layout.addWidget(self.search_input)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._search_memory)
        search_layout.addWidget(search_button)
        layout.addLayout(search_layout)
        
        self.search_table = QTableWidget()
        self.search_table.setColumnCount(3)
        self.search_table.setHorizontalHeaderLabels(["Content", "Score", "Tags"])
        self._stretch_table(self.search_table)
        layout.addWidget(self.search_table)
        
        widget.setLayout(layout)
        return widget

    def _stretch_table(self, table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _add_working_memory(self):
        content = self.working_input.text()
        if content:
            try:
                self.working_memory.add(content)
                self.working_input.clear()
                self._refresh_working_memory()
            except Exception as e:
                pass

    def _add_short_term_memory(self):
        content = self.short_term_input.text()
        if content:
            try:
                self.short_term_memory.add(content)
                self.short_term_input.clear()
                self._refresh_short_term_memory()
            except Exception as e:
                pass

    def _add_long_term_memory(self):
        content = self.long_term_input.text()
        if content:
            try:
                self.long_term_memory.add(content)
                self.long_term_input.clear()
                self._refresh_long_term_memory()
            except Exception as e:
                pass

    def _add_episodic_memory(self):
        episode = self.episodic_input.text()
        if episode:
            try:
                self.episodic_memory.add_episode(episode)
                self.episodic_input.clear()
                self._refresh_episodic_memory()
            except Exception as e:
                pass

    def _add_semantic_memory(self):
        concept = self.semantic_input.text()
        if concept:
            try:
                self.semantic_memory.add_concept(concept)
                self.semantic_input.clear()
                self._refresh_semantic_memory()
            except Exception as e:
                pass

    def _add_persistent_memory(self):
        content = self.persistent_input.text()
        if content:
            try:
                self.persistent_memory.add(content)
                self.persistent_input.clear()
                self._refresh_persistent_memory()
            except Exception as e:
                pass

    def _search_memory(self):
        query = self.search_input.text()
        if query:
            try:
                results = self.searchable_memory.search(query)
                self._refresh_search_results(results)
            except Exception as e:
                pass

    def _refresh_working_memory(self):
        items = self.working_memory.get_all()
        self.working_table.setRowCount(len(items))
        for i, item in enumerate(items):
            self.working_table.setItem(i, 0, QTableWidgetItem(item.id[:8]))
            self.working_table.setItem(i, 1, QTableWidgetItem(str(item.content)))
            self.working_table.setItem(i, 2, QTableWidgetItem(str(item.importance)))

    def _refresh_short_term_memory(self):
        items = self.short_term_memory.get_all()
        self.short_term_table.setRowCount(len(items))
        for i, item in enumerate(items):
            self.short_term_table.setItem(i, 0, QTableWidgetItem(item.id[:8]))
            self.short_term_table.setItem(i, 1, QTableWidgetItem(str(item.content)))
            self.short_term_table.setItem(i, 2, QTableWidgetItem(item.timestamp))

    def _refresh_long_term_memory(self):
        items = self.long_term_memory.search("")
        self.long_term_table.setRowCount(len(items))
        for i, item in enumerate(items):
            self.long_term_table.setItem(i, 0, QTableWidgetItem(item.id[:8]))
            self.long_term_table.setItem(i, 1, QTableWidgetItem(str(item.content)))
            self.long_term_table.setItem(i, 2, QTableWidgetItem(str(item.importance)))
            self.long_term_table.setItem(i, 3, QTableWidgetItem(", ".join(item.tags)))

    def _refresh_episodic_memory(self):
        episodes = self.episodic_memory.get_recent_episodes()
        self.episodic_table.setRowCount(len(episodes))
        for i, episode in enumerate(episodes):
            self.episodic_table.setItem(i, 0, QTableWidgetItem(episode.id[:8]))
            self.episodic_table.setItem(i, 1, QTableWidgetItem(episode.episode))
            self.episodic_table.setItem(i, 2, QTableWidgetItem(str(episode.importance)))

    def _refresh_semantic_memory(self):
        concepts = self.semantic_memory.search_concepts("")
        self.semantic_table.setRowCount(len(concepts))
        for i, concept in enumerate(concepts):
            self.semantic_table.setItem(i, 0, QTableWidgetItem(concept.id[:8]))
            self.semantic_table.setItem(i, 1, QTableWidgetItem(concept.concept))
            self.semantic_table.setItem(i, 2, QTableWidgetItem(str(concept.confidence)))

    def _refresh_persistent_memory(self):
        items = self.persistent_memory.search("")
        self.persistent_table.setRowCount(len(items))
        for i, item in enumerate(items):
            self.persistent_table.setItem(i, 0, QTableWidgetItem(item.id[:8]))
            self.persistent_table.setItem(i, 1, QTableWidgetItem(str(item.content)))
            self.persistent_table.setItem(i, 2, QTableWidgetItem(item.timestamp))

    def _refresh_search_results(self, results):
        self.search_table.setRowCount(len(results))
        for i, (item, score) in enumerate(results):
            self.search_table.setItem(i, 0, QTableWidgetItem(str(item.content)))
            self.search_table.setItem(i, 1, QTableWidgetItem(str(score)))
            self.search_table.setItem(i, 2, QTableWidgetItem(", ".join(item.tags)))
