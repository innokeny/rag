from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from .retriever import Retriever
from .generator import Generator
from .utils import format_context

class GraphState(TypedDict):
    question: str
    context_chunks: List[dict]
    answer: str

class RAGGraph:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        self.graph = self._build_graph()

    def _retrieve(self, state: GraphState) -> GraphState:
        question = state['question']
        chunks = self.retriever.retrieve(question)
        return {"context_chunks": chunks}

    def _generate(self, state: GraphState) -> GraphState:
        question = state['question']
        chunks = state.get('context_chunks', [])
        context = format_context(chunks)
        prompt = self.generator.format_prompt(question, context)
        answer = self.generator.generate(prompt)
        return {"answer": answer}

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(GraphState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def run(self, question: str) -> dict:
        result = self.graph.invoke({"question": question})
        return result