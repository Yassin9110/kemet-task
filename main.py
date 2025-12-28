"""
Main Entry Point

Provides the primary interface for the Multilingual RAG Ingestion Pipeline.
Supports command-line usage and programmatic access.
"""

import argparse
import sys
import os
from typing import Optional, List
from pathlib import Path

from src.config.settings import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator, IngestionResult
from src.retrieval.searcher import Searcher, SearchResponse
from src.retrieval.context_expander import ContextExpander, ExpandedContext
from src.models.document import Document
from src.models.chunks import ParentChunk, ChildChunk
from src.storage.vector_storage import create_vector_storage_from_config



class RAGPipeline:
    """
    High-level API for the Multilingual RAG Ingestion Pipeline.
    
    Provides a simple interface for common operations:
    - Document ingestion
    - Search
    - Context expansion
    - Document management
    
    Usage:
        # Initialize
        pipeline = RAGPipeline(cohere_api_key="your-key")
        
        # Ingest a document
        result = pipeline.ingest("document.pdf")
        
        # Search
        results = pipeline.search("What is machine learning?")
        
        # Expand context
        context = pipeline.expand_context(chunk_id)
    """
    
    def __init__(self, cohere_api_key: Optional[str] = None, config: Optional[PipelineConfig] = None, data_dir: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            cohere_api_key: Cohere API key for embeddings.
            config: Optional custom configuration.
            data_dir: Optional custom data directory.
        """
        # Build configuration
        if config:
            self.config = config
        else:
            self.config = PipelineConfig()
            
            # Override API key if provided
            if cohere_api_key:
                self.config.cohere_api_key = cohere_api_key
            elif os.environ.get("COHERE_API_KEY"):
                self.config.cohere_api_key = os.environ["COHERE_API_KEY"]
            
            # Override data directory if provided
            if data_dir:
                self.config.raw_files_path = os.path.join(data_dir, "raw")
                self.config.documents_path = os.path.join(data_dir, "documents.json")
                self.config.parents_path = os.path.join(data_dir, "parents.json")
                self.config.children_path = os.path.join(data_dir, "children.json")
                self.config.edges_path = os.path.join(data_dir, "edges.json")
                self.config.chroma_path = os.path.join(data_dir, "chroma")



        
        
        # Initialize components
        self.vector_storage = create_vector_storage_from_config(self.config)
        self.orchestrator = PipelineOrchestrator(vector_storage = self.vector_storage, config= self.config )
        self._searcher: Optional[Searcher] = None
        self._expander: Optional[ContextExpander] = None
    
    @property
    def searcher(self) -> Searcher:
        """Lazy initialization of searcher."""
        if self._searcher is None:
            self._searcher = Searcher(self.config, self.vector_storage)
        return self._searcher
    
    @property
    def expander(self) -> ContextExpander:
        """Lazy initialization of context expander."""
        if self._expander is None:
            self._expander = ContextExpander(self.config)
        return self._expander
    
    # =========================================================================
    # Ingestion Methods
    # =========================================================================
    
    def ingest(self, file_path: str, metadata: Optional[dict] = None) -> IngestionResult:
        """
        Ingest a document into the pipeline.
        
        Args:
            file_path: Path to the document file.
            metadata: Optional metadata (user_id, source, tags).
            
        Returns:
            IngestionResult with status and statistics.
        """
        return self.orchestrator.ingest(file_path, metadata)
    
    def ingest_directory(self, directory_path: str, recursive: bool = False, metadata: Optional[dict] = None) -> List[IngestionResult]:
        """
        Ingest all supported documents in a directory.
        
        Args:
            directory_path: Path to the directory.
            recursive: Whether to search subdirectories.
            metadata: Optional metadata to apply to all documents.
            
        Returns:
            List of ingestion results.
        """
        results: List[IngestionResult] = []
        
        # Get supported extensions
        supported_extensions = {
            f".{ext}" for ext in self.config.supported_formats
        }
        
        # Find files
        directory = Path(directory_path)
        
        if recursive:
            files = directory.rglob("*")
        else:
            files = directory.glob("*")
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                result = self.ingest(str(file_path), metadata)
                results.append(result)
        
        return results
    
    # =========================================================================
    # Search Methods
    # =========================================================================
    
    def search(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> SearchResponse:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            document_id: Optional filter by document ID.
            
        Returns:
            SearchResponse with results.
        """
        return self.searcher.search(query=query, top_k=top_k, document_id=document_id, include_parents=True)
    
    def hybrid_search(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> SearchResponse:
        """
        Hybrid search combining vector and keyword matching.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            document_id: Optional filter by document ID.
            
        Returns:
            SearchResponse with results.
        """
        return self.searcher.hybrid_search(query=query, top_k=top_k, document_id=document_id)
    
    # =========================================================================
    # Context Methods
    # =========================================================================
    
    def expand_context(self, chunk_id: str, include_siblings: bool = True, include_semantic: bool = True) -> Optional[ExpandedContext]:
        """
        Expand a chunk with surrounding context.
        
        Args:
            chunk_id: Chunk ID to expand.
            include_siblings: Whether to include sibling chunks.
            include_semantic: Whether to include semantic neighbors.
            
        Returns:
            ExpandedContext or None if chunk not found.
        """
        return self.expander.expand(chunk_id=chunk_id, include_siblings=include_siblings, include_semantic=include_semantic)
    
    def get_retrieval_context(self, query: str, top_k: int = 5, max_tokens: Optional[int] = None) -> str:
        """
        Get combined context for RAG.
        
        Searches for relevant chunks and combines them into
        a single context string suitable for LLM input.
        
        Args:
            query: Search query text.
            top_k: Number of chunks to retrieve.
            max_tokens: Maximum tokens in output.
            
        Returns:
            Combined context string.
        """
        response = self.search(query, top_k=top_k)
        
        chunks = [result.chunk for result in response.results]
        
        return self.expander.build_retrieval_context(chunks=chunks, max_tokens=max_tokens, include_section_headers=True)
    
    # =========================================================================
    # Document Management Methods
    # =========================================================================
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get document metadata by ID.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Document or None if not found.
        """
        return self.orchestrator.get_document(document_id)
    
    def get_chunks(self, document_id: str) -> List[ChildChunk]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            List of child chunks.
        """
        return self.orchestrator.get_chunks(document_id)
    
    def get_parents(self, document_id: str) -> List[ParentChunk]:
        """
        Get all parent chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            List of parent chunks.
        """
        return self.orchestrator.get_parents(document_id)
    
    def list_documents(self) -> List[Document]:
        """
        List all ingested documents.
        
        Returns:
            List of all documents.
        """
        return self.orchestrator.list_documents()
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated data.
        
        Args:
            document_id: Document ID to delete.
            
        Returns:
            True if deletion was successful.
        """
        return self.orchestrator.delete_document(document_id)
    
    def get_statistics(self) -> dict:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with statistics.
        """
        return self.orchestrator.get_statistics()
    
    def get_document_outline(self, document_id: str) -> List[dict]:
        """
        Get document outline/structure.
        
        Args:
            document_id: Document ID.
            
        Returns:
            List of section dictionaries.
        """
        return self.expander.get_document_outline(document_id)


# =============================================================================
# Command Line Interface
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="rag-pipeline",
        description="Multilingual RAG Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a document
  python main.py ingest document.pdf

  # Ingest with metadata
  python main.py ingest document.pdf

  # Ingest a directory
  python main.py ingest-dir ./documents --recursive

  # Search
  python main.py search "What is machine learning?"

  # Search with more results
  python main.py search "neural networks" --top-k 10

  # List documents
  python main.py list

  # Get document info
  python main.py info <document-id>

  # Delete document
  python main.py delete <document-id>

  # Show statistics
  python main.py stats
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--api-key",
        type=str,
        help="Cohere API key (or set COHERE_API_KEY env var)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file", type=str, help="File path to ingest")
    ingest_parser.add_argument("--source", type=str, help="Source identifier")
    ingest_parser.add_argument("--tags", type=str, help="Comma-separated tags")
    ingest_parser.add_argument("--user-id", type=str, help="User identifier")
    
    # Ingest directory command
    ingest_dir_parser = subparsers.add_parser("ingest-dir", help="Ingest a directory")
    ingest_dir_parser.add_argument("directory", type=str, help="Directory path")
    ingest_dir_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subdirectories"
    )
    ingest_dir_parser.add_argument("--source", type=str, help="Source identifier")
    ingest_dir_parser.add_argument("--tags", type=str, help="Comma-separated tags")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for chunks")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    search_parser.add_argument(
        "--document-id", "-d",
        type=str,
        help="Filter by document ID"
    )
    search_parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid search"
    )
    
    # List command
    subparsers.add_parser("list", help="List all documents")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get document info")
    info_parser.add_argument("document_id", type=str, help="Document ID")
    info_parser.add_argument(
        "--outline",
        action="store_true",
        help="Show document outline"
    )
    info_parser.add_argument(
        "--chunks",
        action="store_true",
        help="Show chunk details"
    )
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a document")
    delete_parser.add_argument("document_id", type=str, help="Document ID")
    delete_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation"
    )
    
    # Stats command
    subparsers.add_parser("stats", help="Show pipeline statistics")
    
    # Reprocess command
    reprocess_parser = subparsers.add_parser(
        "reprocess",
        help="Reprocess all documents"
    )
    reprocess_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation"
    )
    
    return parser


def format_ingestion_result(result: IngestionResult) -> str:
    """Format ingestion result for display."""
    lines = [
        f"Document ID: {result.document_id}",
        f"Status: {result.status.value}",
    ]
    
    if result.status.value == "completed":
        lines.extend([
            f"Total Tokens: {result.total_tokens:,}",
            f"Hierarchy Depth: {result.hierarchy_depth}",
            f"Parent Chunks: {result.parent_count}",
            f"Child Chunks: {result.child_count}",
            f"Semantic Edges: {result.edge_count}",
            f"Processing Time: {result.processing_duration_ms / 1000:.2f}s",
        ])
    else:
        lines.append(f"Error: {result.error_message}")
    
    return "\n".join(lines)


def format_search_results(response: SearchResponse) -> str:
    """Format search results for display."""
    lines = [
        f"Query: {response.query}",
        f"Results: {response.total_results}",
        f"Search Time: {response.search_time_ms}ms",
        "-" * 60,
    ]
    
    for i, result in enumerate(response.results, 1):
        chunk = result.chunk
        lines.extend([
            f"\n[{i}] Score: {result.score:.4f}",
            f"    Document: {chunk.document_id[:8]}...",
            f"    Section: {' > '.join(chunk.section_path) if chunk.section_path else 'N/A'}",
            f"    Language: {chunk.language.value}",
            f"    Tokens: {chunk.token_count}",
            f"    Text: {chunk.text[:200]}{'...' if len(chunk.text) > 200 else ''}",
        ])
    
    return "\n".join(lines)


def format_document_list(documents: List[Document]) -> str:
    """Format document list for display."""
    if not documents:
        return "No documents found."
    
    lines = [
        f"{'ID':<40} {'Format':<8} {'Tokens':<10} {'Status':<12} {'Created'}",
        "-" * 90,
    ]
    
    for doc in documents:
        created = doc.created_at.strftime("%Y-%m-%d %H:%M") if doc.created_at else "N/A"
        lines.append(
            f"{doc.document_id:<40} "
            f"{doc.format.value:<8} "
            f"{doc.total_tokens:<10,} "
            f"{doc.ingestion_status.value:<12} "
            f"{created}"
        )
    
    return "\n".join(lines)


def format_document_info(
    doc: Document,
    outline: Optional[List[dict]] = None,
    chunks: Optional[List[ChildChunk]] = None
) -> str:
    """Format document info for display."""
    lines = [
        "=" * 60,
        "Document Information",
        "=" * 60,
        f"ID: {doc.document_id}",
        f"File: {doc.file_path}",
        f"Format: {doc.format.value}",
        f"Status: {doc.ingestion_status.value}",
        f"Total Tokens: {doc.total_tokens:,}",
        f"Hierarchy Depth: {doc.hierarchy_depth}",
        f"Hash: {doc.file_hash[:16]}...",
    ]
    
    # Language distribution
    if doc.language_distribution:
        lang_str = ", ".join(
            f"{lang}: {pct:.0%}"
            for lang, pct in doc.language_distribution.items()
            if pct > 0
        )
        lines.append(f"Languages: {lang_str}")
    
    # Metadata
    if doc.upload_metadata:
        lines.append(f"Metadata: {doc.upload_metadata}")
    
    # Timing
    if doc.created_at:
        lines.append(f"Created: {doc.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if doc.processing_duration_ms:
        lines.append(f"Processing Time: {doc.processing_duration_ms / 1000:.2f}s")
    
    # Outline
    if outline:
        lines.extend([
            "",
            "-" * 60,
            "Document Outline",
            "-" * 60,
        ])
        for section in outline:
            indent = "  " * section['depth']
            lines.append(f"{indent}• {section['title']} ({section['token_count']} tokens)")
    
    # Chunks
    if chunks:
        lines.extend([
            "",
            "-" * 60,
            f"Chunks ({len(chunks)} total)",
            "-" * 60,
        ])
        for i, chunk in enumerate(chunks[:10], 1):  # Show first 10
            lines.append(
                f"  [{i}] {chunk.chunk_id[:8]}... "
                f"({chunk.token_count} tokens, {chunk.language.value})"
            )
        if len(chunks) > 10:
            lines.append(f"  ... and {len(chunks) - 10} more")
    
    return "\n".join(lines)


def format_statistics(stats: dict) -> str:
    """Format statistics for display."""
    lines = [
        "=" * 60,
        "Pipeline Statistics",
        "=" * 60,
        f"Total Documents: {stats['total_documents']}",
        f"  - Completed: {stats['completed_documents']}",
        f"  - Failed: {stats['failed_documents']}",
        f"Total Parent Chunks: {stats['total_parents']:,}",
        f"Total Child Chunks: {stats['total_children']:,}",
        f"Total Vectors: {stats['total_vectors']:,}",
        f"Total Semantic Edges: {stats['total_edges']:,}",
    ]
    
    return "\n".join(lines)


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline(
            cohere_api_key=args.api_key,
            data_dir=args.data_dir
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        return 1
    
    # Execute command
    try:
        if args.command == "ingest":
            # Build metadata
            metadata = {}
            if args.source:
                metadata['source'] = args.source
            if args.tags:
                metadata['tags'] = [t.strip() for t in args.tags.split(",")]
            if args.user_id:
                metadata['user_id'] = args.user_id
            
            result = pipeline.ingest(args.file, metadata if metadata else None)
            print(format_ingestion_result(result))
            
            return 0 if result.status.value == "completed" else 1
        
        elif args.command == "ingest-dir":
            # Build metadata
            metadata = {}
            if args.source:
                metadata['source'] = args.source
            if args.tags:
                metadata['tags'] = [t.strip() for t in args.tags.split(",")]
            
            results = pipeline.ingest_directory(
                args.directory,
                recursive=args.recursive,
                metadata=metadata if metadata else None
            )
            
            print(f"Processed {len(results)} documents")
            succeeded = sum(1 for r in results if r.status.value == "completed")
            failed = len(results) - succeeded
            print(f"  Succeeded: {succeeded}")
            print(f"  Failed: {failed}")
            
            return 0 if failed == 0 else 1
        
        elif args.command == "search":
            if args.hybrid:
                response = pipeline.hybrid_search(
                    args.query,
                    top_k=args.top_k,
                    document_id=args.document_id
                )
            else:
                response = pipeline.search(
                    args.query,
                    top_k=args.top_k,
                    document_id=args.document_id
                )
            
            print(format_search_results(response))
            return 0
        
        elif args.command == "list":
            documents = pipeline.list_documents()
            print(format_document_list(documents))
            return 0
        
        elif args.command == "info":
            doc = pipeline.get_document(args.document_id)
            
            if not doc:
                print(f"Document not found: {args.document_id}", file=sys.stderr)
                return 1
            
            outline = None
            if args.outline:
                outline = pipeline.get_document_outline(args.document_id)
            
            chunks = None
            if args.chunks:
                chunks = pipeline.get_chunks(args.document_id)
            
            print(format_document_info(doc, outline, chunks))
            return 0
        
        elif args.command == "delete":
            if not args.force:
                confirm = input(f"Delete document {args.document_id}? [y/N] ")
                if confirm.lower() != 'y':
                    print("Cancelled.")
                    return 0
            
            success = pipeline.delete_document(args.document_id)
            
            if success:
                print(f"Deleted document: {args.document_id}")
                return 0
            else:
                print(f"Failed to delete document: {args.document_id}", file=sys.stderr)
                return 1
        
        elif args.command == "stats":
            stats = pipeline.get_statistics()
            print(format_statistics(stats))
            return 0
        
        elif args.command == "reprocess":
            if not args.force:
                confirm = input("Reprocess all documents? This will delete and re-ingest. [y/N] ")
                if confirm.lower() != 'y':
                    print("Cancelled.")
                    return 0
            
            results = pipeline.orchestrator.reprocess_all()
            
            print(f"Reprocessed {len(results)} documents")
            succeeded = sum(1 for r in results if r.status.value == "completed")
            failed = len(results) - succeeded
            print(f"  Succeeded: {succeeded}")
            print(f"  Failed: {failed}")
            
            return 0 if failed == 0 else 1
        
        else:
            parser.print_help()
            return 0
    
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# =============================================================================
# Convenience Functions for Programmatic Usage
# =============================================================================

def ingest(
    file_path: str,
    metadata: Optional[dict] = None,
    cohere_api_key: Optional[str] = None
) -> IngestionResult:
    """
    Convenience function to ingest a single document.
    
    Args:
        file_path: Path to the document file.
        metadata: Optional metadata.
        cohere_api_key: Optional API key.
        
    Returns:
        IngestionResult.
    """
    pipeline = RAGPipeline(cohere_api_key=cohere_api_key)
    return pipeline.ingest(file_path, metadata)


def search(
    query: str,
    top_k: int = 5,
    cohere_api_key: Optional[str] = None
) -> SearchResponse:
    """
    Convenience function to search.
    
    Args:
        query: Search query text.
        top_k: Number of results.
        cohere_api_key: Optional API key.
        
    Returns:
        SearchResponse.
    """
    pipeline = RAGPipeline(cohere_api_key=cohere_api_key)
    return pipeline.search(query, top_k=top_k)


def get_context(
    query: str,
    top_k: int = 5,
    max_tokens: Optional[int] = None,
    cohere_api_key: Optional[str] = None
) -> str:
    """
    Convenience function to get retrieval context for RAG.
    
    Args:
        query: Search query text.
        top_k: Number of chunks to retrieve.
        max_tokens: Maximum tokens in output.
        cohere_api_key: Optional API key.
        
    Returns:
        Combined context string.
    """
    pipeline = RAGPipeline(cohere_api_key=cohere_api_key)
    return pipeline.get_retrieval_context(query, top_k=top_k, max_tokens=max_tokens)

def query_file(
    file_path: str,
    query: str,
    top_k: int = 5,
    return_context: bool = True,
    cohere_api_key: Optional[str] = None
) -> str | SearchResponse:
    """
    Convenience function to ingest a file (if needed) and query it.
    
    This function handles both ingestion and search in a single call.
    If the file has already been ingested, it skips ingestion and 
    proceeds directly to search.
    
    Args:
        file_path: Path to the document file.
        query: Search query text.
        top_k: Number of results to return (default: 5).
        return_context: If True, return formatted context string for LLM.
                        If False, return full SearchResponse object.
        cohere_api_key: Optional API key (falls back to COHERE_API_KEY env var).
        
    Returns:
        str: Combined context string if return_context=True.
        SearchResponse: Full search results if return_context=False.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If ingestion or search fails.
        
    Example:
        # Get context for LLM
        context = query_file("document.pdf", "What is machine learning?")
        
        # Get full search response
        response = query_file("document.pdf", "neural networks", return_context=False)
    """
    # Validate file exists
    file_path = str(Path(file_path).resolve())
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Initialize pipeline
    pipeline = RAGPipeline(cohere_api_key=cohere_api_key)
    
    # Check if file is already ingested
    # existing_documents = pipeline.list_documents()
    # file_already_ingested = any(
    #     doc.file_path == file_path for doc in existing_documents
    # )
    
    # Ingest if needed
    # if not file_already_ingested:
    result = pipeline.ingest(file_path)
    if result.status.value != "completed":
        raise Exception(f"Ingestion failed: {result.error_message}")
    
    # Return results based on return_context flag
    if return_context:
        return pipeline.get_retrieval_context(query, top_k=top_k)
    else:
        return pipeline.search(query, top_k=top_k)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the RAG Ingestion Pipeline.")
    # parser.add_argument("file", type=str, help="Path to the file to ingest.")
    # args = parser.parse_args()

    # # Initialize the pipeline
    # pipeline = RAGPipeline()

    # # Ingest the file
    # try:
    #     result = pipeline.ingest(args.file)
    #     print("Ingestion successful:", result)
    # except Exception as e:
    #     print("Error during ingestion:", e)


    # Simple usage - get context string for LLM
    context = query_file("docs\نظرية_النسبية_simple.txt", "ما هي النظرية النسبية؟")
    print(context)

    # # Get full search response with scores and metadata
    # response = query_file("path/to/document.pdf", "key concepts", return_context=False)
    # for result in response.results:
    #     print(f"Score: {result.score}, Text: {result.chunk.text[:100]}")

    # # With custom parameters
    # context = query_file(
    #     file_path="research_paper.pdf",
    #     query="methodology used",
    #     top_k=10,
    #     return_context=True,
    #     cohere_api_key="your-api-key"
    # )