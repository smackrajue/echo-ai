# Changelog - Echo AI

All notable changes to the Echo AI project will be documented in this file.

## [2.1.0] - 2026-01-22
### Fixed
- **Support Email Accuracy**: Corrected the support email address to `rajue@smackcoders.com` across all contact links and buttons.
- **Demo Mode Warnings**: Fixed an issue where "Please process documents first" warning appeared incorrectly while in Demo Mode.
- **URL Processing Status**: Improved visibility of URL processing. The status metric now correctly reflects "Importing..." with a progress counter (e.g., "5/52 URLs") during initial demo data load.

### Improved
- **UI Layout**: Relocated the "Status" metric to the top of the sidebar for better visibility of system readiness.
- **Reliability**: Optimized the incremental loading logic for demo data to prevent startup timeouts on low-resource environments (like Hugging Face Spaces).

## [2.0.0] - 2026-01-21
### Added
- **Persistent Vector Storage**: Migrated from ChromaDB to **Pinecone**. This ensures that uploaded knowledge stays persistent across application restarts and user refreshes.
- **Automated Demo Mode**: Implementation of `demo.txt` for automatic population of knowledge base on the first run.
- **Smart Chunking**: Added incremental, chunked loading for large URL lists (processing 3-5 URLs at a time) to ensure stability during ingestion.
- **Suggested Topics**: New "💡 Suggested Topics" section on the start screen for better user onboarding and quick testing.
- **Multi-Source Support**: Expanded capabilities for processing both PDF uploads and recursive website crawling.
- **Branding**: Official "Powered by smackcoders" branding and professionalized UI aesthetics.

### Improved
- **Memory System**: Enhanced conversational memory using `ConversationBufferMemory` with source citation tracking.
- **Prompt Engineering**: Refined system prompts to force strict adherence to provided context and "NO_ANSWER_FOUND" logic for fallback to support.

## [1.0.0] - 2026-01-19
### Added
- Initial release of the RAG (Retrieval-Augmented Generation) Chatbot.
- Integration with Groq (Llama 3.1) for high-speed inference.
- Basic PDF and URL ingestion using LangChain.
- Local vector storage using ChromaDB.
- Streamlit-based chat interface.
