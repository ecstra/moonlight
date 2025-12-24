import nltk, logging, json, re, sys
from typing import List, Optional, Dict, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger("moonlight_document_chunker")

class DocumentChunker:
    """
    Document chunker for RAG applications.
    
    Uses a unified hierarchical approach:
    1. Split by headers (markdown) as natural boundaries
    2. Split by paragraphs (double newlines)
    3. Split by sentences (NLTK) if needed
    4. Ensure min/max size compliance
    """
    
    def __init__(
        self,
        min_size: int = 800,
        max_size: int = 5040,
        nltk_data_dir: Optional[str] = None,
    ):
        """
        Initialize the DocumentChunker.
        
        Args:
            min_size: Minimum characters per chunk
            max_size: Maximum characters per chunk  
            nltk_data_dir: Custom NLTK data directory (optional)
        """
        self.min_size = min_size
        self.max_size = max_size
        
        if nltk_data_dir is None:
            script_dir = Path(__file__).parent.resolve()
            nltk_data_dir = script_dir / 'nltk_data'
        
        self.nltk_data_path = Path(nltk_data_dir)
        self._setup_nltk()
        
    def _setup_nltk(self) -> None:
        """Setup NLTK data directory and download required packages."""
        try:
            self.nltk_data_path.mkdir(parents=True, exist_ok=True)
            if str(self.nltk_data_path) not in nltk.data.path:
                nltk.data.path.insert(0, str(self.nltk_data_path))
            
            # Check if punkt is already available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                # Only download if not found, and suppress output
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    nltk.download('punkt', download_dir=str(self.nltk_data_path), quiet=True)
                finally:
                    sys.stdout = old_stdout
        except Exception as e:
            logger.error(f"Failed to setup NLTK: {e}")
            raise
    
    def _split_by_headers(self, text: str) -> List[str]:
        """Split text by markdown headers, keeping headers with their content."""
        # Regex to match markdown headers
        header_pattern = r'^(#{1,6}\s+.+)$'
        lines = text.split('\n')
        
        sections = []
        current_section = []
        
        for line in lines:
            if re.match(header_pattern, line.strip()):
                # Save previous section
                if current_section:
                    sections.append('\n'.join(current_section))
                # Start new section with header
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))
        
        return [s.strip() for s in sections if s.strip()]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph boundaries (double newlines)."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences using NLTK."""
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK failed, using simple sentence split: {e}")
            # Fallback: split by periods, exclamation marks, question marks
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() + '.' for s in sentences if s.strip()]
    
    def _create_chunks(self, sections: List[str]) -> List[str]:
        """Create chunks from sections using hierarchical splitting."""
        chunks = []
        current_chunk = ""
        
        for section in sections:
            # If adding this section keeps us under max_size, add it
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + section
            
            if len(potential_chunk) <= self.max_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is ready if it meets min_size
                if current_chunk and len(current_chunk) >= self.min_size:
                    chunks.append(current_chunk)
                    current_chunk = section
                else:
                    # Current chunk is too small, need to split the section
                    if len(section) > self.max_size:
                        # Section is too large, split it further
                        sub_chunks = self._split_large_section(section)
                        
                        # Try to combine current_chunk with first sub_chunk
                        if current_chunk and sub_chunks:
                            potential = current_chunk + "\n\n" + sub_chunks[0]
                            if len(potential) <= self.max_size:
                                chunks.append(potential)
                                chunks.extend(sub_chunks[1:])
                            else:
                                if len(current_chunk) >= self.min_size:
                                    chunks.append(current_chunk)
                                chunks.extend(sub_chunks)
                        else:
                            if current_chunk and len(current_chunk) >= self.min_size:
                                chunks.append(current_chunk)
                            chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        # Section fits, but current_chunk + section is too big
                        current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return self._ensure_min_size(chunks)
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split a large section into smaller chunks."""
        # First try paragraphs
        paragraphs = self._split_by_paragraphs(section)
        if len(paragraphs) > 1:
            return self._create_chunks(paragraphs)
        
        # Then try sentences
        sentences = self._split_by_sentences(section)
        if len(sentences) > 1:
            return self._create_chunks(sentences)
        
        # Last resort: split by character count
        return self._split_by_length(section)
    
    def _split_by_length(self, text: str) -> List[str]:
        """Split text by length as last resort."""
        chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            potential = current_chunk + (" " if current_chunk else "") + word
            if len(potential) <= self.max_size:
                current_chunk = potential
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _ensure_min_size(self, chunks: List[str]) -> List[str]:
        """Ensure all chunks meet minimum size by combining small chunks."""
        if not chunks:
            return []
        
        result = []
        current = ""
        
        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current + "\n\n" + chunk) <= self.max_size:
                current += "\n\n" + chunk
            else:
                if len(current) >= self.min_size:
                    result.append(current)
                    current = chunk
                else:
                    current += "\n\n" + chunk
        
        if current:
            result.append(current)
        
        # Final pass: merge last chunk if it's too small
        while len(result) > 1 and len(result[-1]) < self.min_size:
            last = result.pop()
            result[-1] += "\n\n" + last
        
        return result
    
    def chunk(
        self, 
        text: str
    ) -> Dict[str, Any]:
        """
        Chunk a document into semantically meaningful pieces.
        
        Args:
            text: Input document text
            
        Returns:
            Dictionary containing chunks and statistics
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        if self.min_size >= self.max_size:
            raise ValueError("min_size must be less than max_size")
        
        # Step 1: Try to split by headers first
        sections = self._split_by_headers(text)
        
        # Step 2: If no headers, split by paragraphs
        if len(sections) <= 1:
            sections = self._split_by_paragraphs(text)
        
        # Step 3: If still too few sections, this is mostly plain text
        if len(sections) <= 2:
            sections = [text]  # Keep as single section for sentence-level splitting
        
        # Step 4: Create final chunks
        final_chunks = self._create_chunks(sections)
        
        sizes = [len(chunk) for chunk in final_chunks]
        stats = {
            'total_chunks': len(final_chunks),
            'size_range': {'min': min(sizes) if sizes else 0, 'max': max(sizes) if sizes else 0},
            'average_size': sum(sizes) // len(sizes) if sizes else 0,
            'total_characters': sum(sizes),
            'sections_found': len(sections)
        }
        
        return {
            'chunks': final_chunks,
            'statistics': stats,
            'config': {
                'min_size': self.min_size,
                'max_size': self.max_size
            }
        }

    def save_chunks(
        self, 
        result: Dict[str, Any], 
        output_dir: str,
        format: str = 'json'
    ) -> None:
        """
        Save chunks to files.
        
        Args:
            result: Result from chunk() method
            output_dir: Output directory
            format: Output format ('json', 'txt', 'md')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'json':
                with open(output_path / 'chunks.json', 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            
            elif format in ['txt', 'md']:
                ext = 'md' if format == 'md' else 'txt'
                for i, chunk in enumerate(result['chunks']):
                    with open(output_path / f'chunk_{i+1:03d}.{ext}', 'w', encoding='utf-8') as f:
                        f.write(chunk)
        except Exception as e:
            logger.error(f"Failed to save chunks to {output_path}: {e}")
            raise