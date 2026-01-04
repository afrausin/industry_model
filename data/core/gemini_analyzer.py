"""
Gemini AI Analyzer for Hedgeye Quadrant Framework

Uses Google's Gemini API to analyze qualitative macro documents
and predict quadrant probabilities.
"""

import json
import os
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import google.generativeai as genai

from .data_loader import QualitativeDocument

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class QuadrantProbabilities:
    """Probability distribution across quadrants."""
    quad1: float  # Growth ↑, Inflation ↓
    quad2: float  # Growth ↑, Inflation ↑
    quad3: float  # Growth ↓, Inflation ↓
    quad4: float  # Growth ↓, Inflation ↑
    reasoning: str
    confidence: float
    as_of_date: datetime
    
    def validate(self) -> bool:
        """Check that probabilities sum to 1."""
        total = self.quad1 + self.quad2 + self.quad3 + self.quad4
        return abs(total - 1.0) < 0.01
    
    def most_likely(self) -> int:
        """Return the most likely quadrant."""
        probs = {1: self.quad1, 2: self.quad2, 3: self.quad3, 4: self.quad4}
        return max(probs, key=probs.get)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quad1": self.quad1,
            "quad2": self.quad2,
            "quad3": self.quad3,
            "quad4": self.quad4,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "most_likely": self.most_likely(),
            "as_of_date": self.as_of_date.isoformat(),
        }


@dataclass
class DocumentSummary:
    """Summary of a qualitative document."""
    source: str
    date: datetime
    growth_assessment: str  # "accelerating", "stable", "decelerating"
    inflation_assessment: str  # "accelerating", "stable", "decelerating"
    key_points: List[str]
    risks: List[str]
    raw_summary: str


@dataclass
class PeriodComparison:
    """
    Period-over-period comparison of qualitative documents.
    
    Captures how Fed language and economic assessment has changed
    between two consecutive documents of the same type.
    """
    document_type: str  # "FOMC Statement", "Beige Book", "FOMC Minutes"
    current_date: datetime
    previous_date: datetime
    
    # Change assessments
    growth_change: str  # "improved", "unchanged", "deteriorated"
    inflation_change: str  # "increased", "unchanged", "decreased"
    
    # Language changes
    key_language_changes: List[str]  # Specific phrases that changed
    tone_shift: str  # "more_hawkish", "unchanged", "more_dovish"
    
    # Quadrant implications
    direction_of_travel: str  # "towards_quad1", "towards_quad2", etc.
    transition_signals: List[str]  # Signals suggesting regime change
    
    # Summary
    comparison_summary: str
    
    def to_prompt_text(self) -> str:
        """Format for inclusion in prompts."""
        lines = [
            f"### {self.document_type}: Period Comparison",
            f"Comparing: {self.previous_date.strftime('%b %d, %Y')} → {self.current_date.strftime('%b %d, %Y')}",
            "",
            f"**Growth Change**: {self.growth_change.upper()}",
            f"**Inflation Change**: {self.inflation_change.upper()}",
            f"**Tone Shift**: {self.tone_shift.replace('_', ' ').title()}",
            f"**Direction of Travel**: {self.direction_of_travel.replace('_', ' ').title()}",
        ]
        
        if self.key_language_changes:
            lines.append("")
            lines.append("**Key Language Changes**:")
            for change in self.key_language_changes[:5]:
                lines.append(f"  - {change}")
        
        if self.transition_signals:
            lines.append("")
            lines.append("**Transition Signals**:")
            for signal in self.transition_signals[:3]:
                lines.append(f"  - {signal}")
        
        lines.append("")
        lines.append(f"**Summary**: {self.comparison_summary}")
        
        return "\n".join(lines)


@dataclass
class PortfolioRecommendations:
    """Structured portfolio construction recommendations based on quadrant analysis."""
    
    # Long recommendations (assets to buy/overweight)
    longs: List[Dict[str, str]]  # [{"asset": "Gold", "rationale": "Stagflation hedge", "conviction": "high"}]
    
    # Short recommendations (assets to sell/underweight)
    shorts: List[Dict[str, str]]  # [{"asset": "Growth Stocks", "rationale": "Duration risk", "conviction": "high"}]
    
    # Sector tilts
    sector_overweights: List[str]
    sector_underweights: List[str]
    
    # Risk parameters
    recommended_cash_allocation: float  # 0-1
    risk_level: str  # "defensive", "neutral", "aggressive"
    
    # Hedges
    recommended_hedges: List[str]
    
    # Time horizon and confidence
    time_horizon: str  # "30-day", "90-day"
    confidence: float  # 0-1
    
    # Reasoning
    rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "longs": self.longs,
            "shorts": self.shorts,
            "sector_overweights": self.sector_overweights,
            "sector_underweights": self.sector_underweights,
            "recommended_cash_allocation": self.recommended_cash_allocation,
            "risk_level": self.risk_level,
            "recommended_hedges": self.recommended_hedges,
            "time_horizon": self.time_horizon,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass
class PreprocessedDocument:
    """
    Pre-processed document with quadrant-relevant information extracted.
    
    This is the result of using Flash to extract only the information
    relevant for quadrant analysis from a raw document.
    """
    source: str
    date: datetime
    original_word_count: int
    
    # Growth-related extractions
    growth_signals: List[str]
    growth_direction: str  # "accelerating", "stable", "decelerating", "unclear"
    growth_confidence: float  # 0-1
    
    # Inflation-related extractions
    inflation_signals: List[str]
    inflation_direction: str  # "accelerating", "stable", "decelerating", "unclear"
    inflation_confidence: float  # 0-1
    
    # Forward-looking elements
    forward_guidance: str
    risk_factors: List[str]
    
    # Labor market (key leading indicator)
    labor_market_signals: List[str]
    
    # Consumer/spending signals
    consumer_signals: List[str]
    
    # Summary for downstream use
    quadrant_summary: str
    
    def to_prompt_text(self) -> str:
        """Format for inclusion in downstream prompts."""
        lines = [
            f"### {self.source} ({self.date.strftime('%Y-%m-%d')})",
            f"*Original: {self.original_word_count:,} words → Extracted key signals*",
            "",
            f"**Growth Direction**: {self.growth_direction.upper()} (confidence: {self.growth_confidence:.0%})",
        ]
        
        if self.growth_signals:
            lines.append("Growth Signals:")
            for signal in self.growth_signals[:5]:  # Top 5
                lines.append(f"  - {signal}")
        
        lines.append("")
        lines.append(f"**Inflation Direction**: {self.inflation_direction.upper()} (confidence: {self.inflation_confidence:.0%})")
        
        if self.inflation_signals:
            lines.append("Inflation Signals:")
            for signal in self.inflation_signals[:5]:
                lines.append(f"  - {signal}")
        
        if self.labor_market_signals:
            lines.append("")
            lines.append("**Labor Market**:")
            for signal in self.labor_market_signals[:3]:
                lines.append(f"  - {signal}")
        
        if self.forward_guidance:
            lines.append("")
            lines.append(f"**Forward Guidance**: {self.forward_guidance}")
        
        if self.risk_factors:
            lines.append("")
            lines.append("**Risk Factors**:")
            for risk in self.risk_factors[:3]:
                lines.append(f"  - {risk}")
        
        lines.append("")
        lines.append(f"**Summary**: {self.quadrant_summary}")
        
        return "\n".join(lines)


class GeminiMacroAnalyzer:
    """Analyzes macro documents using Gemini AI."""
    
    SYSTEM_PROMPT = """You are an expert macroeconomic analyst specializing in the Hedgeye 
Risk Range quadrant framework. Your task is to analyze economic data and documents to 
assess the current and future direction of US economic growth and inflation.

The Hedgeye Quadrant Framework:
- Quad 1: Growth ACCELERATING, Inflation DECELERATING (Best for risk assets)
- Quad 2: Growth ACCELERATING, Inflation ACCELERATING (Inflationary boom)
- Quad 3: Growth DECELERATING, Inflation DECELERATING (Deflationary slowdown)
- Quad 4: Growth DECELERATING, Inflation ACCELERATING (Stagflation)

Key principles:
1. Focus on the RATE OF CHANGE (2nd derivative), not absolute levels
2. "Accelerating" means the rate is increasing period-over-period
3. "Decelerating" means the rate is decreasing period-over-period
4. Look for leading indicators that signal transitions between quadrants

When analyzing documents, focus on:
- Forward-looking language about growth expectations
- Inflation trends and Fed's assessment of price pressures
- Labor market conditions as a leading indicator
- Consumer spending patterns
- Manufacturing and services activity
- Financial conditions
"""

    def __init__(
        self, 
        api_key: str, 
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        logs_dir: Optional[Path] = None,
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key
            model: Model name (default: gemini-2.5-flash-preview-05-20)
            temperature: Temperature for generation (default: 0.3)
            logs_dir: Directory to save input/output logs
        """
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.logs_dir = logs_dir or Path(__file__).parent / "logs"
        self.call_counter = 0
        
        # Ensure logs directory exists
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        else:
            self.model = None
    
    def _save_io_log(self, prompt: str, response: str, call_type: str) -> Path:
        """Save input/output to a log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.call_counter += 1
        
        log_file = self.logs_dir / f"{timestamp}_{self.call_counter:03d}_{call_type}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "model": self.model_name,
            "temperature": self.temperature,
            "input_prompt": prompt,
            "input_length_chars": len(prompt),
            "output_response": response,
            "output_length_chars": len(response),
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        return log_file
    
    def _clean_and_parse_json(self, response: str) -> dict:
        """
        Clean and parse JSON response from Gemini.
        
        Handles:
        - Markdown code blocks (```json ... ```)
        - Invalid control characters
        - Unescaped newlines in strings
        """
        import re
        
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            parts = response.split("```")
            if len(parts) >= 2:
                response = parts[1]
                if response.startswith("json"):
                    response = response[4:]
        
        response = response.strip()
        
        # Try to parse directly first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to fix common issues
        # Replace literal newlines in strings with \\n
        # This is a heuristic - find strings and escape their newlines
        try:
            # Remove control characters except \n, \r, \t
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', response)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Last resort: try to extract just the probability values using regex
        try:
            quad1 = float(re.search(r'"quad1_probability"\s*:\s*([\d.]+)', response).group(1))
            quad2 = float(re.search(r'"quad2_probability"\s*:\s*([\d.]+)', response).group(1))
            quad3 = float(re.search(r'"quad3_probability"\s*:\s*([\d.]+)', response).group(1))
            quad4 = float(re.search(r'"quad4_probability"\s*:\s*([\d.]+)', response).group(1))
            
            # Try to extract reasoning
            reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)"(?=\s*,\s*"confidence")', response, re.DOTALL)
            reasoning = reasoning_match.group(1) if reasoning_match else "Extracted via regex fallback"
            
            # Try to extract confidence
            confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            return {
                "quad1_probability": quad1,
                "quad2_probability": quad2,
                "quad3_probability": quad3,
                "quad4_probability": quad4,
                "reasoning": reasoning.replace('\\n', '\n'),
                "confidence": confidence,
            }
        except (AttributeError, ValueError) as e:
            raise json.JSONDecodeError(f"Could not extract probabilities: {e}", response, 0)
    
    def _call_gemini(self, prompt: str, call_type: str = "generic") -> str:
        """
        Make a call to Gemini API with logging.
        
        Args:
            prompt: The prompt to send
            call_type: Type of call for logging (e.g., "summarize", "probability")
            
        Returns:
            Response text from Gemini
        """
        if not self.model:
            raise ValueError("Gemini API key not configured")
        
        # Log input length
        print(f"    [Gemini] Sending prompt: {len(prompt):,} chars to {self.model_name}")
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                top_p=0.8,
                max_output_tokens=32_768,  # Increased for longer responses
            ),
        )
        
        response_text = response.text
        
        # Log output length
        print(f"    [Gemini] Received response: {len(response_text):,} chars")
        
        # Save to log file
        log_file = self._save_io_log(prompt, response_text, call_type)
        print(f"    [Gemini] Saved to: {log_file.name}")
        
        return response_text
    
    def analyze_pdf_multimodal(
        self, 
        pdf_bytes: bytes,
        title: str,
        call_type: str = "pdf_analysis",
    ) -> Dict[str, Any]:
        """
        Analyze a PDF document using Gemini's multimodal capabilities.
        
        This sends the raw PDF bytes directly to Gemini, which can process
        the visual layout and extract information more accurately than
        text-only extraction for complex documents.
        
        Args:
            pdf_bytes: Raw PDF file bytes
            title: Title/description of the PDF
            call_type: Type of call for logging
            
        Returns:
            Parsed analysis results
        """
        if not self.model:
            raise ValueError("Gemini API key not configured")
        
        prompt = f"""{self.SYSTEM_PROMPT}

Analyze the following PDF document: {title}

Provide a structured analysis in the following JSON format:
{{
    "growth_assessment": "accelerating" | "stable" | "decelerating",
    "inflation_assessment": "accelerating" | "stable" | "decelerating",
    "growth_confidence": 0.0 to 1.0,
    "inflation_confidence": 0.0 to 1.0,
    "key_points": [
        "Key insight about growth or inflation 1",
        "Key insight about growth or inflation 2",
        ...
    ],
    "risks": [
        "Identified risk 1",
        "Identified risk 2",
        ...
    ],
    "forward_guidance": "Summary of forward-looking statements",
    "labor_market_assessment": "Summary of labor market conditions",
    "consumer_assessment": "Summary of consumer/spending conditions",
    "summary": "2-3 sentence overall summary focused on quadrant implications"
}}

Focus on extracting information relevant to the Hedgeye Quadrant Framework:
- Rate of change in growth (accelerating/decelerating)
- Rate of change in inflation (accelerating/decelerating)
- Forward-looking indicators and Fed guidance
"""
        
        print(f"    [Gemini] Sending PDF ({len(pdf_bytes):,} bytes): {title}")
        
        # Create multimodal content with PDF
        pdf_part = {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": __import__('base64').b64encode(pdf_bytes).decode('utf-8')
            }
        }
        
        response = self.model.generate_content(
            [prompt, pdf_part],
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                top_p=0.8,
                max_output_tokens=16_384,
            ),
        )
        
        response_text = response.text
        print(f"    [Gemini] Received response: {len(response_text):,} chars")
        
        # Save log (note: we don't save PDF bytes to log, just metadata)
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "model": self.model_name,
            "temperature": self.temperature,
            "pdf_title": title,
            "pdf_size_bytes": len(pdf_bytes),
            "input_prompt": prompt,
            "output_response": response_text,
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.call_counter += 1
        log_file = self.logs_dir / f"{timestamp}_{self.call_counter:03d}_{call_type}.json"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"    [Gemini] Saved to: {log_file.name}")
        
        # Parse response
        return self._clean_and_parse_json(response_text)
    
    def analyze_fomc_minutes_pdfs(
        self,
        pdf_data_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple FOMC Minutes PDFs using multimodal capabilities.
        
        Args:
            pdf_data_list: List of dicts with 'date', 'title', 'pdf_bytes'
            
        Returns:
            List of analysis results
        """
        results = []
        
        for pdf_data in pdf_data_list:
            title = pdf_data.get("title", "FOMC Minutes")
            pdf_bytes = pdf_data.get("pdf_bytes", b"")
            date = pdf_data.get("date")
            
            if not pdf_bytes:
                print(f"Warning: No PDF bytes for {title}")
                continue
            
            try:
                analysis = self.analyze_pdf_multimodal(
                    pdf_bytes=pdf_bytes,
                    title=title,
                    call_type="fomc_minutes_pdf",
                )
                analysis["date"] = date.isoformat() if date else None
                analysis["source"] = "FOMC Minutes (PDF)"
                results.append(analysis)
            except Exception as e:
                print(f"Error analyzing PDF {title}: {e}")
                continue
        
        return results
    
    def preprocess_document(self, doc: QualitativeDocument) -> PreprocessedDocument:
        """
        Pre-process a document to extract only quadrant-relevant information.
        
        This is the FIRST stage of analysis - using Flash to quickly extract
        structured signals from raw documents before downstream analysis.
        
        Args:
            doc: Raw qualitative document (FOMC statement, minutes, Beige Book, etc.)
            
        Returns:
            PreprocessedDocument with extracted quadrant-relevant signals
        """
        prompt = f"""You are a macro analyst extracting information for the Hedgeye Quadrant Framework.

The Hedgeye Quadrant Framework focuses on:
- GROWTH: Is economic growth ACCELERATING or DECELERATING? (Rate of change)
- INFLATION: Is inflation ACCELERATING or DECELERATING? (Rate of change)

Your task is to extract ONLY information relevant to assessing these two dimensions.

## Document to Analyze
Source: {doc.source}
Date: {doc.date.strftime('%B %d, %Y')}

---
{doc.content}
---

## Extraction Task

Extract and structure the quadrant-relevant information from this document.
Focus on:
1. Explicit statements about growth direction or economic activity trends
2. Explicit statements about inflation direction or price pressure trends  
3. Labor market signals (leading indicator for growth)
4. Consumer/spending signals (leading indicator for growth)
5. Forward-looking guidance or expectations
6. Risk factors that could affect growth or inflation trajectory

Provide your extraction in JSON format:
{{
    "growth_signals": [
        "Direct quote or paraphrase about growth/activity trend 1",
        "Direct quote or paraphrase about growth/activity trend 2",
        ...
    ],
    "growth_direction": "accelerating" | "stable" | "decelerating" | "unclear",
    "growth_confidence": 0.0 to 1.0,
    
    "inflation_signals": [
        "Direct quote or paraphrase about inflation/price trend 1",
        "Direct quote or paraphrase about inflation/price trend 2",
        ...
    ],
    "inflation_direction": "accelerating" | "stable" | "decelerating" | "unclear",
    "inflation_confidence": 0.0 to 1.0,
    
    "labor_market_signals": [
        "Key labor market observation 1",
        "Key labor market observation 2"
    ],
    
    "consumer_signals": [
        "Consumer/spending observation 1"
    ],
    
    "forward_guidance": "Summary of forward-looking statements or expectations",
    
    "risk_factors": [
        "Risk that could change growth/inflation trajectory 1",
        "Risk that could change growth/inflation trajectory 2"
    ],
    
    "quadrant_summary": "2-3 sentence summary of what this document implies for growth and inflation direction"
}}

IMPORTANT:
- Extract specific, factual signals - not interpretations
- Focus on RATE OF CHANGE language ("improving", "softening", "accelerating", etc.)
- Include confidence based on how explicit the signals are
- If the document doesn't clearly address growth or inflation, indicate "unclear"

Return ONLY valid JSON.
"""
        
        response = self._call_gemini(prompt, call_type="preprocess_document")
        
        try:
            data = self._clean_and_parse_json(response)
            
            return PreprocessedDocument(
                source=doc.source,
                date=doc.date,
                original_word_count=doc.word_count,
                growth_signals=data.get("growth_signals", []),
                growth_direction=data.get("growth_direction", "unclear"),
                growth_confidence=float(data.get("growth_confidence", 0.5)),
                inflation_signals=data.get("inflation_signals", []),
                inflation_direction=data.get("inflation_direction", "unclear"),
                inflation_confidence=float(data.get("inflation_confidence", 0.5)),
                forward_guidance=data.get("forward_guidance", ""),
                risk_factors=data.get("risk_factors", []),
                labor_market_signals=data.get("labor_market_signals", []),
                consumer_signals=data.get("consumer_signals", []),
                quadrant_summary=data.get("quadrant_summary", ""),
            )
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse preprocessing response: {e}")
            return PreprocessedDocument(
                source=doc.source,
                date=doc.date,
                original_word_count=doc.word_count,
                growth_signals=["Failed to extract"],
                growth_direction="unclear",
                growth_confidence=0.0,
                inflation_signals=["Failed to extract"],
                inflation_direction="unclear",
                inflation_confidence=0.0,
                forward_guidance="",
                risk_factors=[],
                labor_market_signals=[],
                consumer_signals=[],
                quadrant_summary=f"Preprocessing failed: {str(e)[:100]}",
            )
    
    def preprocess_all_documents(
        self, 
        documents: List[QualitativeDocument],
        verbose: bool = True,
    ) -> List[PreprocessedDocument]:
        """
        Pre-process all documents to extract quadrant-relevant information.
        
        This should be called BEFORE any downstream analysis to reduce
        token usage and improve signal quality.
        
        Args:
            documents: List of raw qualitative documents
            verbose: Whether to print progress
            
        Returns:
            List of preprocessed documents
        """
        preprocessed = []
        total_original_words = 0
        
        for i, doc in enumerate(documents):
            if verbose:
                print(f"  Preprocessing [{i+1}/{len(documents)}]: {doc.source} ({doc.word_count:,} words)...")
            
            try:
                result = self.preprocess_document(doc)
                preprocessed.append(result)
                total_original_words += doc.word_count
            except Exception as e:
                print(f"  Warning: Failed to preprocess {doc.source}: {e}")
                # Create a fallback preprocessed doc
                preprocessed.append(PreprocessedDocument(
                    source=doc.source,
                    date=doc.date,
                    original_word_count=doc.word_count,
                    growth_signals=[],
                    growth_direction="unclear",
                    growth_confidence=0.0,
                    inflation_signals=[],
                    inflation_direction="unclear",
                    inflation_confidence=0.0,
                    forward_guidance="",
                    risk_factors=[],
                    labor_market_signals=[],
                    consumer_signals=[],
                    quadrant_summary="Preprocessing failed",
                ))
        
        if verbose:
            # Calculate compression ratio
            extracted_text_len = sum(
                len(p.to_prompt_text()) for p in preprocessed
            )
            compression = (1 - extracted_text_len / (total_original_words * 5)) * 100  # Rough char estimate
            print(f"  → Preprocessed {len(preprocessed)} documents")
            print(f"  → Original: ~{total_original_words:,} words")
            print(f"  → Extracted: ~{extracted_text_len:,} chars (~{compression:.0f}% reduction)")
        
        return preprocessed
    
    def format_preprocessed_docs_for_prompt(
        self,
        preprocessed_docs: List[PreprocessedDocument],
    ) -> str:
        """
        Format preprocessed documents for inclusion in downstream prompts.
        
        This creates a compact, structured summary of all preprocessed documents.
        """
        lines = [
            "## Pre-Extracted Qualitative Signals",
            "",
            "The following signals were extracted from Federal Reserve documents.",
            "Each document has been pre-processed to extract only quadrant-relevant information.",
            "",
        ]
        
        for doc in preprocessed_docs:
            lines.append(doc.to_prompt_text())
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Add aggregate signal summary
        lines.append("## Aggregate Signal Summary")
        lines.append("")
        
        # Count growth directions
        growth_counts = {"accelerating": 0, "stable": 0, "decelerating": 0, "unclear": 0}
        inflation_counts = {"accelerating": 0, "stable": 0, "decelerating": 0, "unclear": 0}
        
        for doc in preprocessed_docs:
            growth_counts[doc.growth_direction] = growth_counts.get(doc.growth_direction, 0) + 1
            inflation_counts[doc.inflation_direction] = inflation_counts.get(doc.inflation_direction, 0) + 1
        
        lines.append(f"**Growth Signal Consensus** (across {len(preprocessed_docs)} documents):")
        for direction, count in sorted(growth_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"  - {direction.upper()}: {count} document(s)")
        
        lines.append("")
        lines.append(f"**Inflation Signal Consensus** (across {len(preprocessed_docs)} documents):")
        for direction, count in sorted(inflation_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"  - {direction.upper()}: {count} document(s)")
        
        return "\n".join(lines)

    def summarize_document(self, doc: QualitativeDocument) -> DocumentSummary:
        """Summarize a single document for quadrant analysis."""
        prompt = f"""{self.SYSTEM_PROMPT}

Analyze the following {doc.source} from {doc.date.strftime('%B %d, %Y')}:

---
{doc.content}
---

Provide a structured analysis in the following JSON format:
{{
    "growth_assessment": "accelerating" | "stable" | "decelerating",
    "inflation_assessment": "accelerating" | "stable" | "decelerating",
    "key_points": ["point 1", "point 2", "point 3"],
    "risks": ["risk 1", "risk 2"],
    "summary": "2-3 sentence summary of the document's macro implications"
}}

Return ONLY valid JSON, no markdown formatting.
"""
        
        response = self._call_gemini(prompt, call_type="summarize_document")
        
        # Parse JSON response
        try:
            # Clean up response if needed
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            data = {
                "growth_assessment": "stable",
                "inflation_assessment": "stable",
                "key_points": ["Unable to parse document"],
                "risks": [],
                "summary": response[:500],
            }
        
        return DocumentSummary(
            source=doc.source,
            date=doc.date,
            growth_assessment=data.get("growth_assessment", "stable"),
            inflation_assessment=data.get("inflation_assessment", "stable"),
            key_points=data.get("key_points", []),
            risks=data.get("risks", []),
            raw_summary=data.get("summary", ""),
        )
    
    def compare_periods(
        self,
        current_doc: QualitativeDocument,
        previous_doc: QualitativeDocument,
    ) -> PeriodComparison:
        """
        Compare two consecutive documents to identify period-over-period changes.
        
        This is key for the Hedgeye framework - we care about the RATE OF CHANGE
        in Fed language and economic assessment, not just absolute levels.
        
        Args:
            current_doc: More recent document
            previous_doc: Prior document (same type)
            
        Returns:
            PeriodComparison with detailed change analysis
        """
        prompt = f"""{self.SYSTEM_PROMPT}

## Task: Period-over-Period Language Change Analysis

Compare these two consecutive {current_doc.source} documents to identify how 
the Fed's assessment has CHANGED. Focus on the DIRECTION OF CHANGE, not absolute levels.

### PREVIOUS ({previous_doc.date.strftime('%B %d, %Y')})
---
{previous_doc.content[:15000]}
---

### CURRENT ({current_doc.date.strftime('%B %d, %Y')})
---
{current_doc.content[:15000]}
---

## Analysis Instructions

1. **Growth Assessment Change**: How has the description of economic activity changed?
   - Look for: "solid" → "moderate", "expanding" → "slowing", etc.
   
2. **Inflation Assessment Change**: How has the description of price pressures changed?
   - Look for: "elevated" → "moderating", "persistent" → "easing", etc.

3. **Tone Shift**: Has the overall tone become more hawkish or dovish?
   - Hawkish: More concern about inflation, less about growth
   - Dovish: More concern about growth, less about inflation

4. **Key Language Changes**: Specific words/phrases that were added, removed, or modified

5. **Transition Signals**: Any language suggesting a shift between quadrants

Provide your analysis in JSON format:
{{
    "growth_change": "improved" | "unchanged" | "deteriorated",
    "inflation_change": "increased" | "unchanged" | "decreased",
    "tone_shift": "more_hawkish" | "unchanged" | "more_dovish",
    "key_language_changes": [
        "Changed 'solid growth' to 'moderate growth'",
        "Added 'downside risks to employment'",
        "Removed reference to 'persistent inflation'"
    ],
    "direction_of_travel": "towards_quad1" | "towards_quad2" | "towards_quad3" | "towards_quad4" | "stable",
    "transition_signals": [
        "Signal suggesting potential regime change 1",
        "Signal suggesting potential regime change 2"
    ],
    "comparison_summary": "2-3 sentence summary of the key changes and their quadrant implications"
}}

IMPORTANT: Focus on what CHANGED, not what stayed the same.
Return ONLY valid JSON.
"""
        
        response = self._call_gemini(prompt, call_type="compare_periods")
        
        try:
            data = self._clean_and_parse_json(response)
            
            return PeriodComparison(
                document_type=current_doc.source,
                current_date=current_doc.date,
                previous_date=previous_doc.date,
                growth_change=data.get("growth_change", "unchanged"),
                inflation_change=data.get("inflation_change", "unchanged"),
                key_language_changes=data.get("key_language_changes", []),
                tone_shift=data.get("tone_shift", "unchanged"),
                direction_of_travel=data.get("direction_of_travel", "stable"),
                transition_signals=data.get("transition_signals", []),
                comparison_summary=data.get("comparison_summary", ""),
            )
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse comparison response: {e}")
            return PeriodComparison(
                document_type=current_doc.source,
                current_date=current_doc.date,
                previous_date=previous_doc.date,
                growth_change="unchanged",
                inflation_change="unchanged",
                key_language_changes=["Failed to parse comparison"],
                tone_shift="unchanged",
                direction_of_travel="stable",
                transition_signals=[],
                comparison_summary=f"Comparison failed: {str(e)[:100]}",
            )
    
    def compare_consecutive_documents(
        self,
        documents: List[QualitativeDocument],
    ) -> List[PeriodComparison]:
        """
        Compare consecutive documents of the same type.
        
        Groups documents by source type (FOMC Statement, Beige Book, etc.)
        and compares each with its predecessor.
        
        Args:
            documents: List of qualitative documents (mixed types OK)
            
        Returns:
            List of PeriodComparison objects
        """
        # Group documents by source type
        by_type: Dict[str, List[QualitativeDocument]] = {}
        for doc in documents:
            source_type = doc.source
            if source_type not in by_type:
                by_type[source_type] = []
            by_type[source_type].append(doc)
        
        # Sort each group by date (most recent first)
        for source_type in by_type:
            by_type[source_type].sort(key=lambda x: x.date, reverse=True)
        
        comparisons = []
        
        # Compare consecutive documents within each type
        for source_type, docs in by_type.items():
            if len(docs) < 2:
                continue  # Need at least 2 docs to compare
            
            # Compare most recent with previous
            current = docs[0]
            previous = docs[1]
            
            print(f"  Comparing {source_type}: {previous.date.strftime('%b %d')} → {current.date.strftime('%b %d')}...")
            
            comparison = self.compare_periods(current, previous)
            comparisons.append(comparison)
        
        return comparisons
    
    def format_period_comparisons_for_prompt(
        self,
        comparisons: List[PeriodComparison],
    ) -> str:
        """Format period comparisons for inclusion in prediction prompt."""
        if not comparisons:
            return ""
        
        lines = [
            "",
            "## Period-over-Period Language Changes",
            "",
            "The following shows how Fed communications have CHANGED between consecutive releases.",
            "This captures the RATE OF CHANGE in Fed language - critical for quadrant transitions.",
            "",
        ]
        
        for comparison in comparisons:
            lines.append(comparison.to_prompt_text())
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Summary of all comparisons
        lines.append("### Overall Direction of Travel")
        
        # Count directions
        directions = [c.direction_of_travel for c in comparisons]
        growth_changes = [c.growth_change for c in comparisons]
        inflation_changes = [c.inflation_change for c in comparisons]
        tone_shifts = [c.tone_shift for c in comparisons]
        
        lines.append(f"- Growth Changes: {', '.join(growth_changes)}")
        lines.append(f"- Inflation Changes: {', '.join(inflation_changes)}")
        lines.append(f"- Tone Shifts: {', '.join(t.replace('_', ' ') for t in tone_shifts)}")
        lines.append(f"- Directions: {', '.join(d.replace('_', ' ') for d in directions)}")
        
        return "\n".join(lines)
    
    def predict_quadrant_probabilities(
        self,
        documents: List[QualitativeDocument],
        quantitative_summary: str,
        gdpnow_estimate: Optional[float] = None,
        use_preprocessing: bool = True,
        use_period_comparisons: bool = True,
    ) -> QuadrantProbabilities:
        """
        Predict quadrant probabilities for the next month.
        
        Args:
            documents: List of recent qualitative documents
            quantitative_summary: Text summary of RoC analysis
            gdpnow_estimate: Current Atlanta Fed GDPNow estimate
            use_preprocessing: If True, preprocess documents first (recommended)
            use_period_comparisons: If True, compare consecutive docs of same type
            
        Returns:
            QuadrantProbabilities with probability distribution
        """
        if use_preprocessing and documents:
            # Stage 1a: Pre-process documents to extract quadrant-relevant info
            print("  Stage 1a: Pre-processing documents with Flash...")
            preprocessed_docs = self.preprocess_all_documents(documents)
            
            # Stage 1b: Compare consecutive documents of same type
            period_comparisons = None
            if use_period_comparisons and len(documents) >= 2:
                print("  Stage 1b: Comparing consecutive documents for language changes...")
                period_comparisons = self.compare_consecutive_documents(documents)
                if period_comparisons:
                    print(f"    → Generated {len(period_comparisons)} period comparisons")
            
            return self.predict_quadrant_from_preprocessed(
                preprocessed_docs=preprocessed_docs,
                quantitative_summary=quantitative_summary,
                gdpnow_estimate=gdpnow_estimate,
                period_comparisons=period_comparisons,
            )
        
        # Legacy mode: send full documents (not recommended for large docs)
        doc_context = []
        for doc in documents:
            doc_context.append(
                f"\n### {doc.source} ({doc.date.strftime('%B %d, %Y')})\n"
                f"{doc.content}\n"
            )
        
        gdpnow_text = ""
        if gdpnow_estimate is not None:
            gdpnow_text = f"\n## Atlanta Fed GDPNow\nCurrent estimate: {gdpnow_estimate}% GDP growth\n"
        
        prompt = f"""{self.SYSTEM_PROMPT}

Based on the following macroeconomic data and documents, estimate the probability 
of each Hedgeye quadrant for the NEXT MONTH (30-day forward looking).

## Quantitative Rate of Change Analysis
{quantitative_summary}
{gdpnow_text}

## Recent Federal Reserve Communications
{"".join(doc_context)}

---

Analyze all the information and provide probability estimates for each quadrant 
for the next month. Consider:
1. The current direction of growth and inflation from quantitative data
2. Forward-looking language in Fed communications
3. Risk factors that could cause transitions
4. Historical patterns around similar conditions

Provide your response in JSON format:
{{
    "quad1_probability": 0.XX,
    "quad2_probability": 0.XX,
    "quad3_probability": 0.XX,
    "quad4_probability": 0.XX,
    "reasoning": "Detailed explanation of your probability assessment",
    "confidence": 0.XX,
    "key_factors": ["factor 1", "factor 2", "factor 3"]
}}

The probabilities MUST sum to 1.0. Return ONLY valid JSON.
"""
        
        response = self._call_gemini(prompt, call_type="predict_quadrant_simple")
        
        try:
            data = self._clean_and_parse_json(response)
            
            # Normalize probabilities to sum to 1
            total = (
                data.get("quad1_probability", 0.25) +
                data.get("quad2_probability", 0.25) +
                data.get("quad3_probability", 0.25) +
                data.get("quad4_probability", 0.25)
            )
            
            return QuadrantProbabilities(
                quad1=data.get("quad1_probability", 0.25) / total,
                quad2=data.get("quad2_probability", 0.25) / total,
                quad3=data.get("quad3_probability", 0.25) / total,
                quad4=data.get("quad4_probability", 0.25) / total,
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
                as_of_date=datetime.now(),
            )
            
        except json.JSONDecodeError:
            # Return uniform distribution on failure
            return QuadrantProbabilities(
                quad1=0.25,
                quad2=0.25,
                quad3=0.25,
                quad4=0.25,
                reasoning=f"Failed to parse response: {response[:500]}",
                confidence=0.0,
                as_of_date=datetime.now(),
            )
    
    def predict_quadrant_from_preprocessed(
        self,
        preprocessed_docs: List[PreprocessedDocument],
        quantitative_summary: str,
        gdpnow_estimate: Optional[float] = None,
        period_comparisons: Optional[List[PeriodComparison]] = None,
    ) -> QuadrantProbabilities:
        """
        Predict quadrant probabilities using PRE-PROCESSED documents.
        
        This is the recommended method - documents have already been
        preprocessed to extract only quadrant-relevant signals.
        
        Args:
            preprocessed_docs: List of preprocessed documents
            quantitative_summary: Text summary of RoC analysis
            gdpnow_estimate: Current Atlanta Fed GDPNow estimate
            period_comparisons: Optional period-over-period comparisons
            
        Returns:
            QuadrantProbabilities with probability distribution
        """
        print("  Stage 2: Generating quadrant probabilities from extracted signals...")
        
        # Format preprocessed docs
        docs_text = self.format_preprocessed_docs_for_prompt(preprocessed_docs)
        
        gdpnow_text = ""
        if gdpnow_estimate is not None:
            gdpnow_text = f"\n## Atlanta Fed GDPNow\nCurrent Real GDP Growth Estimate: **{gdpnow_estimate}%**\n"
        
        # Format period comparisons
        comparisons_text = ""
        if period_comparisons:
            comparisons_text = self.format_period_comparisons_for_prompt(period_comparisons)
        
        prompt = f"""{self.SYSTEM_PROMPT}

Based on the pre-extracted signals, period-over-period changes, and quantitative data below, 
estimate the probability of each Hedgeye quadrant for the NEXT MONTH (30-day forward looking).

## Quantitative Rate of Change Analysis
{quantitative_summary}
{gdpnow_text}

{docs_text}

{comparisons_text}

---

## Your Task

You have THREE sources of information:
1. **Document Signals**: Pre-extracted growth/inflation signals from Fed communications
2. **Period Comparisons**: How Fed language has CHANGED between consecutive releases
3. **Quantitative Data**: Rate of Change analysis from economic time series

The period comparisons are CRITICAL - they show the 2nd derivative of Fed thinking,
which often leads actual economic data changes.

Key considerations:
1. Weight recent documents more heavily than older ones
2. Look for CONSISTENCY between quant data and qualitative signals
3. Period-over-period language changes often signal quadrant transitions BEFORE data
4. If language is shifting towards a quadrant, probability should tilt that direction
5. Forward guidance and tone shifts are leading indicators

Provide your response in JSON format:
{{
    "quad1_probability": 0.XX,
    "quad2_probability": 0.XX,
    "quad3_probability": 0.XX,
    "quad4_probability": 0.XX,
    "reasoning": "Synthesis of quantitative signals, qualitative signals, AND language changes",
    "confidence": 0.XX,
    "quant_qual_alignment": "aligned" | "mixed" | "divergent",
    "language_trend": "stable" | "shifting_hawkish" | "shifting_dovish" | "mixed_signals",
    "transition_risk": "low" | "medium" | "high",
    "key_factors": ["factor 1", "factor 2", "factor 3"]
}}

The probabilities MUST sum to 1.0. Return ONLY valid JSON.
"""
        
        response = self._call_gemini(prompt, call_type="predict_quadrant_preprocessed")
        
        try:
            data = self._clean_and_parse_json(response)
            
            # Normalize probabilities to sum to 1
            total = (
                data.get("quad1_probability", 0.25) +
                data.get("quad2_probability", 0.25) +
                data.get("quad3_probability", 0.25) +
                data.get("quad4_probability", 0.25)
            )
            
            reasoning = data.get("reasoning", "")
            if data.get("quant_qual_alignment"):
                reasoning += f"\n\nQuant/Qual Alignment: {data['quant_qual_alignment']}"
            if data.get("language_trend"):
                reasoning += f"\nLanguage Trend: {data['language_trend'].replace('_', ' ').title()}"
            if data.get("transition_risk"):
                reasoning += f"\nTransition Risk: {data['transition_risk'].upper()}"
            if data.get("key_factors"):
                reasoning += f"\n\nKey Factors:\n" + "\n".join(f"- {f}" for f in data["key_factors"])
            
            return QuadrantProbabilities(
                quad1=data.get("quad1_probability", 0.25) / total,
                quad2=data.get("quad2_probability", 0.25) / total,
                quad3=data.get("quad3_probability", 0.25) / total,
                quad4=data.get("quad4_probability", 0.25) / total,
                reasoning=reasoning,
                confidence=data.get("confidence", 0.5),
                as_of_date=datetime.now(),
            )
            
        except json.JSONDecodeError:
            return QuadrantProbabilities(
                quad1=0.25,
                quad2=0.25,
                quad3=0.25,
                quad4=0.25,
                reasoning=f"Failed to parse response: {response[:500]}",
                confidence=0.0,
                as_of_date=datetime.now(),
            )
    
    def analyze_all_documents(
        self,
        documents: List[QualitativeDocument],
    ) -> List[DocumentSummary]:
        """
        Analyze all qualitative documents and return structured summaries.
        
        This is the first stage - extracting structured insights from each doc.
        """
        summaries = []
        for doc in documents:
            try:
                summary = self.summarize_document(doc)
                summaries.append(summary)
            except Exception as e:
                print(f"Warning: Failed to analyze {doc.source}: {e}")
        return summaries
    
    def format_time_series_for_prompt(
        self,
        growth_data: Dict[str, "pd.DataFrame"],
        inflation_data: Dict[str, "pd.DataFrame"],
        n_recent: int = 12,
    ) -> str:
        """
        Format quantitative time series data for inclusion in prompt.
        
        Args:
            growth_data: Dict of series_id -> DataFrame with growth indicators
            inflation_data: Dict of series_id -> DataFrame with inflation indicators
            n_recent: Number of recent observations to include
            
        Returns:
            Formatted string with time series data
        """
        lines = ["## Quantitative Time Series Data\n"]
        
        # Growth series
        lines.append("### Growth Indicators (Last 12 months)\n")
        for series_id, df in growth_data.items():
            if df.empty:
                continue
            recent = df.tail(n_recent)
            lines.append(f"**{series_id}**:")
            for idx, row in recent.iterrows():
                date_str = idx.strftime("%Y-%m")
                value = row["value"]
                lines.append(f"  {date_str}: {value:.2f}")
            
            # Add YoY and MoM changes
            if len(df) >= 12:
                yoy_change = ((df["value"].iloc[-1] / df["value"].iloc[-12]) - 1) * 100
                lines.append(f"  → YoY Change: {yoy_change:+.1f}%")
            if len(df) >= 3:
                mom_3m = ((df["value"].iloc[-1] / df["value"].iloc[-3]) - 1) * 100
                mom_3m_ann = ((1 + mom_3m/100) ** 4 - 1) * 100
                lines.append(f"  → 3M Change (annualized): {mom_3m_ann:+.1f}%")
            lines.append("")
        
        # Inflation series
        lines.append("\n### Inflation Indicators (Last 12 months)\n")
        for series_id, df in inflation_data.items():
            if df.empty:
                continue
            recent = df.tail(n_recent)
            lines.append(f"**{series_id}**:")
            for idx, row in recent.iterrows():
                date_str = idx.strftime("%Y-%m")
                value = row["value"]
                lines.append(f"  {date_str}: {value:.2f}")
            
            # Add YoY inflation rate
            if len(df) >= 12:
                yoy_inflation = ((df["value"].iloc[-1] / df["value"].iloc[-12]) - 1) * 100
                lines.append(f"  → YoY Inflation Rate: {yoy_inflation:.1f}%")
            if len(df) >= 3:
                mom_3m = ((df["value"].iloc[-1] / df["value"].iloc[-3]) - 1) * 100
                mom_3m_ann = ((1 + mom_3m/100) ** 4 - 1) * 100
                lines.append(f"  → 3M Inflation (annualized): {mom_3m_ann:.1f}%")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_all_series_by_category(
        self,
        all_data: Dict[str, Dict[str, "pd.DataFrame"]],
        series_descriptions: Dict[str, Dict[str, str]],
        n_recent: int = 6,
    ) -> str:
        """
        Format ALL time series data organized by category for comprehensive prompt.
        
        Args:
            all_data: Dict[category, Dict[series_id, DataFrame]]
            series_descriptions: Dict[category, Dict[series_id, description]]
            n_recent: Number of recent observations to show
            
        Returns:
            Formatted string with all time series data
        """
        lines = ["## Complete Quantitative Data (All Available Series)\n"]
        
        category_order = [
            "growth", "inflation", "labor", "rates", 
            "financial_conditions", "housing", "money_credit", 
            "consumer_sentiment", "trade"
        ]
        
        category_titles = {
            "growth": "📈 Growth Indicators",
            "inflation": "📊 Inflation Indicators",
            "labor": "👷 Labor Market",
            "rates": "💵 Interest Rates & Yields",
            "financial_conditions": "🏦 Financial Conditions",
            "housing": "🏠 Housing Market",
            "money_credit": "💰 Money & Credit",
            "consumer_sentiment": "🛒 Consumer Sentiment",
            "trade": "🌍 Trade & Dollar",
        }
        
        for category in category_order:
            if category not in all_data or not all_data[category]:
                continue
            
            title = category_titles.get(category, category.title())
            lines.append(f"\n### {title}\n")
            
            for series_id, df in all_data[category].items():
                if df.empty:
                    continue
                
                # Get description
                desc = series_descriptions.get(category, {}).get(series_id, series_id)
                
                # Get recent values
                recent = df.tail(n_recent)
                latest_value = df["value"].iloc[-1]
                latest_date = df.index[-1].strftime("%Y-%m-%d")
                
                lines.append(f"**{series_id}** ({desc})")
                lines.append(f"  Latest: {latest_value:.2f} ({latest_date})")
                
                # Calculate changes based on data frequency
                if len(df) >= 12:
                    yoy = ((df["value"].iloc[-1] / df["value"].iloc[-12]) - 1) * 100
                    lines.append(f"  YoY Change: {yoy:+.1f}%")
                
                if len(df) >= 3:
                    mom_3 = ((df["value"].iloc[-1] / df["value"].iloc[-3]) - 1) * 100
                    lines.append(f"  3-Period Change: {mom_3:+.1f}%")
                
                if len(df) >= 1:
                    mom_1 = ((df["value"].iloc[-1] / df["value"].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
                    lines.append(f"  1-Period Change: {mom_1:+.1f}%")
                
                # Recent values (compact)
                values_str = ", ".join([f"{v:.1f}" for v in recent["value"].values[-4:]])
                lines.append(f"  Recent values: [{values_str}]")
                lines.append("")
        
        return "\n".join(lines)
    
    def _get_series_value(
        self, 
        all_series: Dict[str, "pd.DataFrame"], 
        series_id: str,
        offset: int = 0,
    ) -> Optional[float]:
        """Safely get a series value at offset from end."""
        if series_id not in all_series:
            return None
        df = all_series[series_id]
        if df.empty or len(df) <= offset:
            return None
        return float(df["value"].iloc[-(1 + offset)])
    
    def _calc_change(
        self, 
        all_series: Dict[str, "pd.DataFrame"], 
        series_id: str,
        periods: int,
    ) -> Optional[float]:
        """Calculate percent change over N periods."""
        if series_id not in all_series:
            return None
        df = all_series[series_id]
        if df.empty or len(df) <= periods:
            return None
        return ((df["value"].iloc[-1] / df["value"].iloc[-(1 + periods)]) - 1) * 100
    
    def _trend_arrow(self, change: Optional[float]) -> str:
        """Return trend arrow based on change."""
        if change is None:
            return "?"
        if change > 0.5:
            return "↑" if change < 5 else "⬆️"
        elif change < -0.5:
            return "↓" if change > -5 else "⬇️"
        return "→"
    
    def format_comprehensive_data_for_prompt(
        self,
        all_series: Dict[str, "pd.DataFrame"],
        n_recent: int = 6,
    ) -> str:
        """
        Format all available series into a well-structured, LLM-readable format.
        
        Groups related series together with context to help Gemini understand
        the relationships between indicators.
        """
        lines = []
        
        # =================================================================
        # SECTION 1: GDP & REAL ECONOMIC ACTIVITY
        # =================================================================
        lines.extend([
            "# 📈 REAL ECONOMIC ACTIVITY (GDP & Output)",
            "",
            "These indicators measure the actual output and production in the economy.",
            "**Key Question**: Is real growth accelerating or decelerating?",
            "",
        ])
        
        # GDP Group
        gdp_series = ["GDPC1", "GDP", "A191RL1Q225SBEA"]
        for sid in gdp_series:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                chg_q = self._calc_change(all_series, sid, 1)  # Quarter
                chg_y = self._calc_change(all_series, sid, 4)  # Year
                if val:
                    lines.append(f"- **{sid}** (Real GDP): {val:,.0f}")
                    if chg_q is not None:
                        lines.append(f"  - QoQ: {chg_q:+.1f}% {self._trend_arrow(chg_q)}")
                    if chg_y is not None:
                        lines.append(f"  - YoY: {chg_y:+.1f}% {self._trend_arrow(chg_y)}")
        
        # Industrial Production & Orders
        lines.append("")
        lines.append("**Production & Orders** (Monthly, more timely than GDP):")
        for sid, desc in [("INDPRO", "Industrial Production Index"), 
                          ("DGORDER", "Durable Goods Orders")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                chg_1m = self._calc_change(all_series, sid, 1)
                chg_3m = self._calc_change(all_series, sid, 3)
                chg_12m = self._calc_change(all_series, sid, 12)
                if val:
                    lines.append(f"- **{sid}** ({desc}): {val:.1f}")
                    if chg_1m is not None:
                        lines.append(f"  - MoM: {chg_1m:+.1f}% {self._trend_arrow(chg_1m)} | 3M: {chg_3m:+.1f}% | YoY: {chg_12m:+.1f}%")
        
        # Retail Sales
        if "RSAFS" in all_series:
            val = self._get_series_value(all_series, "RSAFS")
            chg_1m = self._calc_change(all_series, "RSAFS", 1)
            chg_3m = self._calc_change(all_series, "RSAFS", 3)
            chg_12m = self._calc_change(all_series, "RSAFS", 12)
            lines.append("")
            lines.append("**Consumer Spending**:")
            lines.append(f"- **RSAFS** (Retail Sales): ${val:,.0f}M")
            if chg_1m is not None:
                lines.append(f"  - MoM: {chg_1m:+.1f}% {self._trend_arrow(chg_1m)} | 3M: {chg_3m:+.1f}% | YoY: {chg_12m:+.1f}%")
        
        # =================================================================
        # SECTION 2: INFLATION (CPI vs PCE comparison)
        # =================================================================
        lines.extend([
            "",
            "---",
            "",
            "# 📊 INFLATION INDICATORS",
            "",
            "Comparing CPI (household survey) vs PCE (business survey, Fed's preferred measure).",
            "**Key Question**: Is inflation accelerating or decelerating?",
            "",
        ])
        
        # Build inflation comparison table
        lines.append("| Measure | Latest | MoM | 3M Ann. | YoY | Trend |")
        lines.append("|---------|--------|-----|---------|-----|-------|")
        
        for sid, desc in [("CPIAUCSL", "CPI All Items"), 
                          ("CPILFESL", "Core CPI"),
                          ("PCEPI", "PCE"),
                          ("PCEPILFE", "Core PCE (Fed Target)")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                chg_1m = self._calc_change(all_series, sid, 1)
                chg_3m = self._calc_change(all_series, sid, 3)
                chg_12m = self._calc_change(all_series, sid, 12)
                
                # Annualize 3-month change
                chg_3m_ann = ((1 + (chg_3m or 0)/100) ** 4 - 1) * 100 if chg_3m else None
                
                if val:
                    trend = self._trend_arrow(chg_3m)
                    mom_str = f"{chg_1m:+.2f}%" if chg_1m else "N/A"
                    ann3_str = f"{chg_3m_ann:.1f}%" if chg_3m_ann else "N/A"
                    yoy_str = f"{chg_12m:.1f}%" if chg_12m else "N/A"
                    lines.append(f"| {desc} | {val:.1f} | {mom_str} | {ann3_str} | {yoy_str} | {trend} |")
        
        # Inflation expectations
        lines.append("")
        lines.append("**Inflation Expectations** (Forward-looking):")
        for sid, desc in [("T5YIFR", "5Y5Y Forward Breakeven"),
                          ("MICH", "Michigan 1Y Expectations"),
                          ("DFII10", "10Y TIPS Yield (Real Rate)")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                prev = self._get_series_value(all_series, sid, 1)
                chg = val - prev if val and prev else None
                if val:
                    chg_str = f" ({chg:+.2f}pp)" if chg else ""
                    lines.append(f"- **{sid}** ({desc}): {val:.2f}%{chg_str}")
        
        # =================================================================
        # SECTION 3: LABOR MARKET
        # =================================================================
        lines.extend([
            "",
            "---",
            "",
            "# 👷 LABOR MARKET",
            "",
            "Leading (claims) and lagging (unemployment) indicators of labor conditions.",
            "**Key Question**: Is the labor market tightening or loosening?",
            "",
        ])
        
        # Unemployment and Participation (related)
        lines.append("**Employment Status**:")
        for sid, desc in [("UNRATE", "Unemployment Rate"),
                          ("CIVPART", "Labor Force Participation")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                prev_3m = self._get_series_value(all_series, sid, 3)
                prev_12m = self._get_series_value(all_series, sid, 12)
                if val:
                    chg_3m = val - prev_3m if prev_3m else None
                    chg_12m = val - prev_12m if prev_12m else None
                    chg_3m_str = f" (3M: {chg_3m:+.1f}pp)" if chg_3m else ""
                    chg_12m_str = f" (YoY: {chg_12m:+.1f}pp)" if chg_12m else ""
                    lines.append(f"- **{sid}** ({desc}): {val:.1f}%{chg_3m_str}{chg_12m_str}")
        
        # Payrolls
        if "PAYEMS" in all_series:
            val = self._get_series_value(all_series, "PAYEMS")
            chg_1m = self._get_series_value(all_series, "PAYEMS") - self._get_series_value(all_series, "PAYEMS", 1) if self._get_series_value(all_series, "PAYEMS", 1) else None
            chg_3m_avg = (val - self._get_series_value(all_series, "PAYEMS", 3)) / 3 if self._get_series_value(all_series, "PAYEMS", 3) else None
            lines.append("")
            lines.append("**Job Gains**:")
            if chg_1m:
                lines.append(f"- **PAYEMS** (Nonfarm Payrolls): {val:,.0f}K total")
                lines.append(f"  - Last month: {chg_1m:+,.0f}K jobs")
                if chg_3m_avg:
                    lines.append(f"  - 3-month average: {chg_3m_avg:+,.0f}K/month")
        
        # Jobless Claims (leading indicator)
        lines.append("")
        lines.append("**Jobless Claims** (Leading indicator of labor weakness):")
        for sid, desc in [("ICSA", "Initial Claims (Weekly)"),
                          ("CCSA", "Continuing Claims")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                chg_4w = self._calc_change(all_series, sid, 4)
                if val:
                    trend = self._trend_arrow(chg_4w)
                    chg_str = f" (4-wk change: {chg_4w:+.1f}%)" if chg_4w else ""
                    lines.append(f"- **{sid}** ({desc}): {val:,.0f}{chg_str} {trend}")
        
        # JOLTS
        if "JTSJOL" in all_series:
            val = self._get_series_value(all_series, "JTSJOL")
            chg_12m = self._calc_change(all_series, "JTSJOL", 12)
            lines.append("")
            lines.append("**Job Openings** (Demand for labor):")
            if val:
                chg_str = f" (YoY: {chg_12m:+.1f}%)" if chg_12m else ""
                lines.append(f"- **JTSJOL** (JOLTS Openings): {val:,.0f}K{chg_str}")
        
        # =================================================================
        # SECTION 4: INTEREST RATES & YIELD CURVE
        # =================================================================
        lines.extend([
            "",
            "---",
            "",
            "# 💵 INTEREST RATES & YIELD CURVE",
            "",
            "The yield curve shape is a key recession indicator.",
            "**Key Question**: Is the curve inverted? Are real rates rising?",
            "",
        ])
        
        # Fed Funds
        lines.append("**Federal Reserve Policy Rate**:")
        for sid in ["DFF", "FEDFUNDS"]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                if val:
                    lines.append(f"- **{sid}** (Fed Funds Rate): {val:.2f}%")
                    break
        
        # Treasury Curve (grouped together)
        lines.append("")
        lines.append("**Treasury Yield Curve**:")
        curve_data = {}
        for sid, tenor in [("DGS2", "2Y"), ("DGS10", "10Y"), ("DGS30", "30Y")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                prev = self._get_series_value(all_series, sid, 5)  # ~1 week ago
                if val:
                    curve_data[tenor] = val
                    chg = val - prev if prev else None
                    chg_str = f" ({chg:+.2f}pp vs 1wk ago)" if chg else ""
                    lines.append(f"- **{tenor} Treasury**: {val:.2f}%{chg_str}")
        
        # Spreads (calculated relationships)
        lines.append("")
        lines.append("**Yield Curve Spreads** (Inversion = Recession Signal):")
        for sid, desc in [("T10Y2Y", "10Y-2Y Spread"), ("T10Y3M", "10Y-3M Spread")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                if val:
                    status = "🔴 INVERTED" if val < 0 else "🟢 Normal"
                    lines.append(f"- **{sid}** ({desc}): {val:.2f}% → {status}")
        
        # Real Rates
        if "DFII10" in all_series:
            val = self._get_series_value(all_series, "DFII10")
            if val:
                lines.append("")
                lines.append("**Real Interest Rate**:")
                status = "Restrictive" if val > 1.5 else "Neutral" if val > 0 else "Accommodative"
                lines.append(f"- **DFII10** (10Y TIPS/Real Rate): {val:.2f}% → {status}")
        
        # Mortgage Rate
        if "MORTGAGE30US" in all_series:
            val = self._get_series_value(all_series, "MORTGAGE30US")
            chg_12m = self._calc_change(all_series, "MORTGAGE30US", 52)  # Weekly data
            if val:
                lines.append("")
                lines.append("**Mortgage Rates** (Housing affordability):")
                chg_str = f" (YoY: {chg_12m:+.1f}%)" if chg_12m else ""
                lines.append(f"- **30Y Mortgage**: {val:.2f}%{chg_str}")
        
        # =================================================================
        # SECTION 5: FINANCIAL CONDITIONS
        # =================================================================
        lines.extend([
            "",
            "---",
            "",
            "# 🏦 FINANCIAL CONDITIONS & RISK",
            "",
            "These indicators show how tight/loose financial conditions are.",
            "**Key Question**: Are financial conditions tightening or loosening?",
            "",
        ])
        
        # NFCI
        if "NFCI" in all_series:
            val = self._get_series_value(all_series, "NFCI")
            if val:
                status = "Tight" if val > 0 else "Loose"
                lines.append(f"- **NFCI** (Chicago Fed Financial Conditions): {val:.2f} → {status}")
                lines.append("  *(Positive = Tight, Negative = Loose, 0 = Average)*")
        
        # Credit Spreads
        if "BAMLH0A0HYM2" in all_series:
            val = self._get_series_value(all_series, "BAMLH0A0HYM2")
            chg = self._calc_change(all_series, "BAMLH0A0HYM2", 20)
            if val:
                stress = "Elevated Stress" if val > 5 else "Normal" if val > 3 else "Tight Spreads"
                chg_str = f" (1M chg: {chg:+.0f}bp)" if chg else ""
                lines.append(f"- **HY Credit Spread**: {val:.0f}bp{chg_str} → {stress}")
        
        # VIX
        if "VIXCLS" in all_series:
            val = self._get_series_value(all_series, "VIXCLS")
            if val:
                regime = "🔴 High Fear" if val > 25 else "🟡 Elevated" if val > 18 else "🟢 Calm"
                lines.append(f"- **VIX** (Volatility Index): {val:.1f} → {regime}")
        
        # S&P 500
        if "SP500" in all_series:
            val = self._get_series_value(all_series, "SP500")
            chg_1m = self._calc_change(all_series, "SP500", 20)
            chg_ytd = self._calc_change(all_series, "SP500", 252)
            if val:
                lines.append("")
                lines.append("**Equity Market**:")
                chg_str = f" (1M: {chg_1m:+.1f}%)" if chg_1m else ""
                ytd_str = f" (YTD: {chg_ytd:+.1f}%)" if chg_ytd else ""
                lines.append(f"- **S&P 500**: {val:,.0f}{chg_str}{ytd_str}")
        
        # Dollar
        if "DTWEXBGS" in all_series:
            val = self._get_series_value(all_series, "DTWEXBGS")
            chg_3m = self._calc_change(all_series, "DTWEXBGS", 60)
            if val:
                trend = "Strengthening" if chg_3m and chg_3m > 0 else "Weakening"
                lines.append(f"- **Dollar Index** (Trade-weighted): {val:.1f} → {trend}")
        
        # =================================================================
        # SECTION 6: HOUSING MARKET
        # =================================================================
        lines.extend([
            "",
            "---",
            "",
            "# 🏠 HOUSING MARKET",
            "",
            "Housing is interest-rate sensitive and a leading economic indicator.",
            "",
        ])
        
        for sid, desc in [("HOUST", "Housing Starts"),
                          ("PERMIT", "Building Permits"),
                          ("CSUSHPISA", "Case-Shiller Home Prices")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                chg_3m = self._calc_change(all_series, sid, 3)
                chg_12m = self._calc_change(all_series, sid, 12)
                if val:
                    lines.append(f"- **{sid}** ({desc}): {val:,.0f}")
                    if chg_3m is not None and chg_12m is not None:
                        lines.append(f"  - 3M: {chg_3m:+.1f}% | YoY: {chg_12m:+.1f}%")
        
        # =================================================================
        # SECTION 7: MONEY & LIQUIDITY
        # =================================================================
        lines.extend([
            "",
            "---",
            "",
            "# 💰 MONEY SUPPLY & LIQUIDITY",
            "",
            "Monetary aggregates and Fed balance sheet indicate liquidity conditions.",
            "",
        ])
        
        for sid, desc in [("M2SL", "M2 Money Supply"),
                          ("WALCL", "Fed Balance Sheet"),
                          ("TOTRESNS", "Bank Reserves")]:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                chg_12m = self._calc_change(all_series, sid, 12)
                if val:
                    chg_str = f" (YoY: {chg_12m:+.1f}%)" if chg_12m else ""
                    lines.append(f"- **{sid}** ({desc}): ${val/1000:,.1f}T{chg_str}")
        
        # =================================================================
        # SECTION 8: CONSUMER SENTIMENT
        # =================================================================
        lines.extend([
            "",
            "---",
            "",
            "# 🛒 CONSUMER SENTIMENT",
            "",
            "Forward-looking consumer expectations.",
            "",
        ])
        
        if "UMCSENT" in all_series:
            val = self._get_series_value(all_series, "UMCSENT")
            chg_3m = self._calc_change(all_series, "UMCSENT", 3)
            if val:
                level = "Pessimistic" if val < 70 else "Neutral" if val < 85 else "Optimistic"
                chg_str = f" (3M: {chg_3m:+.1f}%)" if chg_3m else ""
                lines.append(f"- **U of Michigan Sentiment**: {val:.1f}{chg_str} → {level}")
        
        # =================================================================
        # SUMMARY BOX
        # =================================================================
        lines.extend([
            "",
            "---",
            "",
            "# 📋 QUICK REFERENCE SUMMARY",
            "",
            "| Category | Key Indicator | Current | Trend |",
            "|----------|--------------|---------|-------|",
        ])
        
        # Build summary table
        summary_items = [
            ("Growth", "INDPRO", "Industrial Production"),
            ("Inflation", "PCEPILFE", "Core PCE"),
            ("Labor", "UNRATE", "Unemployment"),
            ("Rates", "DGS10", "10Y Treasury"),
            ("Financial", "NFCI", "Financial Conditions"),
            ("Housing", "HOUST", "Housing Starts"),
        ]
        
        for cat, sid, desc in summary_items:
            if sid in all_series:
                val = self._get_series_value(all_series, sid)
                chg = self._calc_change(all_series, sid, 3)
                if val:
                    trend = self._trend_arrow(chg)
                    lines.append(f"| {cat} | {desc} | {val:.1f} | {trend} |")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def format_document_summaries_for_prompt(
        self,
        summaries: List[DocumentSummary],
    ) -> str:
        """
        Format Gemini's document summaries for the final probability prompt.
        """
        lines = ["## Qualitative Document Analysis (Gemini Summaries)\n"]
        
        for summary in summaries:
            lines.extend([
                f"### {summary.source} ({summary.date.strftime('%Y-%m-%d')})",
                f"- **Growth Assessment**: {summary.growth_assessment.upper()}",
                f"- **Inflation Assessment**: {summary.inflation_assessment.upper()}",
                f"- **Key Points**:",
            ])
            for point in summary.key_points:
                lines.append(f"  - {point}")
            if summary.risks:
                lines.append(f"- **Risks**:")
                for risk in summary.risks:
                    lines.append(f"  - {risk}")
            lines.append(f"- **Summary**: {summary.raw_summary}")
            lines.append("")
        
        return "\n".join(lines)
    
    def predict_quadrant_with_full_context(
        self,
        growth_data: Dict[str, "pd.DataFrame"],
        inflation_data: Dict[str, "pd.DataFrame"],
        document_summaries: List[DocumentSummary],
        roc_summary: str,
        gdpnow_estimate: Optional[float] = None,
    ) -> QuadrantProbabilities:
        """
        Enhanced quadrant prediction using full time series + Gemini doc summaries.
        
        This is the TWO-STAGE approach:
        1. First, each document is analyzed by Gemini (done before calling this)
        2. Then, all structured data is combined into one comprehensive prompt
        
        Args:
            growth_data: Dict of growth time series DataFrames
            inflation_data: Dict of inflation time series DataFrames
            document_summaries: Pre-analyzed document summaries from Gemini
            roc_summary: Text summary of Rate of Change calculations
            gdpnow_estimate: Current Atlanta Fed GDPNow estimate
            
        Returns:
            QuadrantProbabilities with probability distribution
        """
        # Format all the data
        time_series_text = self.format_time_series_for_prompt(growth_data, inflation_data)
        doc_summaries_text = self.format_document_summaries_for_prompt(document_summaries)
        
        gdpnow_text = ""
        if gdpnow_estimate is not None:
            gdpnow_text = f"\n## Atlanta Fed GDPNow\nCurrent Real GDP Growth Estimate: **{gdpnow_estimate}%**\n"
        
        prompt = f"""{self.SYSTEM_PROMPT}

You are now making the FINAL quadrant probability assessment based on comprehensive data.

{time_series_text}

{roc_summary}

{gdpnow_text}

{doc_summaries_text}

---

## Your Task

Based on ALL the above information:
1. The raw quantitative time series data showing actual values and trends
2. The Rate of Change analysis showing acceleration/deceleration
3. The Atlanta Fed GDPNow real-time GDP estimate
4. The structured summaries of Federal Reserve qualitative documents

Estimate the probability of each Hedgeye quadrant for the NEXT MONTH (30-day forward).

Key considerations:
- Look for CONSISTENCY or DIVERGENCE between quantitative and qualitative signals
- The Rate of Change (2nd derivative) matters more than absolute levels
- Forward-looking language in Fed documents often leads the data
- Consider which quadrant transition risks are elevated

Provide your response in JSON format:
{{
    "quad1_probability": 0.XX,
    "quad2_probability": 0.XX,
    "quad3_probability": 0.XX,
    "quad4_probability": 0.XX,
    "reasoning": "Detailed explanation synthesizing quant and qual signals",
    "confidence": 0.XX,
    "quant_qual_alignment": "aligned" | "mixed" | "divergent",
    "key_factors": ["factor 1", "factor 2", "factor 3"],
    "transition_risks": ["risk of moving to quad X because..."]
}}

The probabilities MUST sum to 1.0. Return ONLY valid JSON.
"""
        
        response = self._call_gemini(prompt, call_type="predict_quadrant_full")
        
        try:
            data = self._clean_and_parse_json(response)
            
            # Normalize probabilities
            total = (
                data.get("quad1_probability", 0.25) +
                data.get("quad2_probability", 0.25) +
                data.get("quad3_probability", 0.25) +
                data.get("quad4_probability", 0.25)
            )
            
            reasoning = data.get("reasoning", "")
            # Append additional context to reasoning
            if data.get("quant_qual_alignment"):
                reasoning += f"\n\nQuant/Qual Alignment: {data['quant_qual_alignment']}"
            if data.get("key_factors"):
                reasoning += f"\n\nKey Factors: {', '.join(data['key_factors'])}"
            if data.get("transition_risks"):
                reasoning += f"\n\nTransition Risks: {'; '.join(data['transition_risks'])}"
            
            return QuadrantProbabilities(
                quad1=data.get("quad1_probability", 0.25) / total,
                quad2=data.get("quad2_probability", 0.25) / total,
                quad3=data.get("quad3_probability", 0.25) / total,
                quad4=data.get("quad4_probability", 0.25) / total,
                reasoning=reasoning,
                confidence=data.get("confidence", 0.5),
                as_of_date=datetime.now(),
            )
            
        except json.JSONDecodeError:
            return QuadrantProbabilities(
                quad1=0.25,
                quad2=0.25,
                quad3=0.25,
                quad4=0.25,
                reasoning=f"Failed to parse response: {response[:500]}",
                confidence=0.0,
                as_of_date=datetime.now(),
            )
    
    def generate_market_implications(
        self,
        probabilities: QuadrantProbabilities,
        quadrant_assets: Dict[int, Dict[str, str]],
    ) -> str:
        """Generate market implications based on quadrant probabilities."""
        prompt = f"""{self.SYSTEM_PROMPT}

Based on the following quadrant probability distribution, provide actionable 
market implications and asset allocation recommendations.

## Quadrant Probabilities (Next Month)
- Quad 1 (Growth ↑, Inflation ↓): {probabilities.quad1:.1%}
- Quad 2 (Growth ↑, Inflation ↑): {probabilities.quad2:.1%}
- Quad 3 (Growth ↓, Inflation ↓): {probabilities.quad3:.1%}
- Quad 4 (Growth ↓, Inflation ↑): {probabilities.quad4:.1%}

Most Likely: Quad {probabilities.most_likely()}
Confidence: {probabilities.confidence:.0%}

## Reasoning
{probabilities.reasoning}

## Standard Asset Preferences by Quadrant
{json.dumps(quadrant_assets, indent=2)}

---

Provide a concise market outlook with:
1. Primary positioning recommendation
2. Key assets to favor
3. Key assets to avoid
4. Risk factors to monitor
5. Potential catalysts for quadrant transitions

Keep the response practical and actionable for a portfolio manager.
"""
        
        return self._call_gemini(prompt, call_type="market_implications")
    
    def generate_portfolio_recommendations(
        self,
        probabilities: QuadrantProbabilities,
        quadrant_assets: Dict[int, Dict[str, str]],
    ) -> PortfolioRecommendations:
        """
        Generate structured portfolio construction recommendations.
        
        Returns specific longs and shorts with conviction levels and rationale.
        
        Args:
            probabilities: Quadrant probability distribution
            quadrant_assets: Standard asset preferences by quadrant
            
        Returns:
            PortfolioRecommendations with structured longs/shorts
        """
        prompt = f"""{self.SYSTEM_PROMPT}

Based on the following quadrant probability distribution, provide SPECIFIC and STRUCTURED
portfolio construction recommendations.

## Quadrant Probabilities (Next Month)
- Quad 1 (Growth ↑, Inflation ↓): {probabilities.quad1:.1%}
- Quad 2 (Growth ↑, Inflation ↑): {probabilities.quad2:.1%}
- Quad 3 (Growth ↓, Inflation ↓): {probabilities.quad3:.1%}
- Quad 4 (Growth ↓, Inflation ↑): {probabilities.quad4:.1%}

Most Likely: Quad {probabilities.most_likely()}
Confidence: {probabilities.confidence:.0%}

## Reasoning
{probabilities.reasoning}

## Standard Asset Preferences by Quadrant
{json.dumps(quadrant_assets, indent=2)}

---

## YOUR TASK

Provide structured portfolio recommendations in JSON format. Be SPECIFIC with asset names
(use actual ticker symbols or ETF names when possible).

Return ONLY valid JSON in this exact format:
{{
    "longs": [
        {{"asset": "GLD", "asset_name": "Gold ETF", "rationale": "Stagflation hedge, inflation protection", "conviction": "high", "target_weight": 10}},
        {{"asset": "TLT", "asset_name": "20+ Year Treasury ETF", "rationale": "Duration exposure for rate cuts", "conviction": "medium", "target_weight": 5}}
    ],
    "shorts": [
        {{"asset": "QQQ", "asset_name": "NASDAQ 100 ETF", "rationale": "Growth stocks vulnerable in Quad 4", "conviction": "high", "target_weight": 5}},
        {{"asset": "HYG", "asset_name": "High Yield Corporate Bond ETF", "rationale": "Credit spread widening risk", "conviction": "medium", "target_weight": 3}}
    ],
    "sector_overweights": ["Energy", "Utilities", "Consumer Staples", "Healthcare"],
    "sector_underweights": ["Technology", "Consumer Discretionary", "Financials"],
    "recommended_cash_allocation": 0.25,
    "risk_level": "defensive",
    "recommended_hedges": ["Long VIX calls", "Put spreads on SPY", "Gold allocation"],
    "time_horizon": "30-day",
    "confidence": 0.7,
    "rationale": "Given 40% probability of Quad 4 stagflation with significant transition risk, positioning should be defensive with inflation hedges..."
}}

IMPORTANT:
- For "conviction", use: "high", "medium", or "low"
- For "target_weight", provide suggested portfolio weight as percentage (0-100)
- For "risk_level", use: "defensive", "neutral", or "aggressive"
- Include 3-6 longs and 2-4 shorts
- Be specific with real tradeable assets (ETFs preferred for clarity)
- Consider transition risks between quadrants in your recommendations
- Provide hedges appropriate for the uncertainty level

Return ONLY the JSON, no markdown formatting.
"""
        
        response = self._call_gemini(prompt, call_type="portfolio_recommendations")
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response.strip())
            
            return PortfolioRecommendations(
                longs=data.get("longs", []),
                shorts=data.get("shorts", []),
                sector_overweights=data.get("sector_overweights", []),
                sector_underweights=data.get("sector_underweights", []),
                recommended_cash_allocation=data.get("recommended_cash_allocation", 0.1),
                risk_level=data.get("risk_level", "neutral"),
                recommended_hedges=data.get("recommended_hedges", []),
                time_horizon=data.get("time_horizon", "30-day"),
                confidence=data.get("confidence", 0.5),
                rationale=data.get("rationale", ""),
            )
            
        except Exception as e:
            print(f"Warning: Failed to parse portfolio recommendations: {e}")
            # Return default defensive recommendations
            return PortfolioRecommendations(
                longs=[
                    {"asset": "GLD", "asset_name": "Gold ETF", "rationale": "Default defensive position", "conviction": "medium", "target_weight": 5},
                    {"asset": "SHY", "asset_name": "1-3 Year Treasury ETF", "rationale": "Capital preservation", "conviction": "medium", "target_weight": 10},
                ],
                shorts=[],
                sector_overweights=["Utilities", "Consumer Staples"],
                sector_underweights=["Technology", "Consumer Discretionary"],
                recommended_cash_allocation=0.2,
                risk_level="defensive",
                recommended_hedges=["Maintain cash buffer"],
                time_horizon="30-day",
                confidence=0.3,
                rationale=f"Default recommendations due to parsing error: {str(e)[:100]}",
            )
    
    def format_market_pricing_for_prompt(
        self,
        categorized_indices: Dict[str, List],
        economic_calendar: List,
    ) -> str:
        """
        Format FMP market data showing what markets are currently pricing.
        
        This helps Gemini understand market sentiment and expectations.
        
        Args:
            categorized_indices: Dict from FMPDataLoader.load_categorized_indices()
            economic_calendar: List from FMPDataLoader.load_economic_calendar()
            
        Returns:
            Formatted markdown string
        """
        lines = [
            "",
            "---",
            "",
            "# 📈 REAL-TIME MARKET PRICING",
            "",
            "This section shows what financial markets are currently pricing in.",
            "Market prices are forward-looking and often lead economic data.",
            "",
        ]
        
        # =================================================================
        # VOLATILITY / FEAR GAUGE
        # =================================================================
        if "volatility" in categorized_indices and categorized_indices["volatility"]:
            lines.extend([
                "## 😨 VOLATILITY & FEAR GAUGE",
                "",
                "VIX term structure shows market's volatility expectations over time.",
                "- **Contango** (VIX < VIX3M): Normal, complacent market",
                "- **Backwardation** (VIX > VIX3M): Stressed, expecting near-term vol",
                "",
            ])
            
            vix_data = {}
            vvix = None
            move = None
            
            for q in categorized_indices["volatility"]:
                symbol = q.symbol if hasattr(q, 'symbol') else q.get("symbol", "")
                price = q.price if hasattr(q, 'price') else q.get("price", 0)
                change = q.change if hasattr(q, 'change') else q.get("change", 0)
                
                if symbol == "^VIX":
                    vix_data["30D"] = (price, change)
                elif symbol == "^VIX1D":
                    vix_data["1D"] = (price, change)
                elif symbol == "^VIX3M":
                    vix_data["3M"] = (price, change)
                elif symbol == "^VIX6M":
                    vix_data["6M"] = (price, change)
                elif symbol == "^VVIX":
                    vvix = (price, change)
            
            # Check for MOVE in other category
            if "other" in categorized_indices:
                for q in categorized_indices["other"]:
                    symbol = q.symbol if hasattr(q, 'symbol') else q.get("symbol", "")
                    if symbol == "^MOVE":
                        price = q.price if hasattr(q, 'price') else q.get("price", 0)
                        change = q.change if hasattr(q, 'change') else q.get("change", 0)
                        move = (price, change)
            
            # VIX Term Structure Table
            lines.append("| Tenor | VIX Level | Change | Interpretation |")
            lines.append("|-------|-----------|--------|----------------|")
            
            for tenor in ["1D", "30D", "3M", "6M"]:
                if tenor in vix_data:
                    level, chg = vix_data[tenor]
                    if level < 15:
                        interp = "🟢 Low fear"
                    elif level < 20:
                        interp = "🟡 Moderate"
                    elif level < 25:
                        interp = "🟠 Elevated"
                    else:
                        interp = "🔴 High fear"
                    lines.append(f"| {tenor} | {level:.1f} | {chg:+.2f} | {interp} |")
            
            # Term structure interpretation
            if "30D" in vix_data and "3M" in vix_data:
                vix_spot = vix_data["30D"][0]
                vix_3m = vix_data["3M"][0]
                if vix_spot > vix_3m:
                    lines.append("")
                    lines.append("⚠️ **VIX in BACKWARDATION** - Market expects near-term stress")
                else:
                    lines.append("")
                    lines.append("✅ **VIX in Contango** - Normal term structure")
            
            # VVIX (volatility of volatility)
            if vvix:
                lines.append("")
                lines.append(f"**VVIX** (Vol of VIX): {vvix[0]:.1f} ({vvix[1]:+.2f})")
                if vvix[0] > 120:
                    lines.append("  → High uncertainty about volatility direction")
            
            # MOVE (bond volatility)
            if move:
                lines.append("")
                lines.append(f"**MOVE Index** (Bond Volatility): {move[0]:.1f} ({move[1]:+.2f})")
                if move[0] > 100:
                    lines.append("  → Elevated bond market stress")
                elif move[0] < 80:
                    lines.append("  → Calm fixed income markets")
            
            lines.append("")
        
        # =================================================================
        # US EQUITY INDICES
        # =================================================================
        if "us_major" in categorized_indices and categorized_indices["us_major"]:
            lines.extend([
                "## 🇺🇸 US EQUITY INDICES",
                "",
                "| Index | Level | Day Change | % Change |",
                "|-------|-------|------------|----------|",
            ])
            
            index_map = {
                "^GSPC": "S&P 500",
                "^DJI": "Dow Jones",
                "^IXIC": "NASDAQ Comp",
                "^NDX": "NASDAQ 100",
                "^RUT": "Russell 2000",
            }
            
            for q in categorized_indices["us_major"]:
                symbol = q.symbol if hasattr(q, 'symbol') else q.get("symbol", "")
                if symbol in index_map:
                    price = q.price if hasattr(q, 'price') else q.get("price", 0)
                    change = q.change if hasattr(q, 'change') else q.get("change", 0)
                    pct = (change / (price - change) * 100) if (price - change) else 0
                    
                    name = index_map[symbol]
                    lines.append(f"| {name} | {price:,.0f} | {change:+,.0f} | {pct:+.2f}% |")
            
            lines.append("")
            
            # Risk-on/off signal from Russell vs S&P
            spx = next((q for q in categorized_indices["us_major"] 
                       if (q.symbol if hasattr(q, 'symbol') else q.get("symbol")) == "^GSPC"), None)
            rut = next((q for q in categorized_indices["us_major"] 
                       if (q.symbol if hasattr(q, 'symbol') else q.get("symbol")) == "^RUT"), None)
            
            if spx and rut:
                spx_pct = (spx.change if hasattr(spx, 'change') else spx.get("change", 0)) / \
                          ((spx.price if hasattr(spx, 'price') else spx.get("price", 1)) - 
                           (spx.change if hasattr(spx, 'change') else spx.get("change", 0))) * 100
                rut_pct = (rut.change if hasattr(rut, 'change') else rut.get("change", 0)) / \
                          ((rut.price if hasattr(rut, 'price') else rut.get("price", 1)) - 
                           (rut.change if hasattr(rut, 'change') else rut.get("change", 0))) * 100
                
                if rut_pct > spx_pct + 0.5:
                    lines.append("📈 **Risk-On Signal**: Small caps outperforming large caps")
                elif rut_pct < spx_pct - 0.5:
                    lines.append("📉 **Risk-Off Signal**: Large caps outperforming small caps")
            
            lines.append("")
        
        # =================================================================
        # TREASURY YIELDS (Market's Rate Expectations)
        # =================================================================
        if "treasury" in categorized_indices and categorized_indices["treasury"]:
            lines.extend([
                "## 💵 TREASURY YIELDS (Market Rate Expectations)",
                "",
                "Treasury yields show what the market expects for growth and inflation.",
                "",
                "| Maturity | Yield | Change (bp) |",
                "|----------|-------|-------------|",
            ])
            
            tenor_map = {
                "^IRX": ("3-Month", 0.25),
                "^FVX": ("5-Year", 5),
                "^TNX": ("10-Year", 10),
                "^TYX": ("30-Year", 30),
            }
            
            yields = {}
            for q in categorized_indices["treasury"]:
                symbol = q.symbol if hasattr(q, 'symbol') else q.get("symbol", "")
                if symbol in tenor_map:
                    price = q.price if hasattr(q, 'price') else q.get("price", 0)
                    change = q.change if hasattr(q, 'change') else q.get("change", 0)
                    
                    name, years = tenor_map[symbol]
                    yields[years] = price
                    bp_change = change * 100  # Convert to basis points
                    lines.append(f"| {name} | {price:.3f}% | {bp_change:+.1f} bp |")
            
            lines.append("")
            
            # Calculate yield curve metrics
            if 10 in yields and 0.25 in yields:
                spread_10y3m = yields[10] - yields[0.25]
                if spread_10y3m < 0:
                    lines.append(f"🔴 **10Y-3M Spread: {spread_10y3m*100:.0f} bp (INVERTED)**")
                    lines.append("  → Yield curve inversion historically signals recession 12-18 months ahead")
                else:
                    lines.append(f"🟢 **10Y-3M Spread: {spread_10y3m*100:.0f} bp (Normal)**")
            
            if 10 in yields and 5 in yields:
                spread_10y5y = yields[10] - yields[5]
                if spread_10y5y < 0:
                    lines.append(f"  → 10Y-5Y: {spread_10y5y*100:.0f} bp (inverted)")
            
            lines.append("")
        
        # =================================================================
        # GLOBAL INDICES
        # =================================================================
        if "global" in categorized_indices and categorized_indices["global"]:
            lines.extend([
                "## 🌍 GLOBAL EQUITY INDICES",
                "",
            ])
            
            global_map = {
                "^GDAXI": "🇩🇪 DAX (Germany)",
                "^FTSE": "🇬🇧 FTSE 100 (UK)",
                "^STOXX50E": "🇪🇺 Euro Stoxx 50",
            }
            
            for q in categorized_indices["global"]:
                symbol = q.symbol if hasattr(q, 'symbol') else q.get("symbol", "")
                if symbol in global_map:
                    price = q.price if hasattr(q, 'price') else q.get("price", 0)
                    change = q.change if hasattr(q, 'change') else q.get("change", 0)
                    pct = (change / (price - change) * 100) if (price - change) else 0
                    
                    name = global_map[symbol]
                    lines.append(f"- **{name}**: {price:,.0f} ({pct:+.2f}%)")
            
            lines.append("")
        
        # =================================================================
        # UPCOMING ECONOMIC EVENTS
        # =================================================================
        if economic_calendar:
            lines.extend([
                "## 📅 UPCOMING US ECONOMIC RELEASES",
                "",
                "These are the key data releases that could move markets.",
                "",
                "| Date | Event | Previous | Estimate | Impact |",
                "|------|-------|----------|----------|--------|",
            ])
            
            now = datetime.now()
            upcoming = [e for e in economic_calendar if e.date > now][:10]  # Next 10 events
            
            for event in upcoming:
                date_str = event.date.strftime("%m/%d %H:%M")
                prev = f"{event.previous}{event.unit or ''}" if event.previous else "N/A"
                est = f"{event.estimate}{event.unit or ''}" if event.estimate else "N/A"
                impact_emoji = "🔴" if event.impact == "High" else "🟡"
                lines.append(f"| {date_str} | {event.event} | {prev} | {est} | {impact_emoji} {event.impact} |")
            
            lines.append("")
            
            # Recent releases (surprises matter)
            lines.append("**Recent Releases (vs Expectations)**:")
            recent = [e for e in economic_calendar if e.date <= now and e.actual is not None][:5]
            
            for event in recent:
                if event.estimate and event.actual:
                    surprise = event.actual - event.estimate
                    surprise_pct = (surprise / event.estimate * 100) if event.estimate else 0
                    emoji = "📈" if surprise > 0 else "📉" if surprise < 0 else "➡️"
                    lines.append(f"  - {event.event}: {event.actual}{event.unit or ''} vs est {event.estimate}{event.unit or ''} {emoji} ({surprise:+.1f})")
            
            lines.append("")
        
        # =================================================================
        # MARKET SENTIMENT SUMMARY
        # =================================================================
        lines.extend([
            "## 📊 MARKET SENTIMENT SUMMARY",
            "",
        ])
        
        # Aggregate signals
        signals = []
        
        # VIX signal
        if "volatility" in categorized_indices:
            vix = next((q for q in categorized_indices["volatility"] 
                       if (q.symbol if hasattr(q, 'symbol') else q.get("symbol")) == "^VIX"), None)
            if vix:
                vix_level = vix.price if hasattr(vix, 'price') else vix.get("price", 0)
                if vix_level < 15:
                    signals.append("VIX: 🟢 Low fear (complacent)")
                elif vix_level < 20:
                    signals.append("VIX: 🟡 Moderate caution")
                elif vix_level < 25:
                    signals.append("VIX: 🟠 Elevated fear")
                else:
                    signals.append("VIX: 🔴 High fear (panic)")
        
        # Equity signal
        if "us_major" in categorized_indices:
            spx = next((q for q in categorized_indices["us_major"] 
                       if (q.symbol if hasattr(q, 'symbol') else q.get("symbol")) == "^GSPC"), None)
            if spx:
                spx_chg = spx.change if hasattr(spx, 'change') else spx.get("change", 0)
                spx_price = spx.price if hasattr(spx, 'price') else spx.get("price", 1)
                spx_pct = spx_chg / (spx_price - spx_chg) * 100 if (spx_price - spx_chg) else 0
                if spx_pct > 1:
                    signals.append(f"S&P 500: 🟢 Strong rally (+{spx_pct:.1f}%)")
                elif spx_pct > 0:
                    signals.append(f"S&P 500: 🟢 Positive (+{spx_pct:.1f}%)")
                elif spx_pct > -1:
                    signals.append(f"S&P 500: 🟡 Slightly negative ({spx_pct:.1f}%)")
                else:
                    signals.append(f"S&P 500: 🔴 Sell-off ({spx_pct:.1f}%)")
        
        for sig in signals:
            lines.append(f"- {sig}")
        
        lines.append("")
        
        return "\n".join(lines)

